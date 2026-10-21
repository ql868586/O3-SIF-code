# -*- coding: utf-8 -*-
# 纯 Python：复制整个文件到 Jupyter 代码单元格；不要复制 ipynb 的 JSON。
import os, gc, json, time, sys, subprocess, importlib.util
# 单元格内自动检查依赖：不需要粘贴 Notebook JSON 或执行命令行
DEPENDENCIES = {'numpy':'numpy', 'pandas':'pandas', 'scipy':'scipy', 'xgboost':'xgboost',
                'duckdb':'duckdb', 'pyarrow':'pyarrow', 'matplotlib':'matplotlib'}
missing = [pipname for module, pipname in DEPENDENCIES.items()
           if importlib.util.find_spec(module) is None]
if missing:
    print('安装缺少的依赖:', ', '.join(missing), flush=True)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
os.environ.setdefault('MKL_NUM_THREADS','8')
import numpy as np
import pandas as pd
from scipy import stats
import xgboost as xgb

# ================== 只修改此处 ==================
RAW = Path('/root/autodl-tmp/wq/matched_data_albers_all1.csv')
OUT = Path('/root/autodl-tmp/wql/1020_radiation_FULL_FRESH_ONECELL')
YEARS = range(2000, 2023)
ACTIVE_MONTHS = (4, 5, 6, 7, 8, 9)
DEVICE = 'cuda'               # 无 CUDA GPU 改成 'cpu'
N_JOBS = 8
RESUME = True                # 同一版本中断续算，不读旧研究目录
SSRD_UNIT = 'native'         # 若确定是日平均累计 J/m2/day，可选 j_m2_per_day
DUCKDB_GB = 12
BOOTSTRAPS = 1999
# =============================================
CACHE = OUT / 'year_month_pixel_cache_v6'
RESULT = OUT / 'radiation_extension'
SEED, FOLDS, SP_FACTOR = 42, 5, 20
M_A = ['DEM','lccs','t2m','ssrd','tp','u10','v10','sp','stl1','swvl1']
METEO = ['u10','v10','tp','sp']
HP = {
 'treatment':dict(n_estimators=1053,max_depth=9,learning_rate=0.10216433662775991,min_child_weight=25,gamma=4.309033817797793,subsample=0.8908973518801342,colsample_bytree=0.8492767336660302,reg_alpha=0.003354690482344186,reg_lambda=2.033716357895456),
 'outcome':dict(n_estimators=1131,max_depth=10,learning_rate=0.0266047322246523,min_child_weight=3,gamma=0.5110259054946181,subsample=0.7833701909759339,colsample_bytree=0.9168109399871214,reg_alpha=0.007222709294055777,reg_lambda=1.912111490897236)
}
SCRIPT_VERSION = 'O3SIF-fresh-DML-radiation-v6-20261018'

# ---------- 结构优化新增常量（不改变原有行为） ----------
COND_MAX = 1e12
SPATIAL_ID_MULTIPLIER = 10_000_000
ANALYSIS_COLUMNS = [
    'year', 'month', 'spatial_block', 'n_years', 'year_mask',
    'A', 'B', 'o3_anom', 'temp_anom', 'r_sd', 'tr_within', 'tr',
    *[m + '_z' for m in METEO],
]


# ============================================================
# A. 环境 / 目录 / 签名
# ============================================================
def _validate_environment():
    """校验 XGBoost 版本与磁盘空间。"""
    import duckdb, pyarrow, shutil  # noqa: F401  (保留原依赖探测)
    v = tuple(int(x) for x in xgb.__version__.split('.')[:2])
    if v < (2, 0):
        raise RuntimeError('XGBoost >= 2.0 required for device=cuda; use: %pip install -U xgboost')
    free_gb = shutil.disk_usage(OUT.parent if OUT.parent.exists() else RAW.parent).free / 2**30
    if free_gb < 20:
        print(f'警告：可用空间只有 {free_gb:.1f} GiB；可能不足以保存年度缓存和数据库')


def _read_header_and_validate():
    """读取原始 CSV 的列头并校验所需字段。"""
    head = list(pd.read_csv(RAW, nrows=0).columns)
    required = {'SIF', 'O3', 'year', 'month', 'x', 'y', 't2m', 'ssrd', *METEO}
    missing = required - set(head)
    if missing:
        raise ValueError(f'原始 CSV 缺少列：{sorted(missing)}')
    return head


def _ensure_signature():
    """创建输出目录并写入 / 校验输入签名。"""
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULT.mkdir(parents=True, exist_ok=True)
    signature = dict(
        version=SCRIPT_VERSION,
        raw_path=str(RAW.resolve()),
        raw_size=RAW.stat().st_size,
        raw_time=RAW.stat().st_mtime_ns,
        years=[min(YEARS), max(YEARS)],
        model=HP,
        folds=FOLDS,
        block_factor=SP_FACTOR,
        device=DEVICE,
        ssrd_unit=SSRD_UNIT,
        analysis_months=list(ACTIVE_MONTHS),
        control_model=M_A,
    )
    sfile = OUT / 'FRESH_INPUT_SIGNATURE.json'
    if sfile.exists():
        if json.loads(sfile.read_text(encoding='utf-8')) != signature:
            raise RuntimeError('输出目录有其他输入或旧版本缓存，请设置新的 OUT 路径；不要覆盖。')
    else:
        if any(CACHE.glob('year_*_month_pixel.parquet')):
            raise RuntimeError('输出目录有未验证旧缓存，请设置新的 OUT 路径。')
        sfile.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding='utf-8')


def setup():
    """准备环境、目录与输入签名。"""
    if not RAW.is_file():
        raise FileNotFoundError(f'找不到原始 CSV: {RAW}')
    if SSRD_UNIT not in {'native', 'w_m2', 'j_m2_per_day'}:
        raise ValueError('SSRD_UNIT 设为 native、w_m2 或 j_m2_per_day')
    _validate_environment()
    _read_header_and_validate()
    _ensure_signature()
    print('输入文件存在；输出目录：', OUT)


# ============================================================
# B. 原始数据读取与清理
# ============================================================
def _resolve_lccs(header):
    """确定 LCCS 列名（大小写不敏感）；返回 (lccs_col, controls_list)。"""
    lccs = 'lccs' if 'lccs' in header else ('LCCS' if 'LCCS' in header else None)
    controls = [c for c in M_A if c in header or (c == 'lccs' and lccs)]
    return lccs, controls


def _parse_month_column(df):
    """把 month 列原地解析为 1—12 的 int16。"""
    if not pd.api.types.is_numeric_dtype(df['month']):
        df['month'] = pd.to_numeric(
            df['month'].astype(str).str.extract(r'(\d{1,2})$')[0],
            errors='coerce',
        )
    df['month'] = pd.to_numeric(df['month'], errors='coerce')
    if df['month'].isna().any() or not df['month'].between(1, 12).all():
        raise ValueError('month 列不能正确解析为 1—12')
    df['month'] = df['month'].astype('int16')


def _read_selected_columns(header):
    """读取需要的列并做类型转换，返回 (df, lccs_col)。"""
    lccs, controls = _resolve_lccs(header)
    used = list(dict.fromkeys(
        ['SIF', 'O3', 'year', 'month', 'x', 'y']
        + [lccs if c == 'lccs' else c for c in controls]
    ))
    print('读取原始 CSV 中的列：', used)
    df = pd.read_csv(RAW, usecols=used, low_memory=False)
    if lccs == 'LCCS':
        df.rename(columns={'LCCS': 'lccs'}, inplace=True)

    df['year'] = pd.to_numeric(df['year'], errors='raise').astype('int16')
    _parse_month_column(df)

    df = df.loc[df['year'].isin(YEARS)].copy()
    if df.empty:
        raise RuntimeError('研究年份 2000—2022 没有数据')

    for c in ['x', 'y']:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
        if not np.isfinite(df[c]).all():
            raise ValueError(f'坐标 {c} 包含非法值')

    for c in set(df.columns) - {'x', 'y', 'year', 'month'}:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')

    return df, lccs


def _assign_spatial_ids(df):
    """原地分配 _spatial_block、_pixel_id 与月度哑变量，并把坐标转 float32。"""
    if df.duplicated(['x', 'y', 'year', 'month']).any():
        raise ValueError('原始数据具有重复的 x/y/year/month 组合')

    xs = np.sort(df['x'].unique())
    ys = np.sort(df['y'].unique())
    dx = np.median(np.diff(xs))
    dy = np.median(np.diff(ys))
    if not (np.isfinite(dx) and np.isfinite(dy) and dx > 0 and dy > 0):
        raise RuntimeError('坐标网格间距解析失败')

    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    xmin, ymin = float(x.min()), float(y.min())

    df['_spatial_block'] = (
        np.floor((x - xmin) / (dx * SP_FACTOR)).astype(np.int64) * SPATIAL_ID_MULTIPLIER
        + np.floor((y - ymin) / (dy * SP_FACTOR)).astype(np.int64)
    )
    df['_pixel_id'] = (
        np.rint((x - xmin) / dx).astype(np.int64) * SPATIAL_ID_MULTIPLIER
        + np.rint((y - ymin) / dy).astype(np.int64)
    )
    for m in range(2, 13):
        df[f'_month_{m:02d}'] = (df['month'].to_numpy() == m).astype(np.float32)

    df['x'] = df['x'].astype(np.float32)
    df['y'] = df['y'].astype(np.float32)


def _collect_feature_metadata(df):
    """返回模型特征列表和 LCCS 有效类别。"""
    levels = []
    if 'lccs' in df:
        vals = df['lccs'].to_numpy(np.float64)
        levels = np.sort(np.unique(vals[np.isfinite(vals)])).tolist()
    features = (['x', 'y']
                + [f'_month_{m:02d}' for m in range(2, 13)]
                + [c for c in M_A if c in df and c != 'lccs'])
    if 'lccs' in df and not levels:
        raise ValueError('LCCS 非法：没有有效类别')
    return features, levels


def read_raw():
    """读取、清理并准备原始数据，返回 (df, features, levels)。"""
    header = set(pd.read_csv(RAW, nrows=0).columns)
    df, _ = _read_selected_columns(header)
    _assign_spatial_ids(df)
    features, levels = _collect_feature_metadata(df)
    print(f'原始样本 {len(df):,}；空间块 {df._spatial_block.nunique():,}；LCCS 类别 {len(levels)}')
    return df, features, levels


# ============================================================
# C. 空间五折划分
# ============================================================
def folds_global(df):
    """按空间块规模做贪心五折划分。"""
    counts = df['_spatial_block'].value_counts(sort=False)
    items = [(int(k), int(v)) for k, v in counts.items()]
    rng = np.random.default_rng(SEED)
    rng.shuffle(items)
    items.sort(key=lambda kv: kv[1], reverse=True)

    sizes = np.zeros(FOLDS, dtype=np.int64)
    mapping = {}
    for k, n in items:
        f = int(np.argmin(sizes))
        mapping[k] = f
        sizes[f] += n
    if len(mapping) < FOLDS:
        raise RuntimeError('空间块数少于五折')
    print('全国五折样本分布：', sizes.tolist())
    return mapping


# ============================================================
# D. 逐年 DML 拟合
# ============================================================
def _build_design_matrix(work, features, levels):
    """构造特征矩阵 X（含 LCCS 哑变量）。"""
    X = work[features].to_numpy(np.float32)
    if len(levels) > 1:
        v = work['lccs'].to_numpy(np.float64)
        dummy = np.column_stack([(v == lv).astype(np.float32) for lv in levels[1:]])
        X = np.column_stack([X, dummy])
    return X


def _fit_fold_stage(X, train_idx, test_idx, target, params, stage, year, fold):
    """拟合单个空间折的一个阶段；返回 (预测值, OOF RMSE)。"""
    full_params = dict(params)
    full_params.update(
        objective='reg:squarederror',
        eval_metric='rmse',
        tree_method='hist',
        device=DEVICE,
        max_bin=256,
        n_jobs=N_JOBS,
        random_state=SEED + year * 10 + fold,
        verbosity=0,
        validate_parameters=True,
    )
    model = xgb.XGBRegressor(**full_params)
    model.fit(X[train_idx], target[train_idx])
    pred = model.predict(X[test_idx]).astype(np.float32)
    residual = target[test_idx] - pred
    rmse = float(np.sqrt(np.mean(residual.astype('float64') ** 2)))
    del model, residual
    gc.collect()
    return pred, rmse


def _aggregate_monthly(work, year, t, y, th, yh):
    """把正交残差聚合成像元-月份统计量。"""
    a = (t - th).astype(np.float32).astype(np.float64)
    b = (y - yh).astype(np.float32).astype(np.float64)
    z = pd.DataFrame({
        'year': year,
        'month': work['month'].to_numpy(),
        'pixel_id': work['_pixel_id'].to_numpy(),
        'spatial_block': work['_spatial_block'].to_numpy(),
        'x': work['x'].to_numpy(),
        'y': work['y'].to_numpy(),
        't2m': work['t2m'].to_numpy(),
        'O3': t,
        'ssrd': work['ssrd'].to_numpy(),
        **{m: work[m].to_numpy() for m in METEO},
        'A': a * a,
        'B': a * b,
    })
    result = (z.groupby(['year', 'month', 'pixel_id', 'spatial_block'],
                        sort=False, observed=True)
              .agg(
                  x=('x', 'first'), y=('y', 'first'),
                  t2m=('t2m', 'mean'), O3=('O3', 'mean'),
                  ssrd=('ssrd', 'mean'),
                  **{m: (m, 'mean') for m in METEO},
                  numerator_raw=('B', 'sum'),
                  denominator_raw=('A', 'sum'),
                  n_obs=('A', 'size'),
              ).reset_index())
    return result


def annual_fit(work, year, features, levels, fold_map):
    """对单个年份执行五折空间 XGBoost，返回 (像元-月份聚合, 诊断表)。"""
    fold = work['_spatial_block'].map(fold_map).to_numpy(dtype=np.int8)
    if set(np.unique(fold)) != set(range(FOLDS)):
        raise RuntimeError(f'{year}: 五个空间折未全部覆盖')

    X = _build_design_matrix(work, features, levels)
    t = work['O3'].to_numpy(np.float32)
    y = work['SIF'].to_numpy(np.float32)
    th = np.full(len(work), np.nan, dtype=np.float32)
    yh = np.full_like(th, np.nan)
    diag = []

    for f in range(FOLDS):
        train_idx = np.flatnonzero(fold != f)
        test_idx = np.flatnonzero(fold == f)
        if not len(train_idx) or not len(test_idx):
            raise RuntimeError('训练折/测试折为空')
        for stage, target, dest in [('treatment', t, th), ('outcome', y, yh)]:
            pred, rmse = _fit_fold_stage(
                X, train_idx, test_idx, target, HP[stage], stage, year, f,
            )
            dest[test_idx] = pred
            diag.append({
                'year': year, 'fold': f + 1, 'stage': stage,
                'train_n': len(train_idx), 'test_n': len(test_idx),
                'oof_rmse': rmse,
            })

    if not np.isfinite(th).all() or not np.isfinite(yh).all():
        raise RuntimeError(f'{year} 年正交预测包含缺失值')

    result = _aggregate_monthly(work, year, t, y, th, yh)
    del X, work, th, yh
    gc.collect()
    return result, pd.DataFrame(diag)


# ============================================================
# E. 逐年缓存阶段
# ============================================================
def rebuild():
    """逐年拟合 DML 并写出像元-月份 parquet 缓存。"""
    df, features, levels = read_raw()
    fold_map = folds_global(df)
    summaries = []

    for year in YEARS:
        p = CACHE / f'year_{year}_month_pixel.parquet'
        d = CACHE / f'year_{year}_nuisance_diagnostics.csv'
        if RESUME and p.exists() and d.exists():
            import pyarrow.parquet as pq
            cols = set(pq.read_schema(p).names)
            if {'numerator_raw', 'denominator_raw', 't2m', 'O3', 'ssrd', *METEO} <= cols:
                print('继续使用本轮有效缓存：', year)
                continue

        extra = [c for c in features if c not in ['x', 'y']]
        keys = ['year', 'month', 'x', 'y', '_pixel_id', '_spatial_block',
                'SIF', 'O3', 't2m', 'ssrd', *METEO]
        if levels:
            keys.append('lccs')
        keys = list(dict.fromkeys(keys + extra))

        w = df.loc[df['year'] == year, keys].copy()
        w.replace([np.inf, -np.inf], np.nan, inplace=True)
        w.dropna(subset=['SIF', 'O3', *features] + (['lccs'] if levels else []),
                 inplace=True)
        w.reset_index(drop=True, inplace=True)
        if w.empty:
            raise RuntimeError(f'{year}: 完整观测数为 0')

        print(f'\n{year}: 正在进行五折空间 XGBoost，n={len(w):,}')
        v, diag = annual_fit(w, year, features, levels, fold_map)
        tmp = p.with_suffix('.part.parquet')
        v.to_parquet(tmp, index=False)
        os.replace(tmp, p)
        diag.to_csv(d, index=False)
        summaries.append({'year': year, 'n': len(v), 'cache': str(p)})
        del w, v, diag
        gc.collect()

    pd.DataFrame(summaries).to_csv(OUT / 'this_run_new_cache_summary.csv', index=False)
    del df
    gc.collect()
    print('23 年年度 DML 缓存阶段完成')


# ============================================================
# F. DuckDB 数据库装配
# ============================================================
def _create_raw_table(db, files, rad_div):
    months = ','.join(map(str, ACTIVE_MONTHS))
    db.execute(f'''CREATE OR REPLACE TABLE raw AS SELECT
       year::INTEGER year,month::INTEGER month,pixel_id::BIGINT pixel_id,
       spatial_block::BIGINT spatial_block,denominator_raw::DOUBLE A,
       numerator_raw::DOUBLE B,O3::DOUBLE o3,t2m::DOUBLE temp,
       ssrd::DOUBLE/{rad_div} rad,u10::DOUBLE u10,v10::DOUBLE v10,
       tp::DOUBLE tp,sp::DOUBLE sp FROM read_parquet({files})
       WHERE month IN ({months})''')


def _create_valid_table(db):
    variables = ['A', 'B', 'o3', 'temp', 'rad', *METEO]
    ok = ' AND '.join(f'coalesce(isfinite({z}),false)' for z in variables)
    db.execute(f'''CREATE OR REPLACE TABLE valid AS SELECT * FROM raw WHERE A>0 AND {ok}''')


def _create_climate_table(db):
    measures = ['o3', 'temp', 'rad', *METEO]
    means = ','.join(f'avg({z}) AS {z}_mean' for z in measures)
    db.execute(f'''CREATE OR REPLACE TABLE climate AS SELECT pixel_id,month,count(*) n_years,
        bit_or(1::UBIGINT << (year-2000))::UBIGINT year_mask,{means}
        FROM valid GROUP BY pixel_id,month''')


def _create_base_table(db):
    measures = ['o3', 'temp', 'rad', *METEO]
    anomal = ','.join(f'(v.{z}-c.{z}_mean) AS {z}_anom' for z in measures)
    db.execute(f'''CREATE OR REPLACE TABLE base AS SELECT v.year,v.month,v.pixel_id,v.spatial_block,v.A,v.B,
        c.n_years,c.year_mask,{anomal} FROM valid v JOIN climate c USING(pixel_id,month)
        WHERE c.n_years>=10''')


def _compute_std(db):
    """返回 (辐射距平标准差, [气象距平标准差])。"""
    cols = ['rad', *METEO]
    std = db.execute('SELECT ' + ','.join(f'stddev_samp({z}_anom)' for z in cols) + ' FROM base').fetchone()
    if any(v is None or not np.isfinite(v) or v < 1e-12 for v in std):
        raise RuntimeError('某一气象距平没有变异，无法拟合')
    return float(std[0]), [float(v) for v in std[1:]]


def _create_scaled_table(db, rad_std, meteo_std):
    meteo_z = ','.join(f'{z}_anom/{v:.17g} AS {z}_z' for z, v in zip(METEO, meteo_std))
    db.execute(f'''CREATE OR REPLACE TABLE scaled AS SELECT *,rad_anom/{rad_std:.17g} r_sd,
       temp_anom*rad_anom/{rad_std:.17g} tr,{meteo_z} FROM base''')


def _create_trmean_table(db):
    db.execute('''CREATE OR REPLACE TABLE trmean AS SELECT pixel_id,month,avg(tr) tr_mean
        FROM scaled GROUP BY pixel_id,month''')


def _create_analysis_table(db):
    db.execute('''CREATE OR REPLACE TABLE analysis AS SELECT s.*,s.tr-m.tr_mean tr_within
       FROM scaled s JOIN trmean m USING(pixel_id,month)''')


def _check_analysis_support(db):
    support = db.execute(
        'SELECT count(*),count(DISTINCT year),count(DISTINCT spatial_block) FROM analysis'
    ).fetchone()
    if support[0] == 0:
        raise RuntimeError('没有满足观测覆盖条件的分析样本')
    print('气候扩展样本、年份、空间块：', support)


def prepare_db():
    """构建并返回 DuckDB 连接，包含所有派生分析表。"""
    import duckdb, pyarrow.parquet as pq  # noqa: F401
    paths = [CACHE / f'year_{y}_month_pixel.parquet' for y in YEARS]
    if any(not p.exists() for p in paths):
        raise FileNotFoundError('缺少本轮年度 Parquet，不能进入后续分析')

    db = duckdb.connect(str(RESULT / 'radiation.duckdb'))
    db.execute(f'SET threads={N_JOBS}')
    db.execute(f"SET memory_limit='{DUCKDB_GB}GB'")
    temp = RESULT / 'tmp'
    temp.mkdir(exist_ok=True)
    db.execute("SET temp_directory='" + str(temp).replace("'", "''") + "'")

    files = "[" + ",".join("'" + str(p).replace("'", "''") + "'" for p in paths) + "]"
    rad_div = 86400. if SSRD_UNIT == 'j_m2_per_day' else 1.

    _create_raw_table(db, files, rad_div)
    _create_valid_table(db)
    _create_climate_table(db)
    _create_base_table(db)
    rad_std, meteo_std = _compute_std(db)
    _create_scaled_table(db, rad_std, meteo_std)
    _create_trmean_table(db)
    _create_analysis_table(db)
    _check_analysis_support(db)
    return db


# ============================================================
# G. 设计与流式读取
# ============================================================
def stream(db):
    """分块读取 analysis 表。"""
    conn = db.execute('SELECT ' + ','.join(ANALYSIS_COLUMNS) + ' FROM analysis')
    while True:
        d = conn.fetch_df_chunk(32)
        if d.empty:
            break
        yield d


def parameter_names(kind):
    """返回每个模型的设计矩阵参数名。"""
    if kind == 'A':
        return ['temp_K', 'radiation_SD', 'temp_x_radiation',
                *[m + '_z' for m in METEO],
                *[f'year_{y}' for y in range(2001, 2023)]]
    month_cols = [f'month_{m:02d}' for m in ACTIVE_MONTHS]
    if kind == 'T':
        return month_cols + ['temp_K']
    return month_cols + ['temp_K', 'radiation_SD', 'temp_x_radiation']


def design_matrix(d, kind):
    """按模型类型构造设计矩阵 X。"""
    t = d.temp_anom.to_numpy(float)
    r = d.r_sd.to_numpy(float)
    if kind == 'A':
        yy = d.year.to_numpy(np.int16)
        mask = d.year_mask.to_numpy(np.uint64)
        n = d.n_years.to_numpy(float)
        year_cols = [
            (yy == y).astype(float)
            - ((mask >> np.uint64(y - 2000)) & np.uint64(1)).astype(float) / n
            for y in range(2001, 2023)
        ]
        return np.column_stack([t, r, d.tr_within.to_numpy(float),
                                *[d[m + '_z'].to_numpy(float) for m in METEO],
                                *year_cols])
    month_cols = [(d.month.to_numpy() == m).astype(float) for m in ACTIVE_MONTHS]
    if kind == 'T':
        return np.column_stack([*month_cols, t])
    return np.column_stack([*month_cols, t, r, t * r])


# ============================================================
# H. DML 估计
# ============================================================
def _fit_normal_equations(db, kind):
    """先累积正规方程，返回 (beta, inv_H, N, 参数名)。"""
    pnames = parameter_names(kind)
    p = len(pnames)
    H = np.zeros((p, p))
    rhs = np.zeros(p)
    N = 0
    for d in stream(db):
        X = design_matrix(d, kind)
        a = d.A.to_numpy(float)
        b = d.B.to_numpy(float)
        if kind == 'A':
            H += X.T @ X
            rhs += X.T @ d.o3_anom.to_numpy(float)
        else:
            H += X.T @ (a[:, None] * X)
            rhs += X.T @ b
        N += len(d)

    H = (H + H.T) / 2
    cond = np.linalg.cond(H)
    if not np.isfinite(cond) or cond > COND_MAX:
        raise RuntimeError(f'{kind} 设计矩阵近奇异；停止推断')
    inv = np.linalg.inv(H)
    beta = inv @ rhs
    return beta, inv, N, pnames


def _aggregate_influence(db, kind, beta, blocks):
    """按 (空间块, 年份) 累积影响函数，返回 3D 数组 (n_blocks, n_years, p)。"""
    p = len(beta)
    ny = len(YEARS)
    joint = np.zeros((len(blocks) * ny, p))
    for d in stream(db):
        X = design_matrix(d, kind)
        if kind == 'A':
            resid = d.o3_anom.to_numpy(float) - X @ beta
        else:
            resid = d.B.to_numpy(float) - d.A.to_numpy(float) * (X @ beta)
        s = X * resid[:, None]
        idx = (np.searchsorted(blocks, d.spatial_block.to_numpy(np.int64)) * ny
               + (d.year.to_numpy(np.int64) - 2000))
        unique, reverse = np.unique(idx, return_inverse=True)
        aggregated = np.column_stack([
            np.bincount(reverse, weights=s[:, j], minlength=len(unique))
            for j in range(p)
        ])
        joint[unique] += aggregated
    return joint.reshape(len(blocks), ny, p)


def _compute_cluster_cov(joint, inv):
    """计算双向聚类稳健协方差，返回 (V, label, df)。"""
    n_blocks, ny, p = joint.shape
    joint_flat = joint.reshape(n_blocks * ny, p)
    sum_over_years = joint.sum(axis=1)   # (n_blocks, p)
    sum_over_blocks = joint.sum(axis=0)  # (n_years, p)
    sum_over_years = sum_over_years[np.any(np.abs(sum_over_years) > 0, axis=1)]
    sum_over_blocks = sum_over_blocks[np.any(np.abs(sum_over_blocks) > 0, axis=1)]
    influence = joint_flat[np.any(np.abs(joint_flat) > 0, axis=1)]

    nb = len(sum_over_years)
    ny_eff = len(sum_over_blocks)
    ni = len(influence)
    if min(nb, ny_eff, ni) < 2:
        raise RuntimeError('空间块/年份聚类数不足')

    Vy = inv @ (sum_over_blocks.T @ sum_over_blocks * ny_eff / (ny_eff - 1)) @ inv
    Vy = (Vy + Vy.T) / 2
    V2 = inv @ (
        sum_over_years.T @ sum_over_years * nb / (nb - 1)
        + sum_over_blocks.T @ sum_over_blocks * ny_eff / (ny_eff - 1)
        - influence.T @ influence * ni / (ni - 1)
    ) @ inv
    V2 = (V2 + V2.T) / 2

    if not np.isfinite(Vy).all() or np.any(np.diag(Vy) <= 0):
        raise RuntimeError('年份聚类协方差无效')

    use2 = (np.isfinite(V2).all()
            and np.all(np.diag(V2) > 0)
            and np.linalg.eigvalsh(V2).min() >= -1e-9)
    V = V2 if use2 else Vy
    label = 'space_x_year' if use2 else 'year_fallback'
    df = min(nb, ny_eff) - 1 if use2 else ny_eff - 1
    return V, label, df


def _write_joint_test(kind, beta, V, pnames, df):
    """对 radiation_SD 与 temp_x_radiation 做联合 Wald 检验。"""
    idxs = [pnames.index('radiation_SD'), pnames.index('temp_x_radiation')]
    try:
        stat = float(beta[idxs] @ np.linalg.solve(V[np.ix_(idxs, idxs)], beta[idxs]))
        pjoint = float(stats.f.sf(stat / 2, 2, df))
    except np.linalg.LinAlgError:
        stat = pjoint = np.nan
    pd.DataFrame([{'joint_F_approx_p': pjoint, 'wald_statistic': stat}]).to_csv(
        RESULT / f'{kind}_joint_test.csv', index=False
    )


def _year_multiplier_bootstrap(joint, inv, beta, pnames, kind):
    """年份乘子 Bootstrap，返回 DataFrame 参数, year_multiplier_p。"""
    rng = np.random.default_rng(SEED)
    signs = rng.choice([-1., 1.], size=(BOOTSTRAPS, len(YEARS)))
    scale = 1 if kind == 'A' else 10
    perturb = (signs @ joint.sum(axis=0)) @ inv.T * scale
    chosen = ['temp_K'] if kind == 'T' else ['radiation_SD', 'temp_x_radiation']
    rows = []
    for name in chosen:
        idx = pnames.index(name)
        threshold = abs(beta[idx] * scale)
        exceed = np.sum(np.abs(perturb[:, idx]) >= threshold)
        rows.append(dict(parameter=name,
                         year_multiplier_p=(1 + exceed) / (BOOTSTRAPS + 1)))
    return pd.DataFrame(rows)


def estimate(db, kind, blocks):
    """估计一个 DML 模型并写出参数、协方差与 Bootstrap 结果。"""
    beta, inv, N, pnames = _fit_normal_equations(db, kind)
    joint = _aggregate_influence(db, kind, beta, blocks)
    V, cov_label, df = _compute_cluster_cov(joint, inv)

    scale = 1 if kind == 'A' else 10
    est = beta * scale
    se = np.sqrt(np.maximum(0, np.diag(V))) * scale
    crit = stats.t.ppf(.975, df)

    result = pd.DataFrame(dict(
        parameter=pnames,
        estimate=est,
        se=se,
        ci95_low=est - crit * se,
        ci95_high=est + crit * se,
        p_value=2 * stats.t.sf(np.abs(est) / np.where(se > 0, se, np.nan), df),
        covariance=cov_label,
    ))
    result.to_csv(RESULT / f'{kind}_parameters.csv', index=False)

    if kind != 'T':
        _write_joint_test(kind, beta, V, pnames, df)

    bootstrap_df = _year_multiplier_bootstrap(joint, inv, beta, pnames, kind)
    bootstrap_df.to_csv(RESULT / f'{kind}_year_multiplier_p.csv', index=False)

    chosen = ['temp_K'] if kind == 'T' else ['radiation_SD', 'temp_x_radiation']
    print(kind, ':', N, 'obs; covariance=', cov_label)
    print(result[result.parameter.isin(chosen)].to_string(index=False))
    return dict(beta=beta, cov=V, scale=scale, df=df, names=pnames)


def predictions(model, Q):
    """由模型系数与协方差给出预测值与 95% 置信区间。"""
    value = Q @ model['beta'] * model['scale']
    err = np.sqrt(np.maximum(
        0, np.einsum('ij,jk,ik->i', Q, model['cov'], Q)
    )) * model['scale']
    z = stats.t.ppf(.975, model['df'])
    return value, value - z * err, value + z * err


# ============================================================
# I. 绘图
# ============================================================
def _plot_radiation_curve(A, tlo, thi):
    """输出 A 模型温度调节的斜率曲线。"""
    import matplotlib.pyplot as plt
    xx = np.linspace(tlo, thi, 121)
    Q = np.zeros((len(xx), len(A['names'])))
    Q[:, A['names'].index('radiation_SD')] = 1
    Q[:, A['names'].index('temp_x_radiation')] = xx
    e, l, h = predictions(A, Q)

    pd.DataFrame(dict(temp_K=xx, slope=e, ci95_low=l, ci95_high=h)).to_csv(
        RESULT / 'A_radiation_O3_curve.csv', index=False
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xx, e)
    ax.fill_between(xx, l, h, alpha=.18)
    ax.axhline(0, color='grey', ls='--', lw=.8)
    ax.set(xlabel='Temperature anomaly (K)',
           ylabel='O3 change per +1 SD radiation')
    fig.tight_layout()
    fig.savefig(RESULT / 'A_radiation_O3.png', dpi=300)
    plt.close(fig)


def _plot_conditional_heatmap(db, B, tlo, thi, rlo, rhi):
    """温度—辐射联合支持网格上的条件斜率与热力图。"""
    import matplotlib.pyplot as plt
    n = 8
    xt = f'LEAST({n-1},GREATEST(0,CAST(floor((temp_anom-({tlo}))/({thi-tlo})*{n}) AS INTEGER)))'
    xr = f'LEAST({n-1},GREATEST(0,CAST(floor((r_sd-({rlo}))/({rhi-rlo})*{n}) AS INTEGER)))'
    c = db.execute(f'''SELECT {xt} tb,{xr} rb,count(*) n,count(DISTINCT year) ny,
       count(DISTINCT spatial_block) nb,avg(temp_anom) t,avg(r_sd) r FROM analysis
       WHERE temp_anom BETWEEN {tlo} AND {thi}
       AND r_sd BETWEEN {rlo} AND {rhi} GROUP BY 1,2''').df()
    c['supported'] = (c.n >= 1000) & (c.ny >= 10) & (c.nb >= 20)
    c.to_csv(RESULT / 'temperature_radiation_support.csv', index=False)

    Z = np.full((n, n), np.nan)
    rows = []
    month_indices = [B['names'].index(f'month_{m:02d}') for m in ACTIVE_MONTHS]
    for z in c.itertuples(index=False):
        if not z.supported:
            continue
        Q = np.zeros((1, len(B['names'])))
        for idx in month_indices:
            Q[0, idx] = 1 / len(ACTIVE_MONTHS)
        Q[0, B['names'].index('temp_K')] = z.t
        Q[0, B['names'].index('radiation_SD')] = z.r
        Q[0, B['names'].index('temp_x_radiation')] = z.t * z.r
        e, l, h = predictions(B, Q)
        Z[int(z.rb), int(z.tb)] = float(e[0])
        rows.append(dict(tbin=z.tb, rbin=z.rb,
                         est=e[0], low=l[0], high=h[0]))
    pd.DataFrame(rows).to_csv(RESULT / 'B_supported_conditional_slopes.csv', index=False)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('lightgrey')
    lim = max(np.nanmax(np.abs(Z)) if np.isfinite(Z).any() else 0, 1e-10)
    im = ax.imshow(Z, origin='lower', aspect='auto', cmap=cmap,
                   vmin=-lim, vmax=lim, extent=[tlo, thi, rlo, rhi])
    ax.set(xlabel='Temperature anomaly (K)',
           ylabel='Radiation anomaly (SD)')
    fig.colorbar(im, ax=ax, label='O3–SIF slope per +10 ug/m3 O3')
    fig.tight_layout()
    fig.savefig(RESULT / 'B_temperature_radiation_modifier.png', dpi=300)
    plt.close(fig)


def figures(db, A, B):
    """输出辐射调节曲线与条件斜率热力图。"""
    t, r = db.execute(
        'SELECT approx_quantile(temp_anom,[0.02,0.98]),'
        'approx_quantile(r_sd,[0.02,0.98]) FROM analysis'
    ).fetchone()
    tlo, thi = map(float, t)
    rlo, rhi = map(float, r)
    if not (thi > tlo and rhi > rlo):
        print('气候距平范围不足，跳过曲线绘图')
        return
    _plot_radiation_curve(A, tlo, thi)
    _plot_conditional_heatmap(db, B, tlo, thi, rlo, rhi)


# ============================================================
# J. 入口
# ============================================================
def run_all():
    """执行完整流程。"""
    setup()
    rebuild()
    db = prepare_db()
    try:
        blocks = np.array(
            [row[0] for row in
             db.execute('SELECT DISTINCT spatial_block FROM analysis ORDER BY 1').fetchall()],
            dtype=np.int64,
        )
        A = estimate(db, 'A', blocks)
        T = estimate(db, 'T', blocks)
        B = estimate(db, 'B', blocks)
        figures(db, A, B)
        meta = dict(
            version=SCRIPT_VERSION,
            ssrd_unit=SSRD_UNIT,
            months=list(ACTIVE_MONTHS),
            limitations='Conditional associations; SSRD is not PAR or photolysis rate',
        )
        (RESULT / 'RUN_METADATA.json').write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8'
        )
        print('\n全部完成。输出目录：', RESULT)
    finally:
        db.close()


run_all()