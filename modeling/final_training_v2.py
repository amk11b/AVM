import os
import html
import re
import tempfile
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from google.cloud import storage
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import StackingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
from sklearn.preprocessing import FunctionTransformer, RobustScaler

import warnings

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_INPUT = BASE_DIR / "embeddings.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


def _env_int(name, default):
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name, default):
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def image_feature_columns(df):
    return [c for c in df.columns if c.startswith("img_dim_")] + [c for c in ["has_image_embedding"] if c in df.columns]


def _parse_gcs_uri(uri):
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected a gs:// URI, got: {uri}")
    bucket_name, _, blob_name = uri[5:].partition("/")
    if not bucket_name or not blob_name:
        raise ValueError(f"Invalid gs:// URI: {uri}")
    return bucket_name, blob_name


def _download_input(source):
    source = str(source)
    if source.startswith("gs://"):
        bucket_name, blob_name = _parse_gcs_uri(source)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(f"GCS object does not exist: {source}")
        tmp_dir = Path(tempfile.mkdtemp(prefix="avm-input-"))
        local_path = tmp_dir / Path(blob_name).name
        blob.download_to_filename(local_path)
        return local_path
    return Path(source)


def _upload_outputs(local_dir, gcs_uri):
    bucket_name, prefix = _parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    prefix = prefix.rstrip("/")

    for path in Path(local_dir).glob("final_*"):
        if path.is_file():
            blob_name = f"{prefix}/{path.name}" if prefix else path.name
            bucket.blob(blob_name).upload_from_filename(path)


def clean_price(series):
    return pd.to_numeric(series.astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")


def simplify_type(value):
    text = str(value).lower()
    if "flat" in text or "apartment" in text or "maisonette" in text:
        return "Flat"
    if "detached" in text and "semi" not in text:
        return "Detached"
    if "semi" in text:
        return "Semi-Detached"
    if "bungalow" in text:
        return "Bungalow"
    if "retirement" in text:
        return "Retirement"
    return "Terraced"


def clean_text_value(value):
    text = "" if pd.isna(value) else str(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_postcode_sector(value):
    text = clean_text_value(value).upper()
    match = re.search(r"([A-Z]{1,2}\d{1,2}\s?\d)", text)
    if not match:
        return "Unknown"
    return match.group(1).replace(" ", "")


def safe_text_column(df, column):
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[column].fillna("").astype(str).map(clean_text_value)


def make_holdout_strata(df):
    price_bins = pd.qcut(df["price"].rank(method="first"), q=min(10, max(3, len(df) // 500)), labels=False, duplicates="drop")
    strata = df["simple_type"].astype(str) + "_" + price_bins.astype(str)
    strata = strata.fillna("unknown")
    if (strata.value_counts() < 2).any():
        return df["simple_type"].astype(str)
    return strata


def add_text_flags(df):
    text = df["description"].fillna("").astype(str).map(clean_text_value).str.lower()
    df = df.copy()
    df["description_len"] = text.str.len()
    df["description_words"] = text.str.split().str.len()
    df["description_has_digits"] = text.str.contains(r"\d", regex=True).astype(int)
    df["is_modern"] = text.str.contains(
        r"\b(renovated|refurbished|modern|contemporary|updated|immaculate|newly decorated)\b",
        regex=True,
    ).astype(int)
    df["needs_work"] = text.str.contains(
        r"\b(modernisation|modernization|potential|tlc|fixer|project|in need of)\b",
        regex=True,
    ).astype(int)
    df["has_garden"] = text.str.contains(r"\b(garden|patio|terrace|balcony|outside space)\b", regex=True).astype(int)
    df["has_parking"] = text.str.contains(r"\b(parking|garage|drive|driveway|allocated parking)\b", regex=True).astype(int)
    df["luxury"] = text.str.contains(r"\b(luxury|premium|executive|prestigious|bespoke)\b", regex=True).astype(int)
    df["period"] = text.str.contains(r"\b(victorian|edwardian|georgian|period|character)\b", regex=True).astype(int)
    df["has_views"] = text.str.contains(r"\b(view|views|sea view|countryside|open aspect)\b", regex=True).astype(int)
    return df


def fit_target_encoding(train_series, y, smooth=20.0):
    frame = pd.DataFrame({"key": train_series.fillna("Unknown").astype(str), "target": y})
    stats = frame.groupby("key")["target"].agg(["count", "mean"])
    global_mean = y.mean()
    enc = (stats["count"] * stats["mean"] + smooth * global_mean) / (stats["count"] + smooth)
    return enc.to_dict(), global_mean


def apply_target_encoding(series, mapping, global_mean):
    return series.fillna("Unknown").astype(str).map(mapping).fillna(global_mean).astype(float)


def build_neighbor_features(train_df, test_df, target_log):
    if not {"latitude", "longitude"}.issubset(train_df.columns) or not {"latitude", "longitude"}.issubset(test_df.columns):
        empty_train = pd.DataFrame(index=train_df.index)
        empty_test = pd.DataFrame(index=test_df.index)
        for prefix in ["comp_close", "comp_medium", "comp_wide"]:
            empty_train[f"{prefix}_mean"] = 0.0
            empty_train[f"{prefix}_std"] = 0.0
            empty_test[f"{prefix}_mean"] = 0.0
            empty_test[f"{prefix}_std"] = 0.0
        return train_df.copy(), test_df.copy()

    train_df = train_df.copy()
    test_df = test_df.copy()

    for prefix in ["comp_close", "comp_medium", "comp_wide"]:
        for suffix in ["mean", "std"]:
            train_df[f"{prefix}_{suffix}"] = np.nan
            test_df[f"{prefix}_{suffix}"] = np.nan

    for p_type in train_df["simple_type"].unique():
        train_mask = train_df["simple_type"] == p_type
        test_mask = test_df["simple_type"] == p_type
        train_subset = train_df[train_mask]

        if len(train_subset) < 30:
            continue

        coords = np.radians(train_subset[["latitude", "longitude"]].values)
        tree = BallTree(coords, metric="haversine")
        subset_targets = target_log.loc[train_subset.index]

        for prefix, k in [("comp_close", 6), ("comp_medium", 16), ("comp_wide", 31)]:
            _, ind = tree.query(coords, k=k)
            means = []
            stds = []
            for indices in ind:
                vals = subset_targets.iloc[indices[1:]]
                means.append(vals.mean())
                stds.append(vals.std())
            train_df.loc[train_mask, f"{prefix}_mean"] = means
            train_df.loc[train_mask, f"{prefix}_std"] = stds

        if test_mask.sum() > 0:
            test_coords = np.radians(test_df[test_mask][["latitude", "longitude"]].values)
            for prefix, k in [("comp_close", 5), ("comp_medium", 15), ("comp_wide", 30)]:
                _, ind = tree.query(test_coords, k=k)
                means = []
                stds = []
                for indices in ind:
                    vals = subset_targets.iloc[indices]
                    means.append(vals.mean())
                    stds.append(vals.std())
                test_df.loc[test_mask, f"{prefix}_mean"] = means
                test_df.loc[test_mask, f"{prefix}_std"] = stds

    global_mean = target_log.mean()
    global_std = target_log.std()
    for prefix in ["comp_close", "comp_medium", "comp_wide"]:
        train_df[f"{prefix}_mean"] = train_df[f"{prefix}_mean"].fillna(global_mean)
        test_df[f"{prefix}_mean"] = test_df[f"{prefix}_mean"].fillna(global_mean)
        train_df[f"{prefix}_std"] = train_df[f"{prefix}_std"].fillna(global_std)
        test_df[f"{prefix}_std"] = test_df[f"{prefix}_std"].fillna(global_std)

    return train_df, test_df


def build_text_features(train_text, test_text, max_features=3000, svd_components=40):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
    )
    train_tfidf = vectorizer.fit_transform(train_text)
    test_tfidf = vectorizer.transform(test_text)

    n_components = min(svd_components, max(2, train_tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    train_svd = svd.fit_transform(train_tfidf)
    test_svd = svd.transform(test_tfidf)

    train_svd = pd.DataFrame(train_svd, columns=[f"text_svd_{i}" for i in range(train_svd.shape[1])], index=train_text.index)
    test_svd = pd.DataFrame(test_svd, columns=[f"text_svd_{i}" for i in range(test_svd.shape[1])], index=test_text.index)
    return vectorizer, svd, train_svd, test_svd


def build_image_features(train_df, test_df):
    img_cols = sorted([c for c in train_df.columns if c.startswith("img_dim_")], key=lambda c: int(c.split("_")[-1]))
    if not img_cols:
        empty_train = pd.DataFrame(index=train_df.index)
        empty_test = pd.DataFrame(index=test_df.index)
        return None, empty_train, empty_test, []

    train_img = train_df[img_cols].fillna(0.0).astype(float)
    test_img = test_df[img_cols].fillna(0.0).astype(float)

    train_stats = pd.DataFrame(index=train_df.index)
    test_stats = pd.DataFrame(index=test_df.index)
    for frame, source in [(train_stats, train_img), (test_stats, test_img)]:
        frame["img_norm"] = np.linalg.norm(source.values, axis=1)
        frame["img_mean"] = source.mean(axis=1)
        frame["img_std"] = source.std(axis=1)
        frame["img_min"] = source.min(axis=1)
        frame["img_max"] = source.max(axis=1)

    n_components = min(48, train_img.shape[1], max(2, len(train_img) - 1))
    pca = PCA(n_components=n_components, random_state=42, svd_solver="randomized")
    train_pca = pca.fit_transform(train_img)
    test_pca = pca.transform(test_img)
    train_pca = pd.DataFrame(train_pca, columns=[f"img_pca_{i}" for i in range(train_pca.shape[1])], index=train_df.index)
    test_pca = pd.DataFrame(test_pca, columns=[f"img_pca_{i}" for i in range(test_pca.shape[1])], index=test_df.index)
    return pca, pd.concat([train_stats, train_pca], axis=1), pd.concat([test_stats, test_pca], axis=1), img_cols


def build_location_features(train_df, test_df):
    if not {"latitude", "longitude"}.issubset(train_df.columns) or not {"latitude", "longitude"}.issubset(test_df.columns):
        empty_train = pd.DataFrame(index=train_df.index)
        empty_test = pd.DataFrame(index=test_df.index)
        empty_train["location_cluster"] = 0.0
        empty_test["location_cluster"] = 0.0
        return None, empty_train, empty_test

    n_clusters = min(20, max(2, len(train_df) // 250))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_cluster = kmeans.fit_predict(train_df[["latitude", "longitude"]].values)
    test_cluster = kmeans.predict(test_df[["latitude", "longitude"]].values)

    train_out = pd.DataFrame(index=train_df.index)
    test_out = pd.DataFrame(index=test_df.index)
    for frame, source in [(train_out, train_df), (test_out, test_df)]:
        lat = np.radians(source["latitude"].astype(float))
        lon = np.radians(source["longitude"].astype(float))
        frame["lat_sin"] = np.sin(lat)
        frame["lat_cos"] = np.cos(lat)
        frame["lon_sin"] = np.sin(lon)
        frame["lon_cos"] = np.cos(lon)
        frame["lat_lon_interaction"] = source["latitude"].astype(float) * source["longitude"].astype(float)
        frame["distance_from_median_lat"] = source["latitude"].astype(float) - source["latitude"].median()
        frame["distance_from_median_lon"] = source["longitude"].astype(float) - source["longitude"].median()

    train_out["location_cluster"] = train_cluster.astype(float)
    test_out["location_cluster"] = test_cluster.astype(float)
    return kmeans, train_out, test_out


def encode_categories(train_df, test_df):
    encoded = {}
    for col in ["city", "property_type", "simple_type"]:
        categories = pd.Index(train_df[col].fillna("Unknown").astype(str).unique())
        mapping = {value: idx for idx, value in enumerate(categories)}
        encoded[f"{col}_code"] = (
            train_df[col].fillna("Unknown").astype(str).map(mapping).fillna(-1).astype(float),
            test_df[col].fillna("Unknown").astype(str).map(mapping).fillna(-1).astype(float),
        )
    return encoded


def assemble_features(raw_train, raw_test, target_log):
    train_df = add_text_flags(raw_train)
    test_df = add_text_flags(raw_test)

    train_df, test_df = build_neighbor_features(train_df, test_df, target_log)

    cat_info = encode_categories(train_df, test_df)
    for key, (train_col, test_col) in cat_info.items():
        if key.endswith("_mapping"):
            continue
        train_df[key] = train_col
        test_df[key] = test_col

    for col in ["city", "property_type", "simple_type"]:
        mapping, global_mean = fit_target_encoding(train_df[col], target_log)
        train_df[f"{col}_te"] = apply_target_encoding(train_df[col], mapping, global_mean)
        test_df[f"{col}_te"] = apply_target_encoding(test_df[col], mapping, global_mean)

    address_source_train = safe_text_column(train_df, "address")
    address_source_test = safe_text_column(test_df, "address")

    if "address" in train_df.columns and "address" in test_df.columns:
        train_postcode_sector = address_source_train.map(extract_postcode_sector)
        test_postcode_sector = address_source_test.map(extract_postcode_sector)
        mapping, global_mean = fit_target_encoding(train_postcode_sector, target_log)
        train_df["postcode_sector_te"] = apply_target_encoding(train_postcode_sector, mapping, global_mean)
        test_df["postcode_sector_te"] = apply_target_encoding(test_postcode_sector, mapping, global_mean)

    for col in ["district", "comp_source"]:
        if col in train_df.columns and col in test_df.columns:
            mapping, global_mean = fit_target_encoding(train_df[col], target_log)
            train_df[f"{col}_te"] = apply_target_encoding(train_df[col], mapping, global_mean)
            test_df[f"{col}_te"] = apply_target_encoding(test_df[col], mapping, global_mean)

    combo = train_df["city"].fillna("Unknown").astype(str) + "__" + train_df["property_type"].fillna("Unknown").astype(str)
    combo_test = test_df["city"].fillna("Unknown").astype(str) + "__" + test_df["property_type"].fillna("Unknown").astype(str)
    mapping, global_mean = fit_target_encoding(combo, target_log)
    train_df["city_property_te"] = apply_target_encoding(combo, mapping, global_mean)
    test_df["city_property_te"] = apply_target_encoding(combo_test, mapping, global_mean)

    bedrooms_bin = pd.cut(train_df["bedrooms"].fillna(0).clip(0, 8), bins=[-0.1, 0.5, 1.5, 2.5, 3.5, 4.5, 8.1], labels=False)
    bedrooms_bin_test = pd.cut(test_df["bedrooms"].fillna(0).clip(0, 8), bins=[-0.1, 0.5, 1.5, 2.5, 3.5, 4.5, 8.1], labels=False)
    mapping, global_mean = fit_target_encoding(bedrooms_bin.astype(str), target_log)
    train_df["bedrooms_bin_te"] = apply_target_encoding(bedrooms_bin.astype(str), mapping, global_mean)
    test_df["bedrooms_bin_te"] = apply_target_encoding(bedrooms_bin_test.astype(str), mapping, global_mean)

    text_train = (
        safe_text_column(train_df, "description")
        + " "
        + address_source_train
        + " "
        + safe_text_column(train_df, "city")
        + " "
        + safe_text_column(train_df, "property_type")
    ).str.lower()
    text_test = (
        safe_text_column(test_df, "description")
        + " "
        + address_source_test
        + " "
        + safe_text_column(test_df, "city")
        + " "
        + safe_text_column(test_df, "property_type")
    ).str.lower()
    text_vectorizer, text_svd, train_text_feats, test_text_feats = build_text_features(text_train, text_test)

    img_pca, train_img_feats, test_img_feats, img_cols = build_image_features(train_df, test_df)
    kmeans, train_loc_feats, test_loc_feats = build_location_features(train_df, test_df)

    base_numeric_cols = [
        "bedrooms",
        "floor_area",
        "energy_efficiency",
        "epc_records",
        "has_image_embedding",
        "description_len",
        "description_words",
        "description_has_digits",
        "is_modern",
        "needs_work",
        "has_garden",
        "has_parking",
        "luxury",
        "period",
        "has_views",
        "city_code",
        "property_type_code",
        "simple_type_code",
        "city_te",
        "property_type_te",
        "simple_type_te",
        "city_property_te",
        "bedrooms_bin_te",
        "postcode_sector_te",
        "district_te",
        "comp_source_te",
        "best_comp_price",
        "comp_count",
        "sector_sales_count",
        "sector_avg_price",
        "sector_median_price",
        "sector_min_price",
        "sector_max_price",
        "sector_price_stddev",
        "city_sales_count",
        "city_avg_price",
        "city_median_price",
    ]

    for col in ["latitude", "longitude"]:
        if col in train_df.columns and col in test_df.columns:
            base_numeric_cols.insert(1, col)

    for col in ["comp_close_mean", "comp_close_std", "comp_medium_mean", "comp_medium_std", "comp_wide_mean", "comp_wide_std"]:
        if col in train_df.columns and col in test_df.columns:
            base_numeric_cols.append(col)

    train_features = pd.concat(
        [
            train_df[[c for c in base_numeric_cols if c in train_df.columns]].copy(),
            train_img_feats,
            train_text_feats,
            train_loc_feats,
        ],
        axis=1,
    )
    test_features = pd.concat(
        [
            test_df[[c for c in base_numeric_cols if c in test_df.columns]].copy(),
            test_img_feats,
            test_text_feats,
            test_loc_feats,
        ],
        axis=1,
    )

    train_features = train_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    test_features = test_features.replace([np.inf, -np.inf], np.nan).fillna(0)

    preprocessors = {
        "text_vectorizer": text_vectorizer,
        "text_svd": text_svd,
        "image_pca": img_pca,
        "kmeans": kmeans,
        "image_columns": img_cols,
    }
    return train_features, test_features, preprocessors


def fit_models(X_train, y_train_log, X_test, y_test_log, n_jobs, fast_mode):
    lgb_model, xgb_model, cat_model = make_model_objects(n_jobs, fast_mode)

    lgb_model.fit(X_train, y_train_log, eval_set=[(X_test, y_test_log)], callbacks=[lgb.early_stopping(100, verbose=False)])
    xgb_model.fit(
        X_train,
        y_train_log,
        eval_set=[(X_test, y_test_log)],
        verbose=False,
    )
    cat_model.fit(X_train, y_train_log, eval_set=(X_test, y_test_log), use_best_model=True)

    return lgb_model, xgb_model, cat_model


def make_model_objects(n_jobs, fast_mode):
    lgb_model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=2500 if not fast_mode else 1200,
        learning_rate=0.02,
        num_leaves=96,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=0.3,
        random_state=42,
        n_jobs=n_jobs,
        verbose=-1,
    )

    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=2500 if not fast_mode else 1200,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="mae",
        random_state=42,
        n_jobs=n_jobs,
        verbosity=0,
    )

    cat_model = CatBoostRegressor(
        loss_function="MAE",
        iterations=3000 if not fast_mode else 1500,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=4,
        random_strength=1.0,
        bagging_temperature=0.5,
        od_type="Iter",
        od_wait=120,
        verbose=False,
        random_seed=42,
    )
    return lgb_model, xgb_model, cat_model


def fit_final_models(X_train, y_train_log, n_jobs, fast_mode):
    lgb_model, xgb_model, cat_model = make_model_objects(n_jobs, fast_mode)
    lgb_model.fit(X_train, y_train_log)
    xgb_model.fit(X_train, y_train_log, verbose=False)
    cat_model.fit(X_train, y_train_log, verbose=False)
    return lgb_model, xgb_model, cat_model


def build_oof_blend_weights(X_train, y_train_log, n_jobs, fast_mode, folds=3):
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    oof = {
        "lgb": np.full(len(X_train), np.nan),
        "xgb": np.full(len(X_train), np.nan),
        "cat": np.full(len(X_train), np.nan),
    }

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), start=1):
        print(f"   OOF fold {fold}/{folds}...")
        fold_models = fit_models(
            X_train.iloc[tr_idx],
            y_train_log.iloc[tr_idx],
            X_train.iloc[val_idx],
            y_train_log.iloc[val_idx],
            n_jobs=n_jobs,
            fast_mode=fast_mode,
        )

        for name, model in zip(["lgb", "xgb", "cat"], fold_models):
            oof[name][val_idx] = sanitize_log_predictions(model.predict(X_train.iloc[val_idx]))

    y_true = np.expm1(y_train_log.values)
    maes = {}
    for name, preds_log in oof.items():
        maes[name] = mean_absolute_error(y_true, np.expm1(sanitize_log_predictions(preds_log)))

    inv = {name: 1.0 / max(mae, 1.0) for name, mae in maes.items()}
    total = sum(inv.values())
    weights = {name: value / total for name, value in inv.items()}
    return weights, maes


def blend_predictions(prediction_map, weights):
    names = list(weights.keys())
    blended = np.zeros_like(next(iter(prediction_map.values())), dtype=float)
    for name in names:
        blended += weights[name] * sanitize_log_predictions(prediction_map[name])
    return blended


def sanitize_log_predictions(preds):
    preds = np.asarray(preds, dtype=float)
    preds = np.nan_to_num(preds, nan=0.0, posinf=16.5, neginf=0.0)
    return np.clip(preds, 0.0, 16.5)


def evaluate_prices(y_true_log, preds_log, label):
    y_true = np.expm1(sanitize_log_predictions(y_true_log))
    preds = np.expm1(sanitize_log_predictions(preds_log))
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    pct_error = np.abs(preds - y_true) / np.clip(y_true, 1, None) * 100
    medape = np.median(pct_error)
    within_10 = (pct_error <= 10).mean() * 100
    within_15 = (pct_error <= 15).mean() * 100
    within_20 = (pct_error <= 20).mean() * 100

    print(f"\n   {label}:")
    print(f"      MAE:    £{mae:,.0f}")
    print(f"      MedAPE: {medape:.2f}%")
    print(f"      R²:     {r2:.4f}")
    print(f"      Within 10%: {within_10:.1f}%")
    print(f"      Within 15%: {within_15:.1f}%")
    print(f"      Within 20%: {within_20:.1f}%")
    return {
        "mae": mae,
        "medape": medape,
        "r2": r2,
        "within_10": within_10,
        "within_15": within_15,
        "within_20": within_20,
    }


def main():
    input_source = os.getenv("AVM_INPUT_CSV", str(DEFAULT_INPUT))
    model_dir = Path(os.getenv("AVM_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    model_gcs_uri = os.getenv("AVM_MODEL_GCS_URI", "").strip()
    fast_mode = _env_bool("AVM_FAST_MODE", False)
    n_jobs = _env_int("AVM_N_JOBS", max(1, (os.cpu_count() or 2) - 1))
    min_location_coverage = _env_float("AVM_MIN_LOCATION_COVERAGE", 0.75)
    min_image_coverage = _env_float("AVM_MIN_IMAGE_COVERAGE", 0.75)

    model_dir.mkdir(parents=True, exist_ok=True)

    print("📂 Loading data...")
    input_path = _download_input(input_source)
    df = pd.read_csv(input_path)

    print("🧹 Cleaning data...")
    df["price"] = clean_price(df["price"])
    required = ["price", "property_type", "description", "city"]

    if {"latitude", "longitude"}.issubset(df.columns):
        valid_location = df[["latitude", "longitude"]].notna().all(axis=1)
        location_coverage = valid_location.mean()
        print(f"   Location coverage: {location_coverage * 100:.1f}%")
        if location_coverage >= min_location_coverage:
            required.extend(["latitude", "longitude"])
        else:
            df = df.drop(columns=["latitude", "longitude", "district"], errors="ignore")
            print(
                f"   Ignoring sparse location columns below {min_location_coverage * 100:.0f}% coverage."
            )

    if "has_image_embedding" in df.columns:
        image_coverage = pd.to_numeric(df["has_image_embedding"], errors="coerce").fillna(0).astype(bool).mean()
        print(f"   Image embedding coverage: {image_coverage * 100:.1f}%")
        if image_coverage < min_image_coverage:
            df = df.drop(columns=image_feature_columns(df), errors="ignore")
            print(f"   Ignoring image features below {min_image_coverage * 100:.0f}% coverage.")

    df = df.dropna(subset=required)
    df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce").fillna(3).clip(0, 8)
    df = df[(df["price"] >= 50000) & (df["price"] <= 4000000)]
    df["simple_type"] = df["property_type"].apply(simplify_type)
    df = df[df["simple_type"].isin(["Flat", "Detached", "Semi-Detached", "Bungalow", "Terraced", "Retirement"])]

    print(f"   Rows after cleaning: {len(df):,}")
    print(f"   Cities: {df['city'].nunique()} | Property types: {df['property_type'].nunique()}")

    holdout_strata = make_holdout_strata(df)
    train_raw, test_raw = train_test_split(df, test_size=0.2, random_state=42, stratify=holdout_strata)
    y_train_log = np.log1p(train_raw["price"])
    y_test_log = np.log1p(test_raw["price"])

    print("🔧 Building multimodal features...")
    X_train, X_test, preprocessors = assemble_features(train_raw, test_raw, y_train_log)
    feature_columns = list(X_train.columns)
    print(f"   Final feature count: {len(feature_columns)}")

    print("🎯 Learning blend weights from out-of-fold predictions...")
    blend_weights, oof_maes = build_oof_blend_weights(X_train, y_train_log, n_jobs=n_jobs, fast_mode=fast_mode, folds=3)
    print(
        f"   Blend weights: LGB={blend_weights['lgb']:.2f}, XGB={blend_weights['xgb']:.2f}, CAT={blend_weights['cat']:.2f}"
    )

    print("🚀 Training final base models on full training data...")
    lgb_model, xgb_model, cat_model = fit_final_models(X_train, y_train_log, n_jobs=n_jobs, fast_mode=fast_mode)

    print("\n" + "=" * 60)
    print("🏆 MODEL RESULTS")
    print("=" * 60)

    lgb_metrics = evaluate_prices(y_test_log, lgb_model.predict(X_test), "LightGBM")
    xgb_metrics = evaluate_prices(y_test_log, xgb_model.predict(X_test), "XGBoost")
    cat_metrics = evaluate_prices(y_test_log, cat_model.predict(X_test), "CatBoost")
    blend_predictions_map = {
        "lgb": lgb_model.predict(X_test),
        "xgb": xgb_model.predict(X_test),
        "cat": cat_model.predict(X_test),
    }
    blended_test_preds = blend_predictions(blend_predictions_map, blend_weights)
    blend_metrics = evaluate_prices(y_test_log, blended_test_preds, "Weighted blend")

    base_candidates = [
        ("lgb", lgb_model, lgb_metrics),
        ("xgb", xgb_model, xgb_metrics),
        ("cat", cat_model, cat_metrics),
    ]
    best_base = min(base_candidates, key=lambda item: item[2]["mae"])
    if blend_metrics["mae"] <= best_base[2]["mae"]:
        best_model = ("blend", blend_weights, blend_metrics)
    else:
        best_model = best_base
        print("   Blend did not beat the best base model on MAE; exporting the best base model.")

    best_model_name, best_model_obj, best_model_metrics = best_model
    print(f"\n🥇 Best model on holdout: {best_model_name.upper()}")

    if best_model_name == "blend":
        exported_model = {
            "type": "blend",
            "weights": blend_weights,
            "base_models": {
                "lgb": lgb_model,
                "xgb": xgb_model,
                "cat": cat_model,
            },
        }
    else:
        exported_model = {
            "type": best_model_name,
            "model": best_model_obj,
        }

    final_artifacts = {
        "model": exported_model,
        "model_name": best_model_name,
        "base_models": {
            "lgb": lgb_model,
            "xgb": xgb_model,
            "cat": cat_model,
        },
        "blend_weights": blend_weights,
        "oof_mae": oof_maes,
        "preprocessors": preprocessors,
        "feature_columns": feature_columns,
        "metrics": {
            "lgb": lgb_metrics,
            "xgb": xgb_metrics,
            "cat": cat_metrics,
            "blend": blend_metrics,
            "best": best_model_metrics,
        },
        "params": {
            "fast_mode": fast_mode,
            "n_jobs": n_jobs,
            "min_location_coverage": min_location_coverage,
            "min_image_coverage": min_image_coverage,
            "selection_metric": "mae",
        },
    }

    joblib.dump(final_artifacts, model_dir / "final_training_artifacts.joblib")
    joblib.dump(exported_model, model_dir / "final_model.joblib")
    joblib.dump(feature_columns, model_dir / "final_feature_columns.joblib")

    if model_gcs_uri:
        print(f"\n☁️ Uploading artifacts to {model_gcs_uri}...")
        _upload_outputs(model_dir, model_gcs_uri)

    print(f"\n💾 Saved artifacts to {model_dir}")


if __name__ == "__main__":
    main()
