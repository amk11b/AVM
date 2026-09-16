import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import final_training_v2 as ft


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_INPUT = BASE_DIR / "embeddings.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "models" / "local"
KEY_PATH = Path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", str(PROJECT_ROOT / "gcp_key.json")))


if KEY_PATH.exists() and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(KEY_PATH)


def _env_int(name, default):
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name, default):
    value = os.getenv(name)
    return float(value) if value is not None else default


def clean_training_frame(input_source, min_location_coverage):
    input_path = ft._download_input(input_source)
    df = pd.read_csv(input_path, low_memory=False)

    df["price"] = ft.clean_price(df["price"])
    required = ["price", "property_type", "description", "city"]

    if {"latitude", "longitude"}.issubset(df.columns):
        valid_location = df[["latitude", "longitude"]].notna().all(axis=1)
        location_coverage = valid_location.mean()
        print(f"   Location coverage: {location_coverage * 100:.1f}%")
        if location_coverage >= min_location_coverage:
            required.extend(["latitude", "longitude"])
        else:
            df = df.drop(columns=["latitude", "longitude", "district"], errors="ignore")
            print(f"   Ignoring sparse location columns below {min_location_coverage * 100:.0f}% coverage.")

    df = df.dropna(subset=required)
    df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce").fillna(3).clip(0, 8)
    df = df[(df["price"] >= 50000) & (df["price"] <= 4000000)]
    df["simple_type"] = df["property_type"].apply(ft.simplify_type)
    df = df[df["simple_type"].isin(["Flat", "Detached", "Semi-Detached", "Bungalow", "Terraced", "Retirement"])]

    if df.empty:
        raise ValueError("No rows left after cleaning. Check the input CSV and required columns.")

    return df


def make_xgb_model(n_jobs):
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=_env_int("AVM_ABLATION_TREES", 900),
        learning_rate=_env_float("AVM_ABLATION_LEARNING_RATE", 0.03),
        max_depth=_env_int("AVM_ABLATION_MAX_DEPTH", 6),
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


def metrics_for(y_true_log, preds_log):
    y_true = np.expm1(ft.sanitize_log_predictions(y_true_log))
    preds = np.expm1(ft.sanitize_log_predictions(preds_log))
    pct_error = np.abs(preds - y_true) / np.clip(y_true, 1, None) * 100
    return {
        "mae": mean_absolute_error(y_true, preds),
        "medape": np.median(pct_error),
        "r2": r2_score(y_true, preds),
        "within_10": (pct_error <= 10).mean() * 100,
        "within_15": (pct_error <= 15).mean() * 100,
        "within_20": (pct_error <= 20).mean() * 100,
    }


def feature_groups(columns):
    cols = pd.Index(columns)

    text_svd = {c for c in cols if c.startswith("text_svd_")}
    text_flags = {
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
    } & set(cols)
    image = {c for c in cols if c.startswith("img_")} | ({"has_image_embedding"} & set(cols))
    comps = {
        c
        for c in cols
        if c.startswith(("sector_", "city_", "comp_close_", "comp_medium_", "comp_wide_"))
        or c in {"best_comp_price", "comp_count", "comp_source_te"}
    }
    target_encoding = {c for c in cols if c.endswith("_te")}
    location = {
        "latitude",
        "longitude",
        "lat_sin",
        "lat_cos",
        "lon_sin",
        "lon_cos",
        "lat_lon_interaction",
        "distance_from_median_lat",
        "distance_from_median_lon",
        "location_cluster",
        "district_te",
    } & set(cols)

    text_all = text_svd | text_flags
    text_image = text_all | image

    return {
        "all_features": set(cols),
        "no_text_svd": set(cols) - text_svd,
        "no_text_flags": set(cols) - text_flags,
        "no_all_text": set(cols) - text_all,
        "no_image": set(cols) - image,
        "no_comps": set(cols) - comps,
        "no_target_encoding": set(cols) - target_encoding,
        "no_location": set(cols) - location,
        "tabular_no_text_no_image": set(cols) - text_image,
    }


def run_feature_set(name, selected_columns, X_train, X_test, y_train_log, y_test_log, n_jobs):
    selected_columns = [c for c in X_train.columns if c in selected_columns]
    if not selected_columns:
        raise ValueError(f"Feature set {name} has no columns.")

    model = make_xgb_model(n_jobs)
    model.fit(X_train[selected_columns], y_train_log, eval_set=[(X_test[selected_columns], y_test_log)], verbose=False)
    row = metrics_for(y_test_log, model.predict(X_test[selected_columns]))
    row["feature_set"] = name
    row["feature_count"] = len(selected_columns)
    return row


def main():
    input_source = os.getenv("AVM_ABLATION_INPUT_CSV", os.getenv("AVM_INPUT_CSV", str(DEFAULT_INPUT)))
    report_dir = Path(os.getenv("AVM_ABLATION_REPORT_DIR", str(DEFAULT_REPORT_DIR)))
    n_jobs = _env_int("AVM_N_JOBS", max(1, (os.cpu_count() or 2) - 1))
    min_location_coverage = _env_float("AVM_MIN_LOCATION_COVERAGE", 0.75)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("📂 Loading data for ablation...")
    df = clean_training_frame(input_source, min_location_coverage)
    print(f"   Rows after cleaning: {len(df):,}")
    print(f"   Cities: {df['city'].nunique()} | Property types: {df['property_type'].nunique()}")

    holdout_strata = ft.make_holdout_strata(df)
    train_raw, test_raw = train_test_split(df, test_size=0.2, random_state=42, stratify=holdout_strata)
    y_train_log = np.log1p(train_raw["price"])
    y_test_log = np.log1p(test_raw["price"])

    print("🔧 Building shared feature matrix...")
    X_train, X_test, _ = ft.assemble_features(train_raw, test_raw, y_train_log)
    print(f"   Full feature count: {len(X_train.columns)}")

    groups = feature_groups(X_train.columns)
    rows = []
    print("🧪 Running XGBoost feature ablations...")
    for name, selected_columns in groups.items():
        print(f"   {name} ({len(selected_columns)} features)...")
        rows.append(run_feature_set(name, selected_columns, X_train, X_test, y_train_log, y_test_log, n_jobs))

    report = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    baseline_mae = report.loc[report["feature_set"] == "all_features", "mae"].iloc[0]
    report["mae_delta_vs_all"] = report["mae"] - baseline_mae

    report_path = report_dir / "ablation_report.csv"
    report.to_csv(report_path, index=False)

    print("\n" + "=" * 72)
    print("🏁 ABLATION RESULTS")
    print("=" * 72)
    for _, row in report.iterrows():
        print(
            f"{row['feature_set']:<24} "
            f"MAE £{row['mae']:>8,.0f} | "
            f"MedAPE {row['medape']:>5.2f}% | "
            f"R² {row['r2']:>6.4f} | "
            f"ΔMAE £{row['mae_delta_vs_all']:>+8,.0f} | "
            f"{int(row['feature_count'])} features"
        )
    print(f"\n💾 Saved report to {report_path}")


if __name__ == "__main__":
    main()
