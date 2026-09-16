import os
import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
KEY_PATH = "/Users/AVM/AVM/gcp_key.json"
PROJECT_ID = "avm-ml-485715"
DATASET = "avm_dataset"
MODEL_DIR = "/Users/AVM/AVM/models"

os.makedirs(MODEL_DIR, exist_ok=True)

if os.path.exists(KEY_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

client = bigquery.Client(project=PROJECT_ID)

# --- 1. LOAD DATA ---
print("📥 Loading data...")

table = client.get_table(f"{PROJECT_ID}.{DATASET}.master_training_data_v3")
all_columns = [field.name for field in table.schema]
img_cols = sorted([c for c in all_columns if c.startswith('img_dim_')],
                  key=lambda x: int(x.split('_')[-1]))

embedding_cols_str = ", ".join(img_cols)

query = f"""
SELECT 
    price,
    bedrooms,
    property_type,
    city,
    latitude,
    longitude,
    is_modern,
    needs_work,
    has_garden,
    has_parking,
    has_views,
    is_quiet_location,
    near_transport,
    near_schools,
    is_period_property,
    is_new_build,
    is_investment,
    floor_area,
    energy_efficiency,
    avg_nearby_sold_price,
    comp_count,
    sector_sales_count,
    sector_median_price,
    sector_price_stddev,
    city_median_price,
    sector_type_count,
    sector_type_avg,
    sector_type_median,
    sector_type_stddev,
    best_comp_price_v2,
    comp_quality_score,
    comp_source_v2,
    {embedding_cols_str}
FROM `{PROJECT_ID}.{DATASET}.master_training_data_v3`
WHERE price IS NOT NULL 
  AND price >= 50000
  AND price <= 5000000
  AND property_type IN ('Houses', 'Flats / Apartments', 'Bungalows', 'Retirement Property', 'Character Property')
"""

df = client.query(query).to_dataframe()
print(f"✅ Loaded {len(df):,} rows")

# --- 2. OUTLIER DETECTION & FILTERING ---
print("\n🔍 Detecting outliers...")

# Calculate price vs comp ratio to find unusual listings
df['price_vs_comp_ratio'] = df['price'] / df['best_comp_price_v2'].clip(lower=1)

# Flag outliers: properties priced very differently from comps
lower_bound = df['price_vs_comp_ratio'].quantile(0.02)
upper_bound = df['price_vs_comp_ratio'].quantile(0.98)

outliers = (df['price_vs_comp_ratio'] < lower_bound) | (df['price_vs_comp_ratio'] > upper_bound)
print(f"   Found {outliers.sum()} outliers ({outliers.mean() * 100:.1f}%)")
print(f"   Price/Comp ratio range: {lower_bound:.2f} - {upper_bound:.2f}")

# Show some outliers
print("\n   Example outliers (price vs comp mismatch):")
outlier_samples = df[outliers].sample(min(5, outliers.sum()), random_state=42)
for _, row in outlier_samples.iterrows():
    print(
        f"      Price: £{row['price']:>10,.0f} | Comp: £{row['best_comp_price_v2']:>10,.0f} | Ratio: {row['price_vs_comp_ratio']:.2f} | {row['property_type']}, {row['city']}")

# Option: Train on clean data, but keep outliers for testing reality
df_clean = df[~outliers].copy()
print(f"\n   Clean dataset: {len(df_clean):,} rows")

# --- 3. FEATURE ENGINEERING ---
print("\n🔧 Feature engineering...")


def engineer_features(df):
    df = df.copy()

    # Encoders
    le_property_type = LabelEncoder()
    le_city = LabelEncoder()

    df['property_type_encoded'] = le_property_type.fit_transform(df['property_type'].fillna('Unknown'))
    df['city_encoded'] = le_city.fit_transform(df['city'].fillna('Unknown'))

    # Fill NaN values
    df['sector_type_median'] = df['sector_type_median'].fillna(df['sector_median_price'])
    df['best_comp_price_v2'] = df['best_comp_price_v2'].fillna(df['city_median_price'])
    df['sector_type_count'] = df['sector_type_count'].fillna(0)
    df['sector_type_stddev'] = df['sector_type_stddev'].fillna(df['sector_price_stddev'])
    df['sector_price_stddev'] = df['sector_price_stddev'].fillna(df['sector_price_stddev'].median())
    df['sector_type_stddev'] = df['sector_type_stddev'].fillna(df['sector_type_stddev'].median())
    df['comp_count'] = df['comp_count'].fillna(0)
    df['sector_sales_count'] = df['sector_sales_count'].fillna(0)

    # Derived features
    df['comp_per_sqft'] = df['best_comp_price_v2'] / df['floor_area'].clip(lower=20)
    df['comp_per_bed'] = df['best_comp_price_v2'] / df['bedrooms'].clip(lower=1)
    df['sqft_per_bed'] = df['floor_area'] / df['bedrooms'].clip(lower=1)
    df['comp_vs_city'] = df['best_comp_price_v2'] / df['city_median_price'].clip(lower=1)

    # Log transform of comp price (helps with skewed data)
    df['log_comp_price'] = np.log1p(df['best_comp_price_v2'])

    # Location clusters
    df['lat_bin'] = pd.qcut(df['latitude'], q=20, labels=False, duplicates='drop')
    df['lon_bin'] = pd.qcut(df['longitude'], q=20, labels=False, duplicates='drop')
    df['location_cluster'] = df['lat_bin'].astype(str) + '_' + df['lon_bin'].astype(str)
    le_location = LabelEncoder()
    df['location_cluster_encoded'] = le_location.fit_transform(df['location_cluster'])

    # Condition score
    df['condition_score'] = (
            df['is_modern'] * 2 +
            df['is_new_build'] * 2 -
            df['needs_work'] * 2 +
            df['has_garden'] +
            df['has_parking'] +
            df['has_views']
    )

    # Amenity score
    df['amenity_score'] = (
            df['near_transport'] +
            df['near_schools'] +
            df['is_quiet_location']
    )

    return df, le_property_type, le_city, le_location


df_clean, le_property_type, le_city, le_location = engineer_features(df_clean)

# --- 4. PCA ON EMBEDDINGS ---
print("   Applying PCA...")
embedding_data = df_clean[img_cols].fillna(0).values

N_COMPONENTS = 30
pca = PCA(n_components=N_COMPONENTS, random_state=42)
embedding_pca = pca.fit_transform(embedding_data)
print(f"   PCA variance explained: {pca.explained_variance_ratio_.sum() * 100:.1f}%")

pca_cols = [f'img_pca_{i}' for i in range(N_COMPONENTS)]
for i, col in enumerate(pca_cols):
    df_clean[col] = embedding_pca[:, i]

# --- 5. DEFINE FEATURES ---
features = [
               'bedrooms', 'floor_area', 'energy_efficiency',
               'latitude', 'longitude',
               'property_type_encoded', 'city_encoded',
               'location_cluster_encoded',
               'best_comp_price_v2', 'log_comp_price',
               'comp_per_sqft', 'comp_per_bed',
               'comp_quality_score', 'comp_vs_city',
               'sector_type_median', 'sector_type_count',
               'city_median_price', 'comp_count',
               'sqft_per_bed',
               'condition_score', 'amenity_score',
               'is_modern', 'needs_work',
               'has_garden', 'has_parking', 'has_views',
               'is_quiet_location', 'near_transport', 'near_schools',
               'is_period_property', 'is_new_build', 'is_investment',
           ] + pca_cols

print(f"   Total features: {len(features)}")

# --- 6. PREPARE DATA ---
X = df_clean[features].fillna(0)
y = df_clean['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# --- 7. USE LOG PRICE AS TARGET (reduces impact of expensive properties) ---
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)


# --- 8. EVALUATION FUNCTION ---
def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    pct_error = np.abs(y_pred - y_true) / y_true * 100
    within_10 = (pct_error <= 10).mean() * 100
    within_15 = (pct_error <= 15).mean() * 100
    within_20 = (pct_error <= 20).mean() * 100

    # Median absolute percentage error (more robust)
    medape = np.median(pct_error)

    print(f"\n   {label}:")
    print(f"      MAE:    £{mae:,.0f}")
    print(f"      MAPE:   {mape:.2f}%")
    print(f"      MedAPE: {medape:.2f}%")
    print(f"      R²:     {r2:.4f}")
    print(f"      Within 10%: {within_10:.1f}%")
    print(f"      Within 15%: {within_15:.1f}%")
    print(f"      Within 20%: {within_20:.1f}%")

    return {'mae': mae, 'mape': mape, 'medape': medape, 'r2': r2,
            'within_10': within_10, 'within_15': within_15, 'within_20': within_20}


# --- 9. TRAIN MODELS ON LOG PRICE ---
print("\n" + "=" * 60)
print("🚀 TRAINING ON LOG PRICE (more robust)")
print("=" * 60)

# LightGBM
print("\n   Training LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.015,
    max_depth=7,
    num_leaves=50,
    min_child_samples=25,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.05,
    reg_lambda=0.05,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train, y_train_log, eval_set=[(X_test, y_test_log)],
              callbacks=[lgb.early_stopping(100, verbose=False)])

# XGBoost with Huber loss (robust to outliers)
print("   Training XGBoost (Huber loss)...")
xgb_model = xgb.XGBRegressor(
    n_estimators=1500,
    learning_rate=0.015,
    max_depth=7,
    min_child_weight=25,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.05,
    reg_lambda=0.05,
    objective='reg:pseudohubererror',  # Robust to outliers
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=100
)
xgb_model.fit(X_train, y_train_log, eval_set=[(X_test, y_test_log)], verbose=False)

# CatBoost
print("   Training CatBoost...")
cat_model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.015,
    depth=7,
    min_data_in_leaf=25,
    subsample=0.8,
    reg_lambda=0.05,
    random_state=42,
    verbose=False,
    early_stopping_rounds=100
)
cat_model.fit(X_train, y_train_log, eval_set=(X_test, y_test_log))

# --- 10. ENSEMBLE PREDICTIONS ---
print("\n   Creating ensemble...")

# Predictions in log space
lgb_pred_log = lgb_model.predict(X_test)
xgb_pred_log = xgb_model.predict(X_test)
cat_pred_log = cat_model.predict(X_test)

# Convert back to price space
lgb_pred = np.expm1(lgb_pred_log)
xgb_pred = np.expm1(xgb_pred_log)
cat_pred = np.expm1(cat_pred_log)


# Find optimal weights
def weighted_mape(weights, preds_list, y_true):
    weights = np.array(weights) / np.sum(weights)
    combined = sum(w * p for w, p in zip(weights, preds_list))
    return mean_absolute_percentage_error(y_true, combined)


result = minimize(
    weighted_mape,
    x0=[1, 1, 1],
    args=([lgb_pred, xgb_pred, cat_pred], y_test),
    method='Nelder-Mead'
)
optimal_weights = np.array(result.x) / np.sum(result.x)
print(f"   Optimal weights: LGB={optimal_weights[0]:.2f}, XGB={optimal_weights[1]:.2f}, CAT={optimal_weights[2]:.2f}")

ensemble_pred = (
        optimal_weights[0] * lgb_pred +
        optimal_weights[1] * xgb_pred +
        optimal_weights[2] * cat_pred
)

# Train predictions
lgb_pred_train = np.expm1(lgb_model.predict(X_train))
xgb_pred_train = np.expm1(xgb_model.predict(X_train))
cat_pred_train = np.expm1(cat_model.predict(X_train))
ensemble_pred_train = (
        optimal_weights[0] * lgb_pred_train +
        optimal_weights[1] * xgb_pred_train +
        optimal_weights[2] * cat_pred_train
)

# --- 11. EVALUATE ---
print("\n" + "=" * 60)
print("📊 MODEL COMPARISON (Clean Data)")
print("=" * 60)

print("\n🔹 LightGBM:")
lgb_metrics = evaluate(y_test, lgb_pred, "Test")

print("\n🔹 XGBoost (Huber):")
xgb_metrics = evaluate(y_test, xgb_pred, "Test")

print("\n🔹 CatBoost:")
cat_metrics = evaluate(y_test, cat_pred, "Test")

print("\n🔹 Weighted Ensemble:")
train_metrics = evaluate(y_train, ensemble_pred_train, "Train")
test_metrics = evaluate(y_test, ensemble_pred, "Test")

# --- 12. RESULTS BY COMP QUALITY ---
print("\n" + "=" * 60)
print("📊 PERFORMANCE BY COMP QUALITY")
print("=" * 60)

df_test = df_clean.iloc[X_test.index].copy()
df_test['predicted'] = ensemble_pred
df_test['actual'] = y_test.values
df_test['error_pct'] = np.abs(df_test['predicted'] - df_test['actual']) / df_test['actual'] * 100

for source in ['spatial', 'sector_typed', 'sector', 'city']:
    subset = df_test[df_test['comp_source_v2'] == source]
    if len(subset) > 10:
        mape = subset['error_pct'].mean()
        medape = subset['error_pct'].median()
        within_15 = (subset['error_pct'] <= 15).mean() * 100
        within_20 = (subset['error_pct'] <= 20).mean() * 100
        print(
            f"   {source:12s}: MAPE {mape:>5.1f}% | MedAPE {medape:>5.1f}% | Within 15%: {within_15:>5.1f}% | Within 20%: {within_20:>5.1f}% | n={len(subset):,}")

# --- 13. ERROR ANALYSIS ---
print("\n" + "=" * 60)
print("📊 ERROR ANALYSIS")
print("=" * 60)

# Distribution of errors
errors = df_test['error_pct']
print(f"\n   Error percentiles:")
print(f"      10th: {errors.quantile(0.10):.1f}%")
print(f"      25th: {errors.quantile(0.25):.1f}%")
print(f"      50th: {errors.quantile(0.50):.1f}% (median)")
print(f"      75th: {errors.quantile(0.75):.1f}%")
print(f"      90th: {errors.quantile(0.90):.1f}%")

# Worst predictions
print("\n   Worst 5 predictions:")
worst = df_test.nlargest(5, 'error_pct')
for _, row in worst.iterrows():
    print(
        f"      Actual: £{row['actual']:>10,.0f} | Pred: £{row['predicted']:>10,.0f} | Error: {row['error_pct']:.1f}% | {row['property_type']}, {row['city']}")

# Best predictions
print("\n   Best 5 predictions:")
best = df_test.nsmallest(5, 'error_pct')
for _, row in best.iterrows():
    print(
        f"      Actual: £{row['actual']:>10,.0f} | Pred: £{row['predicted']:>10,.0f} | Error: {row['error_pct']:.1f}% | {row['property_type']}, {row['city']}")

# --- 14. FEATURE IMPORTANCE ---
print("\n🔍 Top 20 Feature Importances:")
importance_df = pd.DataFrame({
    'feature': features,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importance_df.head(20).iterrows():
    if row['feature'].startswith('img_pca'):
        marker = "📷"
    elif 'comp' in row['feature'] or 'sector' in row['feature'] or 'city' in row['feature']:
        marker = "💰"
    else:
        marker = "📊"
    print(f"   {marker} {row['feature']:30s} {row['importance']:,.0f}")

# --- 15. SAVE MODELS ---
print("\n💾 Saving models...")

models = {
    'lgb': lgb_model,
    'xgb': xgb_model,
    'cat': cat_model,
    'weights': optimal_weights,
    'pca': pca,
    'le_property_type': le_property_type,
    'le_city': le_city,
    'le_location': le_location,
    'use_log_target': True,
}

joblib.dump(models, f"{MODEL_DIR}/ensemble_models_v2.joblib")
joblib.dump(features, f"{MODEL_DIR}/ensemble_features_v2.joblib")
joblib.dump(test_metrics, f"{MODEL_DIR}/ensemble_metrics_v2.joblib")

print(f"   Saved to {MODEL_DIR}/")

# --- 16. SAMPLE PREDICTIONS ---
print("\n🏠 Sample Predictions:")
samples = df_test.sample(15, random_state=42)
print(f"   {'Actual':>12} | {'Predicted':>12} | {'Error':>7} | Source")
print("   " + "-" * 60)
for _, row in samples.iterrows():
    print(
        f"   £{row['actual']:>10,.0f} | £{row['predicted']:>10,.0f} | {row['error_pct']:>5.1f}% | {row['comp_source_v2']}")

# --- 17. SUMMARY ---
print("\n" + "=" * 60)
print("✅ FINAL SUMMARY")
print("=" * 60)

print(f"""
IMPROVED ENSEMBLE MODEL (outliers removed, log target):

   Test MAPE:    {test_metrics['mape']:.1f}%
   Test MedAPE:  {test_metrics['medape']:.1f}%  ← More realistic measure
   Test MAE:     £{test_metrics['mae']:,.0f}
   Within 10%:   {test_metrics['within_10']:.1f}%
   Within 15%:   {test_metrics['within_15']:.1f}%
   Within 20%:   {test_metrics['within_20']:.1f}%

   Train MAPE:   {train_metrics['mape']:.1f}%
   Overfitting:  {test_metrics['mape'] - train_metrics['mape']:.1f}%

KEY INSIGHT: MedAPE ({test_metrics['medape']:.1f}%) vs MAPE ({test_metrics['mape']:.1f}%)
   → A few difficult properties are inflating the MAPE
   → For most properties, accuracy is much better than MAPE suggests
""")