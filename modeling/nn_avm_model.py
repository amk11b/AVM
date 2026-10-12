import os
import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import gc
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
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
img_cols = sorted([f.name for f in table.schema if f.name.startswith('img_dim_')],
                  key=lambda x: int(x.split('_')[-1]))

query = f"""
SELECT 
    price, bedrooms, property_type, city, latitude, longitude,
    is_modern, needs_work, has_garden, has_parking, has_views,
    is_quiet_location, near_transport, near_schools,
    is_period_property, is_new_build, is_investment,
    floor_area, energy_efficiency, best_comp_price_v2, 
    city_median_price, comp_quality_score, comp_source_v2,
    {", ".join(img_cols)}
FROM `{PROJECT_ID}.{DATASET}.master_training_data_v3`
WHERE price BETWEEN 50000 AND 5000000
  AND bedrooms BETWEEN 1 AND 10
  AND property_type IN ('Houses', 'Flats / Apartments', 'Bungalows')
"""

df = client.query(query).to_dataframe()
print(f"✅ Loaded {len(df):,} properties")

# Fill NaN
df['floor_area'] = df['floor_area'].fillna(df['floor_area'].median())
df['energy_efficiency'] = df['energy_efficiency'].fillna(60)
df['best_comp_price_v2'] = df['best_comp_price_v2'].fillna(df['city_median_price'])
for col in img_cols:
    df[col] = df[col].fillna(0)

# --- 2. FAST SYNTHETIC DATA (vectorized) ---
print("\n🔧 Generating synthetic data (fast method)...")

n_synthetic = 25000  # Smaller but sufficient

# Instead of loop, use vectorized sampling
np.random.seed(42)
idx1 = np.random.randint(0, len(df), n_synthetic)
idx2 = np.random.randint(0, len(df), n_synthetic)
alpha = np.random.uniform(0.3, 0.7, n_synthetic).reshape(-1, 1)

# Numeric columns to interpolate
numeric_cols = ['price', 'bedrooms', 'floor_area', 'energy_efficiency',
                'latitude', 'longitude', 'best_comp_price_v2', 'city_median_price'] + img_cols

df_num = df[numeric_cols].values.astype(np.float32)

synthetic_num = alpha * df_num[idx1] + (1 - alpha) * df_num[idx2]

# Add noise to price
price_noise = np.random.uniform(0.9, 1.1, n_synthetic)
synthetic_num[:, 0] *= price_noise  # price is column 0

# Round bedrooms
synthetic_num[:, 1] = np.clip(np.round(synthetic_num[:, 1]), 1, 10)

# Create synthetic dataframe
df_synthetic = pd.DataFrame(synthetic_num, columns=numeric_cols)

# Copy categorical from base
df_synthetic['property_type'] = df.iloc[idx1]['property_type'].values
df_synthetic['city'] = df.iloc[idx1]['city'].values
df_synthetic['comp_source_v2'] = df.iloc[idx1]['comp_source_v2'].values
df_synthetic['comp_quality_score'] = df.iloc[idx1]['comp_quality_score'].values

# Random binary features
for col in ['is_modern', 'needs_work', 'has_garden', 'has_parking', 'has_views',
            'is_quiet_location', 'near_transport', 'near_schools',
            'is_period_property', 'is_new_build', 'is_investment']:
    prob = df[col].mean()
    df_synthetic[col] = (np.random.random(n_synthetic) < prob).astype(int)

df_synthetic['is_synthetic'] = 1
df['is_synthetic'] = 0

df_combined = pd.concat([df, df_synthetic], ignore_index=True)
print(f"✅ Combined: {len(df_combined):,} (Real: {len(df):,}, Synthetic: {len(df_synthetic):,})")

del df_synthetic, synthetic_num
gc.collect()

# --- 3. FEATURES ---
print("\n🔧 Preparing features...")

le_pt = LabelEncoder()
le_city = LabelEncoder()

df_combined['property_type_enc'] = le_pt.fit_transform(df_combined['property_type'].fillna('Unknown'))
df_combined['city_enc'] = le_city.fit_transform(df_combined['city'].fillna('Unknown'))
df_combined['comp_per_bed'] = df_combined['best_comp_price_v2'] / df_combined['bedrooms'].clip(lower=1)

features = ['bedrooms', 'floor_area', 'energy_efficiency', 'latitude', 'longitude',
            'property_type_enc', 'city_enc', 'best_comp_price_v2', 'city_median_price',
            'comp_per_bed', 'comp_quality_score',
            'is_modern', 'needs_work', 'has_garden', 'has_parking', 'has_views',
            'is_quiet_location', 'near_transport', 'near_schools',
            'is_period_property', 'is_new_build', 'is_investment']

# PCA
print("   PCA on embeddings...")
emb_data = df_combined[img_cols].values.astype(np.float32)
pca = PCA(n_components=30, random_state=42)
emb_pca = pca.fit_transform(emb_data)
print(f"   Variance explained: {pca.explained_variance_ratio_.sum() * 100:.1f}%")

del emb_data
gc.collect()

# Prepare arrays
X = np.hstack([df_combined[features].values.astype(np.float32), emb_pca])
y = df_combined['price'].values.astype(np.float32)
is_synth = df_combined['is_synthetic'].values

# Scale
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Split
real_mask = is_synth == 0
X_real, y_real, y_real_orig = X[real_mask], y_scaled[real_mask], y[real_mask]
X_synth, y_synth = X[~real_mask], y_scaled[~real_mask]

X_r_train, X_r_test, y_r_train, y_r_test, _, y_test_orig = train_test_split(
    X_real, y_real, y_real_orig, test_size=0.2, random_state=42)

X_train = np.vstack([X_r_train, X_synth]).astype(np.float32)
y_train = np.concatenate([y_r_train, y_synth]).astype(np.float32)
X_test = X_r_test.astype(np.float32)
y_test = y_r_test.astype(np.float32)

print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

del df_combined, X_real, y_real, X_synth, y_synth
gc.collect()

# --- 4. NEURAL NETWORK (CPU, small) ---
print("\n" + "=" * 50)
print("🧠 TRAINING NEURAL NETWORK")
print("=" * 50)


class SmallNN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


device = torch.device('cpu')  # Force CPU
print(f"   Device: {device}")

model = SmallNN(X_train.shape[1]).to(device)
criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
    batch_size=1024, shuffle=True
)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)

best_loss = float('inf')
patience = 0

print("   Training...")
for epoch in range(80):
    model.train()
    for bx, by in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(bx).squeeze(), by)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_test_t).squeeze(), y_test_t).item()

    if val_loss < best_loss:
        best_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience = 0
    else:
        patience += 1

    if (epoch + 1) % 20 == 0:
        print(f"      Epoch {epoch + 1}: val_loss={val_loss:.4f}")

    if patience >= 15:
        print(f"      Early stop at epoch {epoch + 1}")
        break

model.load_state_dict(best_state)

# --- 5. LIGHTGBM ---
print("\n" + "=" * 50)
print("🌲 TRAINING LIGHTGBM")
print("=" * 50)

lgb_model = lgb.LGBMRegressor(
    n_estimators=800, learning_rate=0.03, max_depth=6,
    num_leaves=40, min_child_samples=30, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, n_jobs=-1, verbose=-1
)
lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False)])

# --- 6. EVALUATE ---
print("\n" + "=" * 50)
print("📊 RESULTS")
print("=" * 50)


def eval_model(y_true, y_pred, name):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    medape = np.median(np.abs(y_pred - y_true) / y_true) * 100
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    w15 = (np.abs(y_pred - y_true) / y_true <= 0.15).mean() * 100
    w20 = (np.abs(y_pred - y_true) / y_true <= 0.20).mean() * 100

    print(f"\n{name}:")
    print(f"   MAPE: {mape:.1f}% | MedAPE: {medape:.1f}% | MAE: £{mae:,.0f}")
    print(f"   R²: {r2:.3f} | Within 15%: {w15:.1f}% | Within 20%: {w20:.1f}%")
    return mape, medape, w15, w20


# Predictions
model.eval()
with torch.no_grad():
    nn_pred = scaler_y.inverse_transform(model(X_test_t).cpu().numpy()).flatten()

lgb_pred = scaler_y.inverse_transform(lgb_model.predict(X_test).reshape(-1, 1)).flatten()
ens_pred = (nn_pred + lgb_pred) / 2

y_true = y_test_orig

nn_m = eval_model(y_true, nn_pred, "Neural Network")
lgb_m = eval_model(y_true, lgb_pred, "LightGBM")
ens_m = eval_model(y_true, ens_pred, "Ensemble")

# --- 7. SAVE ---
print("\n💾 Saving...")
torch.save({'state': model.state_dict(), 'scaler_X': scaler_X, 'scaler_y': scaler_y,
            'pca': pca, 'features': features, 'le_pt': le_pt, 'le_city': le_city},
           f"{MODEL_DIR}/nn_final.pt")
joblib.dump(lgb_model, f"{MODEL_DIR}/lgb_final.joblib")

# --- 8. SAMPLES ---
print("\n🏠 Samples:")
for i in np.random.choice(len(y_true), 8, replace=False):
    print(
        f"   Actual: £{y_true[i]:>9,.0f} | Ensemble: £{ens_pred[i]:>9,.0f} | Err: {abs(ens_pred[i] - y_true[i]) / y_true[i] * 100:>5.1f}%")

print("\n✅ Done!")