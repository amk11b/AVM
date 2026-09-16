import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import re # <--- For text searching

# 1. Load Data
print("📂 Loading Visual & Text Data...")
df = pd.read_csv("/Users/AVM/AVM/modeling/embeddings.csv")

# 2. Advanced Cleaning
print("🧹 Cleaning Data...")

# Clean Price
if df['price'].dtype == 'object':
    df['price'] = df['price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])

# Filter Outliers
df = df[(df['price'] > 50000) & (df['price'] < 1000000)]

# Fix Bedrooms
if 'bedrooms' in df.columns:
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())

# Encode Categories
le = LabelEncoder()
for col in ['property_type', 'city']:
    if col in df.columns:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])

# --- NEW: TEXT MINING (The "Language" Mode) ---
print("📝 Extracting keywords from descriptions...")

# 1. Ensure description exists
if 'description' not in df.columns:
    df['description'] = ""
df['description'] = df['description'].fillna("").astype(str).str.lower()

# 2. Define Keywords
renovation_words = ['renovated', 'refurbished', 'modern', 'immaculate', 'stunning', 'newly', 'contemporary']
fixer_words = ['modernisation', 'potential', 'tlc', 'fixer', 'improvement', 'unmodernised', 'project']

# 3. Create Binary Features (1 if word exists, 0 if not)
def check_keywords(text, keywords):
    return 1 if any(word in text for word in keywords) else 0

df['is_modern'] = df['description'].apply(lambda x: check_keywords(x, renovation_words))
df['needs_work'] = df['description'].apply(lambda x: check_keywords(x, fixer_words))

print(f"   - found {df['is_modern'].sum()} 'modern' homes")
print(f"   - found {df['needs_work'].sum()} 'fixer-uppers'")

# 3. Separate Features
drop_cols = ['url', 'main_image_url', 'scraped_at', 'address', 'description', 'title', 'price']
features = df.drop(columns=drop_cols, errors='ignore')

# Identify Columns
visual_cols = [c for c in features.columns if c.startswith('img_dim_')]
# Now 'is_modern' and 'needs_work' are part of fact_cols automatically
fact_cols = [c for c in features.columns if not c.startswith('img_dim_')]

print(f"   - Facts found: {fact_cols}")

# --- Compress Visuals (PCA) ---
print("📉 Compressing visual features...")
pca = PCA(n_components=30, random_state=42)
visual_data = features[visual_cols]
compressed_visuals = pca.fit_transform(visual_data)

X_visuals = pd.DataFrame(compressed_visuals, columns=[f'pca_{i}' for i in range(30)], index=features.index)
X_facts = features[fact_cols]
X = pd.concat([X_facts, X_visuals], axis=1)

y = np.log1p(df['price'])

# 4. Train
print(f"🧠 Training on {len(X)} rows with {X.shape[1]} features")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# 5. Evaluate
preds_log = model.predict(X_test)
predictions = np.expm1(preds_log)
y_test_actual = np.expm1(y_test)

print("\n" + "="*30)
print(f"👁️ + 📝 MULTI-MODAL MODEL RESULTS")
if len(y_test_actual) > 0:
    mae = mean_absolute_error(y_test_actual, predictions)
    r2 = r2_score(y_test_actual, predictions)

    print(f"Mean Absolute Error: £{mae:,.0f}")
    print(f"R² Score:            {r2:.2f}")

    print("\n🏆 Top 5 Drivers:")
    importance = pd.Series(model.feature_importances_, index=X.columns)
    print(importance.nlargest(5))
else:
    print("Not enough data.")
print("="*30 + "\n")