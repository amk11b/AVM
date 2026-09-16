from google.cloud import bigquery
import os

KEY_PATH = "/Users/AVM/AVM/gcp_key.json"
PROJECT_ID = "avm-ml-485715"
DATASET = "avm_dataset"

if os.path.exists(KEY_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

client = bigquery.Client(project=PROJECT_ID)

# Get the actual image embedding columns
table = client.get_table(f"{PROJECT_ID}.{DATASET}.listings_raw")
columns = [field.name for field in table.schema]
img_cols = sorted([c for c in columns if c.startswith('img_dim_')],
                  key=lambda x: int(x.split('_')[-1]))

print(f"Found {len(img_cols)} image embedding columns")

embedding_cols = ", ".join([f"l.{col}" for col in img_cols])

query = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.master_training_data` AS
WITH 
listings AS (
    SELECT 
        *,
        SAFE_CAST(price AS FLOAT64) AS price_numeric,
        SAFE_CAST(bedrooms AS INT64) AS bedrooms_int,
        SAFE_CAST(latitude AS FLOAT64) AS lat,
        SAFE_CAST(longitude AS FLOAT64) AS lon,
        ST_GEOGPOINT(SAFE_CAST(longitude AS FLOAT64), SAFE_CAST(latitude AS FLOAT64)) AS listing_geom,
        REPLACE(UPPER(REGEXP_EXTRACT(address, r'([A-Za-z]{{1,2}}\\d{{1,2}}\\s?\\d[A-Za-z]{{2}})')), ' ', '') AS postcode_clean,

        -- Condition signals
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'modern|renovated|refurbished|immaculate|luxury|stunning|contemporary') THEN 1 ELSE 0 END AS is_modern,
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'needs work|fixer|tlc|modernisation|potential|project|dated|tired') THEN 1 ELSE 0 END AS needs_work,

        -- Property features
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'garden|patio|terrace|outdoor space') THEN 1 ELSE 0 END AS has_garden,
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'garage|parking|driveway|off-street') THEN 1 ELSE 0 END AS has_parking,
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'view|views|overlook|panoramic') THEN 1 ELSE 0 END AS has_views,

        -- Location signals
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'quiet|peaceful|secluded|cul-de-sac') THEN 1 ELSE 0 END AS is_quiet_location,
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'station|transport|tube|train|underground') THEN 1 ELSE 0 END AS near_transport,
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'school|schools|catchment') THEN 1 ELSE 0 END AS near_schools,

        -- Property type/age signals
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'period|victorian|edwardian|georgian|character|original features') THEN 1 ELSE 0 END AS is_period_property,
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'new build|newly built|brand new|never lived') THEN 1 ELSE 0 END AS is_new_build,

        -- Investment signals
        CASE WHEN REGEXP_CONTAINS(LOWER(description), r'investment|tenant|let|rental income|buy to let') THEN 1 ELSE 0 END AS is_investment

    FROM `{PROJECT_ID}.{DATASET}.listings_raw`
    WHERE SAFE_CAST(latitude AS FLOAT64) IS NOT NULL
      AND SAFE_CAST(longitude AS FLOAT64) IS NOT NULL
      AND SAFE_CAST(price AS FLOAT64) > 0
),

epc_stats AS (
    SELECT 
        REPLACE(UPPER(POSTCODE), ' ', '') AS postcode_clean,
        SAFE_CAST(NUMBER_HEATED_ROOMS AS INT64) AS beds_approx, 
        AVG(SAFE_CAST(TOTAL_FLOOR_AREA AS FLOAT64)) AS avg_floor_area,
        AVG(SAFE_CAST(CURRENT_ENERGY_EFFICIENCY AS FLOAT64)) AS avg_energy_efficiency
    FROM `{PROJECT_ID}.{DATASET}.epc_raw`
    WHERE SAFE_CAST(TOTAL_FLOOR_AREA AS FLOAT64) > 0
    GROUP BY 1, 2
),

epc_fallback AS (
    SELECT 
        REPLACE(UPPER(POSTCODE), ' ', '') AS postcode_clean,
        AVG(SAFE_CAST(TOTAL_FLOOR_AREA AS FLOAT64)) AS avg_floor_area,
        AVG(SAFE_CAST(CURRENT_ENERGY_EFFICIENCY AS FLOAT64)) AS avg_energy_efficiency
    FROM `{PROJECT_ID}.{DATASET}.epc_raw`
    WHERE SAFE_CAST(TOTAL_FLOOR_AREA AS FLOAT64) > 0
    GROUP BY 1
),

sold_homes AS (
    SELECT 
        lr.price AS sold_price,
        ST_GEOGPOINT(SAFE_CAST(g.lon AS FLOAT64), SAFE_CAST(g.lat AS FLOAT64)) AS sold_geom
    FROM `{PROJECT_ID}.{DATASET}.land_registry_raw` lr
    JOIN `{PROJECT_ID}.{DATASET}.geocodes` g
        ON REPLACE(UPPER(lr.postcode), ' ', '') = REPLACE(UPPER(g.postcode), ' ', '')
    WHERE lr.transfer_date >= '2020-01-01'
      AND SAFE_CAST(g.lat AS FLOAT64) IS NOT NULL
),

comps_ranked AS (
    SELECT 
        l.main_image_url,
        s.sold_price,
        ROW_NUMBER() OVER (PARTITION BY l.main_image_url ORDER BY ST_DISTANCE(l.listing_geom, s.sold_geom)) AS rank
    FROM listings l
    JOIN sold_homes s ON ST_DWITHIN(l.listing_geom, s.sold_geom, 2000)
),

comps_agg AS (
    SELECT main_image_url, AVG(sold_price) AS avg_nearby_sold_price, COUNT(*) AS comp_count
    FROM comps_ranked WHERE rank <= 5 GROUP BY main_image_url
)

SELECT 
    -- Target
    l.price_numeric AS price,

    -- Core property features
    l.bedrooms_int AS bedrooms,
    l.property_type,
    l.city,
    l.lat AS latitude,
    l.lon AS longitude,

    -- Text-derived condition signals
    l.is_modern,
    l.needs_work,

    -- Text-derived property features
    l.has_garden,
    l.has_parking,
    l.has_views,

    -- Text-derived location signals
    l.is_quiet_location,
    l.near_transport,
    l.near_schools,

    -- Text-derived property type signals
    l.is_period_property,
    l.is_new_build,
    l.is_investment,

    -- EPC features
    COALESCE(e.avg_floor_area, ef.avg_floor_area, 70) AS floor_area,
    COALESCE(e.avg_energy_efficiency, ef.avg_energy_efficiency, 60) AS energy_efficiency,

    -- Comps
    c.avg_nearby_sold_price,
    COALESCE(c.comp_count, 0) AS comp_count,

    -- Keep description for potential future NLP
    l.description,

    -- Reference
    l.main_image_url,
    l.address,

    -- Image embeddings (all 2048 dimensions)
    {embedding_cols}

FROM listings l
LEFT JOIN epc_stats e ON l.postcode_clean = e.postcode_clean AND l.bedrooms_int = e.beds_approx
LEFT JOIN epc_fallback ef ON l.postcode_clean = ef.postcode_clean
LEFT JOIN comps_agg c ON l.main_image_url = c.main_image_url
"""

print(f"\n🚀 Recreating master_training_data with description + enhanced features...")
print(f"   Query length: {len(query):,} characters")

job = client.query(query)
job.result()

table = client.get_table(f"{PROJECT_ID}.{DATASET}.master_training_data")
print(f"✅ Created table with {table.num_rows:,} rows and {len(table.schema)} columns")

# Show a summary of the new columns
print(f"\n📊 Column summary:")
non_embedding_cols = [f.name for f in table.schema if not f.name.startswith('img_dim_')]
print(f"   Tabular features: {len(non_embedding_cols)}")
print(f"   Image embedding dims: {len([f for f in table.schema if f.name.startswith('img_dim_')])}")
print(f"\n   Tabular columns:")
for col in non_embedding_cols:
    print(f"      - {col}")