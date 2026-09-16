from google.cloud import bigquery
import os

KEY_PATH = "/Users/AVM/AVM/gcp_key.json"
PROJECT_ID = "avm-ml-485715"
DATASET = "avm_dataset"

if os.path.exists(KEY_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

client = bigquery.Client(project=PROJECT_ID)

# Create enhanced comps using postcode sectors
query = """
CREATE OR REPLACE TABLE `avm-ml-485715.avm_dataset.master_training_data_v2` AS
WITH 
-- Extract postcode sector (e.g., "NG1 1" from "NG1 1AB")
listings_with_sector AS (
    SELECT 
        *,
        REGEXP_EXTRACT(address, r'([A-Za-z]{1,2}\d{1,2}\s?\d)') AS postcode_sector
    FROM `avm-ml-485715.avm_dataset.master_training_data`
),

-- Land registry with postcode sector
land_registry_sectors AS (
    SELECT 
        price AS sold_price,
        transfer_date,
        property_type,
        REGEXP_EXTRACT(postcode, r'([A-Za-z]{1,2}\d{1,2}\s?\d)') AS postcode_sector
    FROM `avm-ml-485715.avm_dataset.land_registry_raw`
    WHERE transfer_date >= '2022-01-01'  -- More recent comps
      AND price > 50000
      AND price < 5000000
),

-- Calculate sector-level stats
sector_stats AS (
    SELECT 
        postcode_sector,
        COUNT(*) AS sector_sales_count,
        AVG(sold_price) AS sector_avg_price,
        APPROX_QUANTILES(sold_price, 2)[OFFSET(1)] AS sector_median_price,
        MIN(sold_price) AS sector_min_price,
        MAX(sold_price) AS sector_max_price,
        STDDEV(sold_price) AS sector_price_stddev
    FROM land_registry_sectors
    WHERE postcode_sector IS NOT NULL
    GROUP BY postcode_sector
    HAVING COUNT(*) >= 3  -- Need at least 3 sales for reliable stats
),

-- City-level fallback stats
city_stats AS (
    SELECT 
        l.city,
        COUNT(*) AS city_sales_count,
        AVG(lr.price) AS city_avg_price,
        APPROX_QUANTILES(lr.price, 2)[OFFSET(1)] AS city_median_price
    FROM `avm-ml-485715.avm_dataset.listings_raw` l
    CROSS JOIN `avm-ml-485715.avm_dataset.land_registry_raw` lr
    WHERE lr.transfer_date >= '2022-01-01'
      AND lr.price > 50000
      AND lr.price < 5000000
    GROUP BY l.city
)

SELECT 
    l.*,

    -- Sector-level comps (much better coverage)
    s.sector_sales_count,
    s.sector_avg_price,
    s.sector_median_price,
    s.sector_min_price,
    s.sector_max_price,
    s.sector_price_stddev,

    -- Derived features
    CASE WHEN s.sector_avg_price IS NOT NULL THEN 1 ELSE 0 END AS has_sector_comps,

    -- Price position within sector
    SAFE_DIVIDE(l.price - s.sector_min_price, s.sector_max_price - s.sector_min_price) AS price_position_in_sector,

    -- City fallback
    c.city_avg_price,
    c.city_median_price,

    -- Best available comp price
    COALESCE(l.avg_nearby_sold_price, s.sector_median_price, c.city_median_price) AS best_comp_price,

    -- Comp confidence level
    CASE 
        WHEN l.avg_nearby_sold_price IS NOT NULL THEN 'spatial'
        WHEN s.sector_median_price IS NOT NULL THEN 'sector'
        WHEN c.city_median_price IS NOT NULL THEN 'city'
        ELSE 'none'
    END AS comp_source

FROM listings_with_sector l
LEFT JOIN sector_stats s ON UPPER(REPLACE(l.postcode_sector, ' ', '')) = UPPER(REPLACE(s.postcode_sector, ' ', ''))
LEFT JOIN city_stats c ON l.city = c.city
"""

print("🔧 Creating enhanced training data with sector-level comps...")
job = client.query(query)
job.result()

# Check the results
check_query = """
SELECT 
    comp_source,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as pct
FROM `avm-ml-485715.avm_dataset.master_training_data_v2`
GROUP BY comp_source
ORDER BY count DESC
"""

print("\n📊 Comp Coverage:")
for row in client.query(check_query):
    print(f"   {row.comp_source:10s}: {row.count:,} ({row.pct}%)")

table = client.get_table(f"{PROJECT_ID}.{DATASET}.master_training_data_v2")
print(f"\n✅ Created table with {table.num_rows:,} rows")