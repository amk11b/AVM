import pandas as pd
df = pd.read_csv("/modeling/embeddings.csv")
print(f"Total Rows: {len(df)}")
print(f"Missing Bedrooms: {df['bedrooms'].isnull().sum()}")
print(f"Average Bedrooms: {df['bedrooms'].mean()}")
print(df[['city', 'bedrooms']].groupby('city').count())