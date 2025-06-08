import pandas as pd
import os

# Step 1: Load dataset
DATA_PATH = os.path.join("..", "data", "air_quality.csv")
df = pd.read_csv(DATA_PATH)

# Step 2: Preview data
print("📊 Dataset Shape:", df.shape)
print("🧾 Columns:", df.columns.tolist())
print("🕵️ Missing values:\n", df.isnull().sum())

# Step 3: Basic cleaning — drop rows with missing critical values
df = df.dropna(subset=['City', 'Date', 'PM2.5', 'PM10', 'NO2', 'SO2', 'O3'])

# Step 4: Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Step 5: Feature Engineering - extract year, month
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Step 6: AQI Calculation (simple average of pollutants — for demo)
df['AQI'] = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'O3']].mean(axis=1)

# Step 7: Save cleaned dataset
OUTPUT_PATH = os.path.join("..", "data", "air_quality_cleaned.csv")
df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Cleaned data saved to {OUTPUT_PATH}")
