import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("realistic_traffic_data.csv")

# Convert datetime
df['Datetime'] = pd.to_datetime(df['Datetime'])

print("Dataset Shape:", df.shape)
print(df.head())

# -------------------------------
# BASIC STATS
# -------------------------------
print("\nSummary Statistics:")
print(df.describe())

print("\nCongestion Distribution:")
print(df['Congestion_Level'].value_counts())

# -------------------------------
# PEAK HOUR ANALYSIS
# -------------------------------
hourly_traffic = df.groupby('Hour')['Vehicle_Count'].mean()

print("\nTraffic by Hour:")
print(hourly_traffic)

# -------------------------------
# DAY-WISE ANALYSIS
# -------------------------------
daily_traffic = df.groupby('Day')['Vehicle_Count'].mean()

# -------------------------------
# CITY ANALYSIS (Top 10)
# -------------------------------
city_traffic = df.groupby('City')['Vehicle_Count'].mean().sort_values(ascending=False).head(10)

print("\nTop 10 Cities by Traffic:")
print(city_traffic)

# -------------------------------
# WEATHER IMPACT
# -------------------------------
weather_impact = df.groupby('Weather')['Vehicle_Count'].mean()

print("\nWeather Impact:")
print(weather_impact)

# -------------------------------
# INCIDENT IMPACT
# -------------------------------
incident_impact = df.groupby('Incident')['Vehicle_Count'].mean()

print("\nIncident Impact:")
print(incident_impact)

# -------------------------------
# SPEED VS TRAFFIC
# -------------------------------
correlation = df[['Vehicle_Count', 'Avg_Speed']].corr()

print("\nCorrelation (Traffic vs Speed):")
print(correlation)

# -------------------------------
# FEATURE ENGINEERING (FOR POWER BI)
# -------------------------------

# Peak Hour Flag
df['Peak_Hour'] = df['Hour'].apply(lambda x: 1 if (7 <= x <= 10 or 17 <= x <= 21) else 0)

# Traffic Category
df['Traffic_Category'] = pd.cut(
    df['Vehicle_Count'],
    bins=[0, 100, 300, 1000],
    labels=['Low', 'Medium', 'High']
)

# Weekend Flag
df['Is_Weekend'] = df['Datetime'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)

# -------------------------------
# SAVE PROCESSED DATA
# -------------------------------
df.to_csv("traffic_analysis_output.csv", index=False)

print("\n✅ Processed dataset saved for Power BI")

# -------------------------------
# VISUALIZATIONS
# -------------------------------

# 1. Traffic by Hour (MAIN GRAPH)
plt.figure()
hourly_traffic.plot(marker='o')
plt.title("Traffic Pattern by Hour")
plt.xlabel("Hour")
plt.ylabel("Vehicle Count")
plt.grid()
plt.show()

# 2. Weather Impact
plt.figure()
weather_impact.plot(kind='bar')
plt.title("Traffic by Weather")
plt.xlabel("Weather")
plt.ylabel("Vehicle Count")
plt.show()

# 3. Incident Impact
plt.figure()
incident_impact.plot(kind='bar')
plt.title("Traffic by Incident")
plt.xlabel("Incident")
plt.ylabel("Vehicle Count")
plt.show()

# 4. Top Cities
plt.figure()
city_traffic.plot(kind='bar')
plt.title("Top 10 Cities by Traffic")
plt.xlabel("City")
plt.ylabel("Vehicle Count")
plt.show()