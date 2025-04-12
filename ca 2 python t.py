import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load dataset
df = pd.read_csv(r"C:\Users\sunny\Downloads\WorldBank.csv")

# === Basic Inspection ===
print("Shape of dataset:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nColumn names:\n", df.columns.tolist())
print("\nSample data:\n", df.sample(5))

# === Data Summary ===
print("\n=== Summary (Numerical) ===")
print(df.describe())

print("\n=== Summary (Categorical) ===")
print(df.describe(include='object'))

print("\n=== Unique Values Per Column ===")
print(df.nunique())

# === Missing Value Analysis ===
print("\n=== Missing Value Analysis (Table) ===")
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Values': missing_values,
    'Percent (%)': missing_percent
})
print(missing_df[missing_df["Missing Values"] > 0].sort_values(by="Percent (%)", ascending=False))

# === Handling Missing Values ===
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)
for col in df.select_dtypes(include='object'):
    df[col] = df[col].fillna(df[col].mode()[0])


# === Bar Plot: Top 10 Countries by GDP ===
top_countries = df.groupby("Country Name")["GDP (USD)"].mean().sort_values(ascending=False).head(10)
top_countries_df = top_countries.reset_index()
top_countries_df.columns = ["Country", "GDP"]

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_countries_df,
    x="GDP",
    y="Country",
    hue="Country",
    palette="viridis",
    legend=False
)
plt.title("Top 10 Countries by Average GDP", fontsize=16, weight='bold')
plt.xlabel("Average GDP (USD)")
plt.ylabel("Country")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# === Pie Chart: GDP Share by Region ===
region_gdp = df.groupby("Region")["GDP (USD)"].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 8))
colors = sns.color_palette("pastel")
plt.pie(region_gdp, labels=region_gdp.index, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
plt.title("GDP Share by Region", fontsize=16, weight='bold')
plt.tight_layout()
plt.show()


# === Pivot the Data ===
gdp_pivot = df.pivot_table(index="Country Name", columns="Year", values="GDP (USD)")

plt.figure(figsize=(14, 10))
sns.set(style="whitegrid")
correlation_matrix = df.select_dtypes(include=np.number).corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

plt.title("Full Correlation Heatmap of Numerical Features", fontsize=16, weight="bold")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# === Stacked Bar Chart: Income Group per Region ===
region_income = df.groupby(["Region", "IncomeGroup"]).size().unstack().fillna(0)

region_income.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Accent')
plt.title("Income Group Distribution by Region", fontsize=16, weight='bold')
plt.xlabel("Region")
plt.ylabel("Count")
plt.legend(title="Income Group")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === Heatmap of GDP Over Years for Top 10 Countries ===

df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

df = pd.read_csv(r"C:\Users\sunny\Downloads\WorldBank.csv")

# Clean Data - Ensure correct types
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

top_10_gdp_countries = df.groupby("Country Name")["GDP (USD)"].mean().sort_values(ascending=False).head(10).index

top_10_poor_countries = df.groupby("Country Name")["GDP (USD)"].mean().sort_values(ascending=True).head(10).index

# Filter data for the richest and poorest countries
rich_countries_data = df[df["Country Name"].isin(top_10_gdp_countries)]
poor_countries_data = df[df["Country Name"].isin(top_10_poor_countries)]

gdp_rich_pivot = rich_countries_data.pivot_table(index="Country Name", columns="Year", values="GDP (USD)")

gdp_poor_pivot = poor_countries_data.pivot_table(index="Country Name", columns="Year", values="GDP (USD)")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))  # 1 row, 2 columns

# Richest Countries Heatmap
sns.heatmap(
    gdp_rich_pivot, 
    cmap="YlGnBu", 
    annot=False, 
    linewidths=0.5, 
    linecolor='gray', 
    ax=axes[0]
)
axes[0].set_title("GDP of Top 10 Richest Countries Over Years", fontsize=16, weight='bold')
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Country")

# Poorest Countries Heatmap
sns.heatmap(
    gdp_poor_pivot, 
    cmap="YlOrRd", 
    annot=False, 
    linewidths=0.5, 
    linecolor='gray', 
    ax=axes[1]
)
axes[1].set_title("GDP of Top 10 Poorest Countries Over Years", fontsize=16, weight='bold')
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Country")
plt.tight_layout()
plt.show()

# === Box Plot: GDP Distribution by Income Group ===
plt.figure(figsize=(10, 6))
sns.barplot(x=top_countries.values, y=top_countries.index, hue=top_countries.index, palette="viridis", legend=False)
plt.yscale("log")  # optional for better scaling
plt.title("GDP Distribution by Income Group", fontsize=16, weight='bold')
plt.xlabel("Income Group")
plt.ylabel("GDP (USD) (Log Scale)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# === Scatter Plot ===
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="GDP per capita (USD)", y="Life expectancy at birth (years)", hue="IncomeGroup")
plt.xscale("log")
plt.title("GDP per Capita vs Life Expectancy")
plt.show()

# === Line Plot: Life Expectancy in India ===
country = "India"
df_country = df[df["Country Name"] == country]

plt.figure(figsize=(12, 6))
sns.set(style="darkgrid", context="talk", palette="Set2")
sns.lineplot(data=df_country, x="Year", y="Life expectancy at birth (years)", marker="o", linewidth=2.5, color="#4c72b0")
plt.title(f" Life Expectancy Over Time in {country} ", fontsize=18, weight='bold')
plt.xlabel("Year", fontsize=14)
plt.ylabel("Life Expectancy (Years)", fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# === Lollipop Chart: Average GDP by Region (REPLACED Bar Chart) ===
avg_gdp = df.groupby("Region")["GDP (USD)"].mean().sort_values()
avg_gdp_df = avg_gdp.reset_index()

plt.figure(figsize=(12, 6))
plt.hlines(y=avg_gdp_df["Region"], xmin=0, xmax=avg_gdp_df["GDP (USD)"], color='red')
plt.plot(avg_gdp_df["GDP (USD)"], avg_gdp_df["Region"], "o", markersize=10, color="#007acc")
plt.title("Average GDP by Region (Lollipop Chart)", fontsize=16, weight='bold')
plt.xlabel("Average GDP (USD)")
plt.ylabel("Region")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# === Z-Test (T-Test) between High and Low Income ===
high = df[df["IncomeGroup"] == "High income"]["GDP per capita (USD)"].dropna()
low = df[df["IncomeGroup"] == "Low income"]["GDP per capita (USD)"].dropna()

if len(high) >= 4 and len(low) >= 4:
    z_stat, p_value = stats.ttest_ind(high, low, equal_var=False)
    print("\n=== Z-Test (High vs Low Income) ===")
    print(f"Z-statistic: {z_stat:.2f}, p-value: {p_value:.4f}")
else:
    print("\n⚠️ Not enough data in one of the groups for a valid t-test.")
