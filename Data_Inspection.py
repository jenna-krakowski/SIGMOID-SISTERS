import pandas as pd

# Read the entire CSV file
pricing = pd.read_csv("pricing.csv")

# Show the first 10 rows
print("FIRST 10 ROWS:")
print(pricing.head(10))

# Show column names
print("\nCOLUMN NAMES:")
print(pricing.columns)

# Show number of rows and columns
print("\nSHAPE:")
print(pricing.shape)

# Show data types
print("\nDATA TYPES:")
print(pricing.dtypes)

# Show missing values
print("\nMISSING VALUES:")
print(pricing.isna().sum())

# Show summary statistics
print("\nSUMMARY STATISTICS:")
print(pricing.describe())