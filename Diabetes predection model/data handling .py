# DIABETES DATA PREPROCESSING

import pandas as pd
import os

#  Load Dataset
df = pd.read_csv("diabetes dataset.csv")

# Columns where zero is invalid
invalid_zero_columns = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]

# Check zero count before replacement
print("Zero count BEFORE replacement:")
for col in invalid_zero_columns:
    print(f"{col}:", (df[col] == 0).sum())

#  Replace zero with median
for col in invalid_zero_columns:
    
    # Calculate median of column
    median_value = df[col].median()
    
    # Replace 0 with median
    df[col] = df[col].replace(0, median_value)

#  Check zero count after replacement
print("\nZero count AFTER replacement:")
for col in invalid_zero_columns:
    print(f"{col}:", (df[col] == 0).sum())

# Save cleaned dataset
df.to_csv("diabetes_cleaned.csv", index=False)

print("\nData cleaning completed successfully.")