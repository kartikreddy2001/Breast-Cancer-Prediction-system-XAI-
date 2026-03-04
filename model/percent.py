import pandas as pd

# Load CSV file
df = pd.read_csv("tumor.csv")

# Replace 'column_name' with your actual column name
percentage = df['Class'].value_counts(normalize=True) * 100

print("Percentage distribution:")
print(percentage)

# If you specifically want only 2 and 4
print("\nClass 2 Percentage:", percentage.get(2, 0))
print("Class 4 Percentage:", percentage.get(4, 0))
