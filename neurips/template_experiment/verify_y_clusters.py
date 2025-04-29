import pandas as pd

# Load the CSV
df = pd.read_csv('filtered_styles.csv', header=None, names=['filename', 'movement', 'year'])

movement_years = df.groupby('movement')['year'].agg(['min', 'max'])

for movement, row in movement_years.iterrows():
    print(f"{movement}: {int(float(row['min']))}-{int(float(row['max']))}")
