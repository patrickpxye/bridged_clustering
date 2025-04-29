import csv
import os
import sys

# Input CSV file
input_csv = '/Users/ellietanimura/bridged_clustering/neurips/template_experiment/wikiart_metadata.csv'

# List of styles you want to extract (EDIT THIS)
target_styles = ['Early_Renaissance', 'Naive_Art_Primitivism']  # Example list

# Lowercase the target styles for case-insensitive matching
target_styles_lower = [style.lower() for style in target_styles]

# Read the CSV
all_rows = []
all_styles = set()

with open(input_csv, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        all_rows.append(row)
        all_styles.add(row['style'].lower())

# Check if all requested styles exist
missing_styles = [style for style in target_styles_lower if style not in all_styles]
if missing_styles:
    print(f"Error: The following requested styles were not found in the CSV: {missing_styles}")
    sys.exit(1)

# Filter rows
filtered_rows = [row for row in all_rows if row['style'].lower() in target_styles_lower]

# Auto-generate output filename
safe_styles = [style.lower().replace(' ', '_') for style in target_styles]
output_csv = f"/Users/ellietanimura/bridged_clustering/neurips/template_experiment/filtered_styles_{'_'.join(safe_styles)}.csv"

# Write filtered rows
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['filename', 'style', 'year']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in filtered_rows:
        writer.writerow(row)

print(f"Filtered {len(filtered_rows)} rows into {output_csv}.")
