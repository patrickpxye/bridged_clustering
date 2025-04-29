import os
import csv
import re

# Path to your main directory
root_dir = '/Users/ellietanimura/bridged_clustering/neurips/template_experiment/wikiart'
output_csv = '/Users/ellietanimura/bridged_clustering/neurips/template_experiment/wikiart_metadata.csv'

# Define a regex pattern to find a 4-digit number before the file extension
year_pattern = re.compile(r'[-_](\d{4})(?:\.jpg|\.jpeg)$', re.IGNORECASE)

# Store the rows
rows = []

# Walk through subdirectories
for style in os.listdir(root_dir):
    style_path = os.path.join(root_dir, style)
    if not os.path.isdir(style_path):
        continue  # skip files, only go into directories

    for filename in os.listdir(style_path):
        if not (filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg')):
            continue  # skip non-jpg files

        match = year_pattern.search(filename)
        if match:
            year = int(match.group(1))
            if 1000 <= year <= 2025:
                rows.append({
                    'filename': filename,
                    'style': style,
                    'year': year
                })

# Write to CSV
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['filename', 'style', 'year']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"CSV file saved to {output_csv} with {len(rows)} entries.")
