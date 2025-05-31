import csv
filename = 'town5_long_r0d2.csv'
with open(filename, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        print(f"Row {i}: {row}")