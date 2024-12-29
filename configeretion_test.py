import csv
import time

config = []
with open('configuration.csv', 'r') as file:
    csv_reader = csv.reader(file)
    config = [[int(val) for val in row] for row in csv_reader]

print(config)

for i in range(len(config)):
    print(config[i])
    if config[i][0] == 0:
        print("YES")
        print(i)
