import csv

file=open("train.csv", "r")
reader = csv.reader(file)
averages = [0 for x in range(15)]
feature_train, output_train = [],[]

for line in reader:
    for i in range(15):
        plus = (int(line[i]) if(line[i] != '') else 0)
        averages[i] += plus

averages = [x / reader.line_num for x in averages]
print(averages)

lineIdentificator = 1
file=open("train.csv", "r")
reader = csv.reader(file)
for line in reader:
    oneLine = [int(line[i]) if line[i] != '' else int(round(averages[i])) for i in range(15)]
    feature_train += [oneLine]
    lineIdentificator += 1
    output_train += [int(line[15])]

print(feature_train)
print(output_train)