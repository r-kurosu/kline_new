import pandas as pd
import sys

assignment_file = "/Users/takedakiyoshi/lab/kline/KLINE/4月のミーティング向けの資料/exp_assignment.xlsx"
leftrt_file = "/Users/takedakiyoshi/lab/kline/KLINE/4月のミーティング向けの資料/exp_leftRT.xlsx"

assignment = pd.read_excel(assignment_file)
Ddict = {}

for row in assignment.itertuples():
    if row[1] not in Ddict:
        Ddict[row[1]] = {}
        Ddict[row[1]][row[7]] = float(row[8])
    else:
        if row[6] in Ddict[row[1]]:
            Ddict[row[1]][row[7]] += float(row[8])
        else:
            Ddict[row[1]][row[7]] = float(row[8])

leftRT = pd.read_excel(leftrt_file)
for row in leftRT.itertuples():
    if row[1] in Ddict:
        Ddict[row[1]]["left"] = row[2]
    else:
        Ddict[row[1]] = {}
        Ddict[row[1]]["left"] = row[2]

PercentDict = {}
for k, v in Ddict.items():
    total = sum(v.values())
    PercentDict[k] = {}
    for k2, v2 in v.items():
        PercentDict[k][k2] = v2/total * 100
    tmpDict = v
sortedLDict = sorted(PercentDict.items(), key=lambda x: x[0])
# print(sortedLDict)
for item in sortedLDict:
    print(item)
