######################################## Load Libraries ###########################################

import pandas as pd
from collections import Counter

##################################### Read Data ###################################################
binary = pd.read_excel("D:/University Selection System/Datasets/Fall 2014 Results.xlsx")

print ("TrainData Columns:\n",binary.columns)
 # Explore Data
binary.describe()

 # Columns
binary.dtypes.index

Unis = binary.values[:,1]
# print(Unis)
Unis.sort()
Uni={}
# count=0
# Count=[]
for i in Unis:
    if i not in Uni:
         Uni[i] = 1
    else:
         Uni[i] += 1

print ("\n",dict(Counter(Uni).most_common()))

for i in Uni:
    #print(i)
    x=binary.loc[binary['UniversityApplied'] == i]
    # print("Completed {0}".format(i))
    x.to_excel('{0}.xlsx'.format(i))
    print("File Created")'''

