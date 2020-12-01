import pandas as pd

df = pd.read_csv("E:\\drowsinessdetection\\data.csv",header = None)

labels = []

for i in range(0,136):
    labels.append(str(i))

df.columns = labels


maxObj = df.max()

print(maxObj)

maxObj.to_csv("datamax.csv")