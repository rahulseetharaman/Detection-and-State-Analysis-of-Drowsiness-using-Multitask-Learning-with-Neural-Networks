import pandas as pd

df = pd.read_csv("E:\\drowsinessdetection\\multiclassdata.csv",header = None)

labels = []

for i in range(0,136):
    labels.append(str(i))

df.columns = labels


maxObj = df.max()
for label,content in df.items():
    if float(maxObj[label])!=float(0):
        df[label] = df[label].div(float(maxObj[label]))

df.to_csv("normalizedmulticlass.csv")