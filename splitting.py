import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("E:\\drowsinessdetection\\normalizedmulticlass.csv",header = None)

labels = []

for i in range(0,136):
    labels.append(str(i))

df.columns = labels

df_y = pd.read_csv("E:\\drowsinessdetection\\multiclasslabels.csv",header = None)

df_y.columns = ["blinking","nonsleepy","sleepy","yawning","eyes open","eyes closed","head still","head nodding","head looking aside","mouth still","mouth yawn","mouth talking"]


X_train ,X_test, Y_train, Y_test = train_test_split(df,df_y,test_size=0.25)

X_train.to_csv("mtrainX.csv")
X_test.to_csv("mtestX.csv")
Y_train.to_csv("mtrainY.csv")
Y_test.to_csv("mtestY.csv")