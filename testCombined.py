import pandas as pd
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
model = load_model("multiclass1-2.h5")

labels = []

for i in range(0,136):
    labels.append(str(i))

X_test = pd.read_csv("E:\\drowsinessdetection\\testX.csv",header = None)
print(X_test.shape[0])

X_test.columns = labels

Y_test = pd.read_csv("E:\\drowsinessdetection\\testY.csv",header = None,dtype="int64")

columns =  ["blinking","nonsleepy","sleepy","yawning","eyes open","eyes closed","head still","head nodding","head looking aside","mouth still","mouth yawn","mouth talking"]
Y_test.columns = [0]
# print(Y_test[0])
x_test = X_test.values.tolist()
print(len(x_test))
predicted = model.predict(X_test)
print(len(predicted))
y_pred = []
# y_pred2 = []
# y_pred3 = []
# y_pred4 = []
# # y_pred5 = []
# # y_pred6 = []
y_test = Y_test.values.tolist()

count = 0
c = 0

for i in x_test:
    print(c)
    predicted = model.predict(np.array([i,]))
    x = 0
    if predicted[0][0]>0.5 or predicted[2][0] > 0.5 or predicted[3][0] > 0.5 or predicted[5][0] > 0.5 or predicted[7][0] > 0.5 or predicted[10][0] > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

    col = []
    for j in range(0,len(predicted)):
        if predicted[j][0]>0.5:
            col.append(columns[j])
    print(col)
    print(y_test[c])

    c+=1


# print(count/Y_test.shape[0]*100)
# print(confusion_matrix(Y_test.values.tolist(), y_pred))
# print(classification_report(Y_test.values.tolist(),y_pred))

