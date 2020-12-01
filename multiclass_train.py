import csv
from sklearn.model_selection import train_test_split
import numpy
from keras.utils.vis_utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense

X_train = pd.read_csv("E:\\drowsinessdetection\\mtrainX.csv",header = None)

labels = []

for i in range(0,136):
    labels.append(str(i))

X_test = pd.read_csv("E:\\drowsinessdetection\\mtestX.csv",header = None)

X_train.columns = labels
X_test.columns = labels

Y_train = pd.read_csv("E:\\drowsinessdetection\\mtrainY.csv",header = None)
Y_test = pd.read_csv("E:\\drowsinessdetection\\mtestY.csv",header = None)

Y_train.columns =  ["blinking","nonsleepy","sleepy","yawning","eyes open","eyes closed","head still","head nodding","head looking aside","mouth still","mouth yawn","mouth talking"]
Y_test.columns =  ["blinking","nonsleepy","sleepy","yawning","eyes open","eyes closed","head still","head nodding","head looking aside","mouth still","mouth yawn","mouth talking"]

Y1_tr = Y_train["blinking"]
Y2_tr = Y_train["nonsleepy"]
Y3_tr = Y_train["sleepy"]
Y4_tr = Y_train["yawning"]
Y5_tr = Y_train["eyes open"]
Y6_tr = Y_train["eyes closed"]
Y7_tr = Y_train["head still"]
Y8_tr = Y_train["head nodding"]
Y9_tr = Y_train["head looking aside"]
Y10_tr = Y_train["mouth still"]
Y11_tr = Y_train["mouth yawn"]
Y12_tr = Y_train["mouth talking"]

Y1_te = Y_test["blinking"]
Y2_te = Y_test["nonsleepy"]
Y3_te = Y_test["sleepy"]
Y4_te = Y_test["yawning"]
Y5_te = Y_test["eyes open"]
Y6_te = Y_test["eyes closed"]
Y7_te = Y_test["head still"]
Y8_te = Y_test["head nodding"]
Y9_te = Y_test["head looking aside"]
Y10_te = Y_test["mouth still"]
Y11_te = Y_test["mouth yawn"]
Y12_te = Y_test["mouth talking"]


visible = Input(shape=(136,))

c1 = Dense(256,activation='relu')(visible)
c2 = Dense(256,activation='relu')(c1)
c3 = Dense(512,activation='relu')(c2)
c4 = Dense(512,activation='relu')(c3)
c5 = Dense(256,activation='relu')(c4)
c6 = Dense(128,activation='relu')(c5)
c7 = Dense(64,activation='relu')(c6)
c8 = Dense(32,activation='relu')(c7)
c9 = Dense(16,activation='relu')(c8)

s1 = Dense(1,activation='sigmoid')(c9)
s2 = Dense(1,activation='sigmoid')(c9)
s3 = Dense(1,activation='sigmoid')(c9)
s4 = Dense(1,activation='sigmoid')(c9)
s5 = Dense(1,activation='sigmoid')(c9)
s6 = Dense(1,activation='sigmoid')(c9)
s7 = Dense(1,activation='sigmoid')(c9)
s8 = Dense(1,activation='sigmoid')(c9)
s9 = Dense(1,activation='sigmoid')(c9)
s10 = Dense(1,activation='sigmoid')(c9)
s11 = Dense(1,activation='sigmoid')(c9)
s12 = Dense(1,activation='sigmoid')(c9)

model = Model(inputs=visible,outputs=[s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y=[Y1_tr,Y2_tr,Y3_tr,Y4_tr,Y5_tr,Y6_tr,Y7_tr,Y8_tr,Y9_tr,Y10_tr,Y11_tr,Y12_tr], epochs=100, batch_size=16)
# model.save("multiclass1.h5")

_, accuracy = model.evaluate(X_test, [Y1_te,Y2_te,Y3_te,Y4_te,Y5_te,Y6_te,Y7_te,Y8_te,Y9_te,Y10_te,Y11_te])
print(accuracy)

predicted = model.predict(X_test)
p = []
for x in numpy.nditer(predicted):
    xi = []
    for i in range(0,12):

        if x[i] < float(0.5):
            xi.append(0.0)
        else:
            xi.append(1.0)
    p.append(xi)

count = 0
for i in range(0,Y_test.shape[0]):
    v = 1
    for j in range(0,Y_test[i].shape[0]):
        if Y_test[i].loc[j]!=p[i][j]:
            v = 0
            break
    if v==1:
        count+=1

print(count/Y_test.shape[0]*100)