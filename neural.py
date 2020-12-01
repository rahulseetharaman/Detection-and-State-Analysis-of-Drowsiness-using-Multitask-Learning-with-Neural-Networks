import csv
from sklearn.model_selection import train_test_split
import numpy
from keras.utils.vis_utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense


X_train = pd.read_csv("E:\\drowsinessdetection\\trainX.csv",header = None)

labels = []

for i in range(0,136):
    labels.append(str(i))

X_test = pd.read_csv("E:\\drowsinessdetection\\testX.csv",header = None)

X_train.columns = labels
X_test.columns = labels

Y_train = pd.read_csv("E:\\drowsinessdetection\\trainY.csv",header = None)
Y_test = pd.read_csv("E:\\drowsinessdetection\\testY.csv",header = None)

Y_test.columns = ["label"]





model = Sequential()
model.add(Dense(256, input_dim=136, activation='relu'))
model.add(Dense(256,activation='relu'))
# # model.add(Dense(64,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(512,activation='relu'))
# model.add(Dense(512,activation='relu'))
# # model.add(Dense(64,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
# model.add(Dense(8,activation='relu'))

model.add(Dense(1, activation='sigmoid'))
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset

# Xout = numpy.array(X)
model.fit(X_train, Y_train, epochs=100,batch_size=512)
# model.save("model2.h5")
# evaluate the keras model
_, accuracy = model.evaluate(X_train, Y_train)
print('Accuracy: %.2f' % (accuracy*100))
# Xtest = numpy.array(Xt)
predicted = model.predict_classes(X_test)
count = 0
j = 0
for i in predicted:
    if int(i) == int(Y_test.loc[j]):
        count = count+1
    j+=1
print(count)
print((count/len(Y_test))*100)
print("=== Confusion Matrix ===")
print(confusion_matrix(Y_test, predicted))
print('\n')
print("=== Classification Report ===")
print(classification_report(Y_test, predicted))
print('\n')
# model_json = model.to_json()
# with open("model_642.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model_1282.h5")
# print("Saved model to disk")

