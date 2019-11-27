import pandas as pd
import numpy
# import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layer import Dense, Dropout, LSTM

from pandas import DataFrame, Series, concat, read_csv, datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

from math import sqrt
from matplotlib import pyplot



file_name = 'data_akbilgic.xlsx' 
#df = pd.read_excel(file_name, index_col=0)
df = pd.read_excel(file_name, index_col=None)


#changing the column names
df.rename(columns={"Unnamed: 0": "Date","TL BASED": "TL", "USD BASED": "USD", "imkb_x": "SP", "Unnamed: 4":"DAX", "Unnamed: 5": "FTSE",
 "Unnamed: 6": "NIKKEI",  "Unnamed: 7": "BOVESPA", "Unnamed: 8": "EU", "Unnamed: 9": "EM"}, inplace=True)

df = df.drop(columns="TL")

print(df)


train = df.iloc[1:420] #First 419 rows starting form the index 1,:
train = train.drop(columns="Date")

test = df.iloc[420:539] #First 117 rows starting frm the index 420,:

test = test.drop(columns="Date")
#print(train.iloc[0,1])

print("dim: train: "+str(train.shape))
train_x = train.iloc[0:,1:]
print("dim: train_x: "+ str(train_x.shape))
train_y = train.iloc[0:,:1]
print("dim: train_y: "+ str(train_y.shape)+"\n\n")

print("dim: test: "+str(test.shape))
test_x = test.iloc[0:,1:]
print("dim: test_x: "+ str(test_x.shape))
test_y = test.iloc[0:,:1]
print("dim: test_y: "+ str(test_y.shape))


# model = Sequential()
#
# model.add(LSTM(64, input_shape=(1,7), activation='relu', return_sequences=True))
# model.add(Dropout(0.2))
#
# model.add(LSTM(64, activation='relu'))
# model.add(Dropout(0.2))
#
#
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.2))
#
# model.add(Dense(5, activation='softmax'))
#
# opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
#
# model.compile(loss='mse',optimizer=opt, metrics=['accuracy'])
# model.fit(train_x, train_y, epochs=3, validation_data=(test_x, test_y))

