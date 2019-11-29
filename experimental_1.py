import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#SCALING
from sklearn.preprocessing import MinMaxScaler
#KERAS
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization




###FUNC###Cleaning the dataframe
def cleaner (dataframe):
        
        dataframe.rename(columns={'Unnamed: 0': 'Date','TL BASED': 'TL', 'USD BASED': 'USD_ISE', 'imkb_x': 'SP', 'Unnamed: 4':'DAX', 'Unnamed: 5': 'FTSE',
         'Unnamed: 6': 'NIKKEI',  'Unnamed: 7': 'BOVESPA', 'Unnamed: 8': 'EU', 'Unnamed: 9': 'EM'}, inplace=True)
         #Dropping the duplicate column for Turkish Lira
         ##Keeping the USD function
        dataframe = dataframe.drop(columns="TL")
        #Dropping the duplicate named row
        dataframe = dataframe.drop(dataframe.index[0], axis=0)
        #replacing '0' with mean of that column
        dataframe.replace(0,dataframe.mean(axis=0),inplace=True)
        
        return dataframe


###FUNC###Splitting the Training and Test sets
def splitter (dataframe):
    
    train = dataframe.iloc[:419]
    test = dataframe.iloc[419:]
    return train, test
    

###FUNC###Splitting the training and test set into x's and y's
def x_y_Splitter(train, test):
    
    train_x = train.iloc[:, 1:]
    train_y = train.iloc[:,0]
    
    test_x = test.iloc[:,1:]
    test_y = test.iloc[:,0]
    
    return train_x, train_y, test_x, test_y
    
###FUNC###Reshaping the inputs
def reshaping(frame,sc):
    
    (x,y) = frame.shape
    frame = frame.values
    frame=sc.fit_transform(frame)
    frame = frame.reshape(x,1,y)
    return frame
    
###FUNC###ndArray to a matrix
def ndArr_toMat(ndarr,sc):
    
    ndarr = ndarr.values
    (x,)=ndarr.shape
    ndarr = ndarr.reshape(-1,1)
    ndarr = sc.fit_transform(ndarr) 
    return ndarr

###FUNC###ndArray to a matrix
def Plotter(orig, pred,sc):
    
    #Inverse Transformation of the Scaled Values
    orig = sc.inverse_transform(orig)
    pred = sc.inverse_transform(pred) 
    
    
    plt.plot(orig, color = 'black', label = 'ISE_USD_original')
    plt.plot(pred, color = 'red', label = 'ISE_USD_predicted')
    plt.title('ISE STOCKS PREDICTION')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    return
    

###FUNC###lstm##NN

def lstm_net(x_train, y_train, x_test, y_test):
    
    (dim_input,x,y) = x_train.shape

    #print(str(dim_input)+" "+str(x)+" "+str(y))
    model = Sequential()
    model.add(LSTM(units = 128, activation='relu', return_sequences = True, input_shape = (x_train.shape[1:])))#org1
    #model.add(BatchNormalization())
    #model.add(Dropout(0.2))

    model.add(LSTM(units=128, activation='relu'))#org2
    #model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    
    model.add(Dense(units=1, activation='relu'))#org3
    #model.add(BatchNormalization())
    #model.add(Dropout(0.2))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mse'])

    model.fit(x_train, y_train, epochs = 100, batch_size = dim_input, validation_data = (x_test, y_test))
    #model.fit(x_train, y_train, epochs = 100, batch_size = 32, validation_data = (x_test, y_test))
    
    y_predict = model.predict(x_test)
    
    
    
    print(model.summary())
    print("Dimensions: Training set x: "+"(Instances, Rows, Columns)"+str(x_train.shape))
    print("Dimensions: Training set y: "+"(Rows, Columns)"+str(y_train.shape))
    print("---------------------------------------------------")
    print("Dimensions: Test set x: "+"(Instances, Rows, Columns)"+str(x_test.shape))
    print("Dimensions: Tese set y: "+"(Rows, Columns)"+str(y_test.shape))
    print("-----------------------------------------------")
    print("Dimensions: Prediction set y: (Rows, Columns)"+str(y_predict.shape))
    return y_predict



##InfFlow.###    
##1##reading the file
dataset = pd.read_excel('data_akbilgic.xlsx', index_col=0)

##2##Cleaning up
dataset = cleaner(dataset)
##dataset.to_excel("output.xlsx")


##3##Splitting up
train, test = splitter(dataset)

#splitting
x_train, y_train, x_test, y_test = x_y_Splitter(train, test)



sc = MinMaxScaler(feature_range=(0,1))
###Reshaping X's and Y's
x_train = reshaping(x_train,sc)
x_test = reshaping(x_test,sc)


y_train = ndArr_toMat(y_train,sc)
y_test = ndArr_toMat(y_test,sc)

# print(y_train.shape)
# print(y_test.shape)

###Training the Neural Net and getting predictions

y_predict = lstm_net(x_train, y_train, x_test, y_test)

# print(y_predict)
# print(y_test)




####Plotting the results
Plotter(y_test, y_predict,sc)


















