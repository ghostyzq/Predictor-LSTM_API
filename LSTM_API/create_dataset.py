import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_dataset(dataset, look_back):
    """
    dataset:a df with a column of datetime and a column called total
    look_back: how many minutes you want to predict
    """
    dataX = dataset[1:len(dataset)-look_back][['total']]
    dataX.index=range(len(dataset)-look_back-1)
    dataY = dataset[look_back+1:len(dataset)][['total']]
    temp = pd.DataFrame()
    for j in range(look_back-1):
        #for i in range(len(dataset)-look_back-1):
        temp = dataset[j+1:len(dataset)-look_back+j][['total']]
        temp.index = range(len(dataset)-look_back-1)
        dataX[str(j)] = temp
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    ss = StandardScaler()
    train_X = ss.fit_transform(dataX)
    origin_data = ss.inverse_transform(train_X)
    train_X = train_X.reshape(len(train_X),look_back,1)
    train_Y = ss.fit_transform(dataY)
    origin_data2 = ss.inverse_transform(train_Y)
    return train_X, train_Y, origin_data, origin_data2

"""
look_back = 10
model_path = '../models'
dataset = pd.read_csv('../training_data/caty22primary.db.ebay.com.csv')
dataX,dataY, origin_data, origin_data2 = create_dataset(dataset,look_back)
"""


if __name__ == '__main__':
    create_dataset()