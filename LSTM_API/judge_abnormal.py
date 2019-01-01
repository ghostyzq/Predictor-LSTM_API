import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_dataset(dataset, look_back):
    """
    dataset:a df
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

def judge_abnormal(data,model_path,look_back,residual_train,a,b):
    """

    :param data: the data you want to judge
    :param model_path: the place you store your model
    :param look_back: how long do you want to use to predict
    :param residual_train:
    :param a: the lower of confidence interval
    :param b: the upper of confidence interval
    :return:
    """
    #load model
    try:
        model = load_model(model_path)
        dataX2,dataY2, origin_data12, origin_data22 = create_dataset(data, look_back)
        y_pred = model.predict(dataX2)
        plt.plot(y_pred)
        plt.plot(dataY2)
        residual = pd.DataFrame(dataY2-y_pred)
        residual.columns=['r']
        plt.plot(residual.r)
        """
        p=gp.ggplot(gp.aes(x='date',y='beef'),data=meat)+gp.geom_line(color='blue')+gp.ggtitle(u'折线图')
        print p
        """
        if residual.r.iloc[len(residual) - 1] < st.scoreatpercentile(residual_train, a) or residual.r.iloc[
            len(residual) - 1] > st.scoreatpercentile(residual_train, b):
            result = 'anomaly'
            print('anomaly')
        else:
            result = 'not anomaly'
            print('not anomaly')
    except ImportError as exc:
        raise ImportError(
            "Couldn't return the result"
        ) from exc
    return result

"""
    a = 2.5
    b = 97.5
    dataset2 = df[39226:42000]
    judge_abnormal(dataset2,model_path,look_back,residual_train,a,b)
"""
if __name__ == '__main__':
    judge_abnormal()