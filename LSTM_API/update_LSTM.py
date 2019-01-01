import pandas as pd
from keras.models import load_model
import sched
import time
import numpy as np
from sklearn.preprocessing import StandardScaler

def timedTask():
    # 初始化 sched 模块的 scheduler 类
    scheduler = sched.scheduler(time.time, time.sleep)
    # 增加调度任务
    scheduler.enter(1209600, 1, update_LSTM)
    # 运行任务
    scheduler.run()

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

def update_LSTM(dataX,dataY,times,model_path):
    """

    ::param dataX: datax generated from Create_dataset
    :param dataY: dataY generated from Create_dataset
    :param times: how many rounds you want to train
    :param model_path: the path you want to store your model
    :return:
    """
    model = load_model(model_path)
    model.fit(dataX , dataY, epochs = times , batch_size=1, verbose = 2,shuffle=False)
    model.save(model_path)
    y_pred = model.predict(dataX)
    residual_train = pd.DataFrame(dataY-y_pred)
    return residual_train


if __name__ == '__main__':
    timedTask()
