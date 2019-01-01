import pandas as pd
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense

def get_LSTM(dataX,dataY,times,model_path):
    """

    :param dataX: datax generated from Create_dataset
    :param dataY: dataY generated from Create_dataset
    :param times: how many rounds you want to train
    :param model_path: the path you want to store your model
    :return:
    """
    model = Sequential()

    model.add(LSTM(activation='tanh',input_shape = dataX[0].shape, output_dim=5, return_sequences = False))
    model.add(Dense(output_dim = 1))
    model.compile(optimizer='adam', loss='mae',metrics=['mse'])

    model.fit(dataX , dataY, epochs = times , batch_size=1, verbose = 2,shuffle=False)
    model.save(model_path)
    y_pred = model.predict(dataX)
    residual_train = pd.DataFrame(dataY-y_pred)
    return residual_train

if __name__ == '__main__':
    get_LSTM()
#residual_train = building_LSTM(dataX,dataY,times,model_path)