import numpy as np
from read_data import readmat

"""def create_dataset(X,Y,look_back):
    dataX =[]
    dataY=[]
    for i in range(len(X)-look_back+1):
        x=X[i:i+look_back]
        y=Y[i:i+look_back]
        dataX.append(x)
        dataY.append(y)
    print(len(dataX))
    return np.array(dataX),np.array(dataY)"""

def create_dataset(X,Y,look_back,sample_len):
    dataX = []
    dataY = []
    for i in range(0,len(X),sample_len):
        for j in range (i,i+sample_len-look_back+1):
            x = X[j:j+look_back]
            y = Y[j:j+look_back]
            dataX.append(x)
            dataY.append(y)
    return np.array(dataX), np.array(dataY)


def create_dataset(X,Y,look_back):
    dataX = []
    dataY = []
    for j in range(0, len(X) - look_back + 1):
        x = X[j:j + look_back]
        y = Y[j:j + look_back]
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)



def main():
    real_wiener_all, wiener_all = readmat('nnet_data_logpower_noise/dt05_bus_real/F01_050C010K_BUS.io.mat',
                                          keys=('Input', 'Output'), read='inout')
    X = np.zeros((len(wiener_all), 1, 257 * 5))
    for i in range(np.size(X, 0)):
        x = wiener_all[i].reshape((1, 257 * 5))

        X[i][0] = x

    Y = np.zeros((len(wiener_all), 1, 257))
    for i in range(np.size(Y, 0)):
        y = real_wiener_all[i].reshape((1, 257))

        Y[i][0] = y
    X1,Y1 = create_dataset(X,Y,400)
    print(X1.shape,Y1.shape)
    print(X.shape,Y.shape)
if __name__ == "__main__":
    main()
    """(1, 400, 1, 1285) (1, 400, 1, 257)
(400, 1, 1285) (400, 1, 257)"""