import os
from read_data import readmat
import numpy as np
import random
import pickle
from numpy.random import seed
from create_dataset import  create_dataset
eps = 10 ** (-16)

def min_len():

    Path = ['nnet_data_logpower_noise_chans_train_normalized/dt05_caf_real','nnet_data_logpower_noise_chans_train_normalized/dt05_bus_real',
            'nnet_data_logpower_noise_chans_train_normalized/dt05_ped_real', 'nnet_data_logpower_noise_chans_train_normalized/dt05_str_real']
    # ,'nnet_data_logpower_1/dt05_caf_real','nnet_data_logpower_1/dt05_ped_real','nnet_data_logpower_1/dt05_str_real'
    #,'nnet_data_logpower_noise/dt05_str_real','nnet_data_logpower_noise/dt05_ped_real'
    # 'chans_noadapt_ds_real/dt05_bus_real','chans_noadapt_ds_real/dt05_caf_real','chans_noadapt_ds_real/dt05_ped_real','chans_noadapt_ds_real/dt05_str_real'

    print("min_len")

    Filenames = np.array([])

    for p in Path:

        for root, dirs, files in os.walk(p):
            for filename in files:

                if filename.split('.')[1] == 'phase' or filename.split('.')[1] == 'predicted':
                    continue

                fp = os.path.join(p, filename)


                Filenames = np.append(Filenames,fp)


    min_len = 100000000
    for fp in Filenames:
        real_wiener_all, wiener_all = readmat(fp, read='inout', keys=('Input', 'Output'))
        if len(wiener_all)<min_len:
            min_len = len(wiener_all)
    return min_len




def make_ds(Filenames,context,zero_delimiter = False):
    X,Y = 0,0

    if zero_delimiter:
        xdel = np.zeros((1, context, 257))
        ydel = np.zeros((1, context, 257 * 5))
    for fp in Filenames:

        real_wiener_all, wiener_all = readmat(fp, read='inout', keys=('Input', 'Output'))

        dslength = len(wiener_all)

        X_ = np.zeros((dslength, 1, 257 * 5))
        Y_ = np.zeros((dslength, 1, 257))


        for i in range(np.size(X_, 0)):
            x = wiener_all[i].reshape((1, 257 * 5))
            # x = np.array([max(x[0][i],-35) for i in range (1285)])
            X_[i][0] = x


        for i in range(np.size(Y_, 0)):
            y = real_wiener_all[i].reshape((1, 257))
            # y = np.array([max(y[0][i], -35) for i in range(257)])
            Y_[i][0] = y

        if context:
            X_,Y_ = create_dataset(X_,Y_,context)

            X_ = np.squeeze(X_)
            Y_ = np.squeeze(Y_)

            if zero_delimiter:
               newX_ = np.zeros((len(X_)+1,context,257*5))
               newY_ = np.zeros((len(Y_)+1,context,257))

               newX_[1:len(X_)+1,:,:] = X_
               newY_[1:len(Y_)+1,:,:] = Y_
               X_=newX_
               Y_ =newY_






        if type(X) != type(X_):
            X = X_
            Y = Y_
        else:
            X = np.concatenate((X, X_))
            Y = np.concatenate((Y, Y_))
            print(X.shape)



    return X,Y



def main():


    Path = ['nnet_data_logpower_noise_chans_train_normalized/dt05_caf_real','nnet_data_logpower_noise_chans_train_normalized/dt05_bus_real',
            'nnet_data_logpower_noise_chans_train_normalized/dt05_ped_real', 'nnet_data_logpower_noise_chans_train_normalized/dt05_str_real']
    # ,'nnet_data_logpower_1/dt05_caf_real','nnet_data_logpower_1/dt05_ped_real','nnet_data_logpower_1/dt05_str_real'
    #,'nnet_data_logpower_noise/dt05_str_real','nnet_data_logpower_noise/dt05_ped_real'
    # 'chans_noadapt_ds_real/dt05_bus_real','chans_noadapt_ds_real/dt05_caf_real','chans_noadapt_ds_real/dt05_ped_real','chans_noadapt_ds_real/dt05_str_real'


    Filenames = np.array([])

    for p in Path:

        for root, dirs, files in os.walk(p):
            for filename in files:

                if filename.split('.')[1] == 'phase' or filename.split('.')[1] == 'predicted':
                    continue

                fp = os.path.join(p, filename)
                print(type(fp),fp)

                Filenames = np.append(Filenames,fp)


    random.seed(1)
    newFilenames = Filenames[:]
    random.shuffle(newFilenames)

    pickle.dump(newFilenames[int(0.6 * len(newFilenames)):len(newFilenames)], open("validation_set.p", "wb"))
    pickle.dump(newFilenames[0:int(0.6*len(newFilenames))],open("train_set.p","wb"))


    X,Y = make_ds(newFilenames[0:int(0.6*len(newFilenames))],context=87,zero_delimiter=True)
    VX,VY = make_ds(newFilenames[int(0.6*len(newFilenames)):len(newFilenames)],context=87,zero_delimiter=True)





    from keras import losses
    from keras.optimizers import RMSprop
    from keras.models import Sequential
    from keras.layers import LSTM, Dense,Flatten,Lambda,BatchNormalization,LeakyReLU


    model = Sequential()
    eps = 0.000000000000000001

    print(X.shape,Y.shape)
    model.add(LSTM(1285, return_sequences=True, stateful=False, input_shape=(87, 1285)))
    #model.add(LSTM(257))
    #model.add(Dense(1285))

    model.add(Dense(257))
    model.add(LeakyReLU(alpha=-eps))



    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.00)
    model.compile(loss=losses.mean_squared_logarithmic_error, optimizer=rmsprop, metrics=['logcosh', 'mae'])
    history = model.fit(x=X, y=Y, epochs=500, shuffle=False, batch_size=10000, verbose=2,validation_data=(VX,VY))

    print(history.history.keys())

    model_yaml = model.to_yaml()
    with open("lstm_5ch_2_layers_real_chans_87.yaml", "w") as yaml_file:
       yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights("lstm_5ch_2_layers_real_chans_87.h5")
    print("Saved model to disk")
    score = model.evaluate(VX, VY, verbose=1,batch_size = 10000)
    for i in range(len(score)):
       print(model.metrics_names[i],":",score[i])



if __name__ == "__main__":
    main()


    compare
    dt05_real WER: 12.99% (Average), 16.11% (BUS), 12.77% (CAFE), 9.96% (PEDESTRIAN), 13.14% (STREET)
    test
    dt05_real
    WER: 13.66 % (Average), 16.73 % (BUS), 12.70 % (CAFE), 9.10 % (PEDESTRIAN), 16.28 % (STREET)
