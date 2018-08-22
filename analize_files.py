from read_data import readmat
import numpy as np
from read_data import readmat
import numpy as np
import os
from keras.models import model_from_yaml
from keras import losses
from scipy.io import savemat
import matplotlib.pyplot as plt

from os import mkdir , path

def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = np.log(np.clip(y_pred, 0.00001, None) + 1.)
    second_log = np.log(np.clip(y_true, 0.00001, None) + 1.)
    return np.mean(np.square(first_log - second_log))

def main():
    best =  ['nnet_data_logpower_noise_chans_train/dt05_ped_real/M03_051C0104_PED.io.mat',
    'nnet_data_logpower_noise_chans_train/dt05_ped_real/M03_050C010N_PED.io.mat',
    'nnet_data_logpower_noise_chans_train/dt05_ped_real/M03_053C0111_PED.io.mat',
    'nnet_data_logpower_noise_chans_train/dt05_ped_real/M03_22HC010R_PED.io.mat',
    'nnet_data_logpower_noise_chans_train/dt05_ped_real/M03_052C010J_PED.io.mat',
    'nnet_data_logpower_noise_chans_train/dt05_ped_real/M03_421C020Q_PED.io.mat',
    'nnet_data_logpower_noise_chans_train/dt05_ped_real/M03_22GC010D_PED.io.mat',
    'nnet_data_logpower_noise_chans_train/dt05_ped_real/M03_420C020P_PED.io.mat',
    'nnet_data_logpower_noise_chans_train/dt05_ped_real/M03_422C020F_PED.io.mat']

    worst = [
    'nnet_data_logpower_noise_chans_train/dt05_bus_real/F01_422C0213_BUS.io.mat',
    'nnet_data_logpower_noise_chans_train/dt05_bus_real/F01_420C020K_BUS.io.mat',
    'nnet_data_logpower_noise_chans_train/dt05_bus_real/F01_051C0104_BUS.io.mat',
    'nnet_data_logpower_noise_chans_train/dt05_str_real/F01_053C0115_STR.io.mat',
    'nnet_data_logpower_noise_chans_train/dt05_str_real/F04_423C0204_STR.io.mat',
    'nnet_data_logpower_noise_chans_train/dt05_bus_real/F01_051C0111_BUS.io.mat',
    'nnet_data_logpower_noise_chans_train/dt05_str_real/F04_423C020I_STR.io.mat',
    'nnet_data_logpower_noise_chans_train/dt05_str_real/F01_421C020A_STR.io.mat',
    'nnet_data_logpower_noise_chans_train/dt05_str_real/F01_22GC0113_STR.io.mat']

    yaml_file_name = 'lstm_5ch_2_layers_real_chans.yaml'
    h5_file_name = 'lstm_5ch_2_layers_real_chans.h5'

    yaml_file = open(yaml_file_name, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights(h5_file_name)

    print("Loaded model from disk")
    loaded_model.compile(loss=losses.mean_squared_logarithmic_error, optimizer='rmsprop', metrics=['mse','mae','logcosh'])


    for filename in worst:
        try:
            real_wiener_all,wiener_all = readmat(filename,keys=('Input','Output'),read='inout')
            print(len(real_wiener_all))
        except:
            print(filename, 'error')
            continue

        X = np.zeros((len(wiener_all),1,257*5))
        for i in range(np.size(X, 0)):
            x = wiener_all[i].reshape((1, 257 * 5))

            X[i][0] = x

        Y = np.zeros((len(wiener_all),1,257))
        for i in range(np.size(Y, 0)):
            y = real_wiener_all[i].reshape((1, 257))

            Y[i][0] = y

        Y1 = loaded_model.predict(X, verbose=1)

        num = -1
        bestloss = 0
        for i in range(len(Y1)):

            loss = mean_squared_logarithmic_error(Y[i][0][0:60],Y1[i][0][0:60])
            if bestloss < loss :
                bestloss = loss
                num = i

            #Sum_loss +=mean_squared_logarithmic_error(Y[i][0][0:60],Y1[i][0][0:60])
            #Sum_loss  +=losses.mean_absolute_error(Y[i][0][0:60],Y1[i][0][0:60]).eval()
            Ys = Y[i][0]
            Ys = np.expand_dims(Ys, 2)
            Y1s = Y1[i][0]
            Y1s = np.expand_dims(Y1s, 2)
            I = X[i][0]

            pre = plt.plot(Y1s[0:257], label='predicted')

            inp = plt.plot(Ys[0:257], label='Output')
            #plt.legend()
            #plt.show()
        print(filename,num,bestloss)
if __name__ == "__main__":
    main()
"""
best of the best
Loaded model from disk
126

nnet_data_logpower_noise_chans_train/dt05_ped_real/M03_051C0104_PED.io.mat 5 8.330936374528283e-07
316

nnet_data_logpower_noise_chans_train/dt05_ped_real/M03_050C010N_PED.io.mat 101 5.904973809289847e-07
519

nnet_data_logpower_noise_chans_train/dt05_ped_real/M03_053C0111_PED.io.mat 455 6.960394673604922e-07
188

nnet_data_logpower_noise_chans_train/dt05_ped_real/M03_22HC010R_PED.io.mat 1 8.253777686378926e-07
647


nnet_data_logpower_noise_chans_train/dt05_ped_real/M03_052C010J_PED.io.mat 643 7.452765724145863e-07
466


nnet_data_logpower_noise_chans_train/dt05_ped_real/M03_421C020Q_PED.io.mat 461 9.309602410013848e-07
167


278

nnet_data_logpower_noise_chans_train/dt05_ped_real/M03_420C020P_PED.io.mat 277 7.432435037992432e-07
206

nnet_data_logpower_noise_chans_train/dt05_ped_real/M03_422C020F_PED.io.mat 205 8.052200649549534e-07


worst of the worst 

229

nnet_data_logpower_noise_chans_train/dt05_bus_real/F01_422C0213_BUS.io.mat 159 0.00021650234460668198
407

nnet_data_logpower_noise_chans_train/dt05_bus_real/F01_420C020K_BUS.io.mat 214 0.0002794284970510724
196

nnet_data_logpower_noise_chans_train/dt05_bus_real/F01_051C0104_BUS.io.mat 151 0.0001328717316867537
278

nnet_data_logpower_noise_chans_train/dt05_str_real/F01_053C0115_STR.io.mat 227 0.0001466209861238384
302

nnet_data_logpower_noise_chans_train/dt05_str_real/F04_423C0204_STR.io.mat 174 0.00016139868736578537
247

nnet_data_logpower_noise_chans_train/dt05_bus_real/F01_051C0111_BUS.io.mat 34 0.00013630864166735987
321

nnet_data_logpower_noise_chans_train/dt05_str_real/F04_423C020I_STR.io.mat 27 0.00015746184975293318
291

nnet_data_logpower_noise_chans_train/dt05_str_real/F01_421C020A_STR.io.mat 105 0.00014849431986658436
505

nnet_data_logpower_noise_chans_train/dt05_str_real/F01_22GC0113_STR.io.mat 190 0.00020820768709907238

   """