from read_data import readmat
import numpy as np
import os
from keras.models import model_from_yaml
from keras import losses
from scipy.io import savemat
import matplotlib.pyplot as plt
from os import mkdir , path
#from create_dataset import create_dataset
import pickle


#with open('validation_set.p', 'rb') as f:
   #  Filenames= pickle.load(f)
Filenames = np.array([])
Path = ['chans_normalized_aligned_mean_ds/dt05_str_real',
        'chans_normalized_aligned_mean_ds/dt05_bus_real',
        'chans_normalized_aligned_mean_ds/dt05_caf_real',
        'chans_normalized_aligned_mean_ds/dt05_ped_real']

"""
'nnet_data_logpower_noise_chans_train_normalized/dt05_bus_real',
        'nnet_data_logpower_noise_chans_train_normalized/dt05_caf_real',
        'nnet_data_logpower_noise_chans_train_normalized/dt05_ped_real', """

for p in Path:

    for root, dirs, files in os.walk(p):
        for filename in files:

            if filename.split('.')[1] == 'phase' or filename.split('.')[1] == 'predicted' :
                continue

            fp = os.path.join(p, filename)


            Filenames = np.append(Filenames,fp)

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


for filename in Filenames:

    wiener_all = readmat(filename,keys=('Input'),read='in')



    X = np.zeros((len(wiener_all),1,257*5))
    for i in range(np.size(X, 0)):
        x = wiener_all[i].reshape((1, 257 * 5))

        X[i][0] = x
    """
    Y = np.zeros((len(wiener_all),1,257))
    for i in range(np.size(Y, 0)):
        y = real_wiener_all[i].reshape((1, 257))

        Y[i][0] = y
    """
   # X, Y = create_dataset(X, Y, 2)

   # X, Y = np.squeeze(X), np.squeeze(Y)

    #score = loaded_model.evaluate(X, Y, verbose=1, batch_size=10000)
    #for i in range(len(score)):
    #  print(loaded_model.metrics_names[i],":",score[i])

    print(filename.split('/')[0] + '/' + filename.split('/')[1] + "/" + filename.split('/')[2].split('.')[
        0] + '.predicted.mat')
    Y1 = loaded_model.predict(X,verbose = 1)
    savedict = {'predicted': Y1}

    #if not path.exists(filename.split('/')[0]+'/'+filename.split('/')[1]+ '_predicted'):
    #    mkdir(filename.split('/')[0]+'/'+filename.split('/')[1])

    print(filename.split('/')[0]+'/'+filename.split('/')[1]+"/"+filename.split('/')[2].split('.')[0] + '.predicted.mat')
    savemat(filename.split('/')[0]+'/'+filename.split('/')[1]+"/"+filename.split('/')[2].split('.')[0] + '.predicted.mat',savedict)


"""
    for i in range(100,110):
        Ys = Y[i][0]
        Ys = np.expand_dims(Ys,2)
        Y1s = Y1[i][0]
        Y1s = np.expand_dims(Y1s,2)
        I = X[i][0]
      
    
    
      
        pre = plt.plot(Y1s[0:257],label = 'predicted')
    
        inp = plt.plot(Ys[0:257],label = 'Output')
        plt.legend()
        plt.show()
        dt05_real WER: 19.67% (Average), 23.53% (BUS), 12.50% (CAFE), 47.06% (PEDESTRIAN), 17.72% (STREET)

        dt05_real WER: 20.08% (Average), 23.53% (BUS), 12.50% (CAFE), 47.06% (PEDESTRIAN), 18.99% (STREET)

        
    """