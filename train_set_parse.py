import pickle
import numpy as np
from matplotlib import pyplot as plt

import os

#Tra= pickle.load( open( "train_set.p", "rb" ) )

Path =[
             'nnet_data_logpower_noise_chans_train_normalized/dt05_str_real']

Filenames = []
for p in Path:

    for root, dirs, files in os.walk(p):
        for filename in files:

            if filename.split('.')[1] == 'phase' or filename.split('.')[1] == 'predicted':
                continue

            fp = os.path.join(p, filename)
            print(type(fp), fp)

            Filenames = np.append(Filenames, fp)


Tra = Filenames


counter = {'bus':0,'caf':0, 'str':0, 'ped': 0}
namelist = {'bus':[],'caf':[], 'str':[], 'ped': []}
result_list = []
for name in Tra:
    noise_type = name.split('/')[1].split('_')[1]
    counter[noise_type]+=1
    namelist[noise_type].append(name)
for noise_type in counter.keys():
    for i in range(counter[noise_type]):
       result_list.append(namelist[noise_type][i])

print("Total:", sum(counter.values()))
print(counter)

from keras.optimizers import RMSprop
from keras.models import model_from_yaml
from keras import losses
from read_data import  readmat

from scipy.io import savemat
yaml_file_name = 'lstm_5ch_2_layers_real_chans.yaml'
h5_file_name = 'lstm_5ch_2_layers_real_chans.h5'

yaml_file = open(yaml_file_name, 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights(h5_file_name)
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
losses_stat = dict()
loaded_model.compile (loss = losses.mean_squared_logarithmic_error,optimizer = rmsprop)



"""
for filename in result_list:
    try:
        real_wiener_all,wiener_all = readmat(filename,keys=('Input','Output'),read='inout')

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

    Y1 = loaded_model.predict(X,verbose = 1)

    losses_stat[filename] = np.mean(losses.mean_squared_logarithmic_error(Y,Y1).eval())
"""


noise_type_losses ={'bus':[],'caf':[], 'str':[], 'ped': []}
for noise_type in namelist.keys():
    for filename in namelist[noise_type]:
        try:
            real_wiener_all, wiener_all = readmat(filename, keys=('Input', 'Output'), read='inout')

        except:
            print(filename, 'error')
            continue

        X = np.zeros((len(wiener_all), 1, 257 * 5))
        for i in range(np.size(X, 0)):
            x = wiener_all[i].reshape((1, 257 * 5))

            X[i][0] = x

        Y = np.zeros((len(wiener_all), 1, 257))
        for i in range(np.size(Y, 0)):
            y = real_wiener_all[i].reshape((1, 257))

            Y[i][0] = y

        Y1 = loaded_model.predict(X, verbose=1)
        losses_stat[filename] = np.mean(losses.mean_squared_logarithmic_error(Y, Y1).eval())
        noise_type_losses[noise_type].append(losses_stat[filename])


sorted_by_value = sorted(losses_stat.items(), key=lambda kv: kv[1])
sorted_by_value_reverse = sorted(losses_stat.items(), key=lambda kv: kv[1],reverse=True)
print("10 best")
for i in range(10):
    print(sorted_by_value[i])

print("10 worst")
for i in range(10):
    print(sorted_by_value_reverse[i])

plt.figure(1)
plt.subplot(211)
plt.title('BUS')
plt.hist(noise_type_losses['bus'])
plt.subplot(212)
plt.title('CAF')
plt.hist(noise_type_losses['caf'])
plt.subplot(221)
plt.title('STR')
plt.hist(noise_type_losses['str'])
plt.subplot(222)
plt.title('PED')
plt.hist(noise_type_losses['ped'])
plt.show()





