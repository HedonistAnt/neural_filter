import pickle
import os
Val= pickle.load( open( "validation_set.p", "rb" ) )
print(Val)
counter = {'bus':0,'caf':0, 'str':0, 'ped': 0}
namelist = {'bus':[],'caf':[], 'str':[], 'ped': []}
result_list = []

Path = ['nnet_data_logpower_noise_chans_train_normalized/dt05_caf_real',
        'nnet_data_logpower_noise_chans_train_normalized/dt05_bus_real',
        'nnet_data_logpower_noise_chans_train_normalized/dt05_ped_real',
        'nnet_data_logpower_noise_chans_train_normalized/dt05_str_real']


for p in Path:

    for root, dirs, files in os.walk(p):
        for filename in files:

            if filename.split('.')[1] == 'predicted' :
                os.remove(os.path.join(p,filename))
                print('Deleted: ', filename)

for name in Val:
    noise_type = name.split('/')[1].split('_')[1]
    counter[noise_type]+=1
    namelist[noise_type].append(name)
for noise_type in counter.keys():
    for i in range (min(counter.values())):
        result_list.append(namelist[noise_type][i])

print(len(result_list))


from keras.models import model_from_yaml
from keras import losses
from read_data import  readmat
import numpy as np
from scipy.io import savemat
yaml_file_name = 'lstm_5ch_2_layers_real_chans.yaml'
h5_file_name = 'lstm_5ch_2_layers_real_chans.h5'

yaml_file = open(yaml_file_name, 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights(h5_file_name)

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
    savedict = {'predicted': Y1}

    print('nnet_data_logpower_noise_chans_train_normalized'+'/'+filename.split('/')[1]+"/"+filename.split('/')[2].split('.')[0] + '.predicted.mat')
    savemat('nnet_data_logpower_noise_chans_train_normalized'+'/'+filename.split('/')[1]+"/"+filename.split('/')[2].split('.')[0] + '.predicted.mat',savedict)
