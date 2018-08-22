from read_data import readmat
import numpy as np
import os
from keras.models import model_from_yaml
from keras import losses
from scipy.io import savemat
import matplotlib.pyplot as plt
from os import mkdir , path
import pickle

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
print(loaded_model.trainable_weights)

for layer in loaded_model.layers:
        if "LSTM" in str(layer):
            weightLSTM = layer.get_weights()
warr,uarr, barr = weightLSTM
print(warr)
print(uarr)
#,barr)

