from scipy.io import loadmat
import numpy as np
import h5py
from scipy.io import savemat
import matplotlib.pyplot as plt
from keras.models import model_from_yaml
def readmat(file_name,datalen=np.inf,read ='inout', **keys):

    inp = keys['keys']
    print(keys['keys'])
    data = h5py.File(file_name,'r')

    if read == "in" or  read == "inout":
        test = data[inp]
    else :
        test = data[out]


    if datalen == np.inf:
        datalen = len(test)
    wiener_all = []
    if read == 'in' or read == 'inout':
        test = data[inp]

        for i in range(datalen):
             st = test[i][0]
             obj = data[st]
             wiener_all.append(obj.value)

    real_wiener_all = []
    if read == 'out' or read == 'inout':
        test = data[out]
        
        for i in range(datalen):
            st = test[i][0]
            obj = data[st]
            real_wiener_all.append(obj.value)

    return   np.array(wiener_all)

if __name__ == "__main__":
 _,INP1 =  readmat('chans_noadapt_ds_real_aligned/dt05_bus_real1',read='in',keys = ('Input','Output'))

 print(INP1.shape)

 yaml_file = open('lstm_5ch_3_layers_real2.yaml', 'r')
 loaded_model_yaml = yaml_file.read()
 yaml_file.close()
 loaded_model = model_from_yaml(loaded_model_yaml)
 loaded_model.load_weights("lstm_5ch_3_layers_real2.h5")

 X1 = np.zeros((100, 1, 257 * 5))
 for i in range(np.size(X1, 0)):
     X1[i] = INP1[i].reshape((1, 257 * 5))



 Y1 = loaded_model.predict(X1, verbose=1)




 print(Y1.shape)
 Y1 = Y1[:,0,:]


 print(Y1.shape)

 savedict = {'predicted': Y1}

 savemat('new_aligned.mat', savedict)



 Ys1 = Y1[0]





 plt.plot(Ys1)



 plt.show()









