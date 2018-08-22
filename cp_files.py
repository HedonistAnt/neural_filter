import os
from shutil import copyfile
import soundfile as sf
import numpy as np
test_file_path = [
                    '/media/maria/1E86F68359E110D6/CHiME3/tools/ASR_6ch_track/enhan/beamformit_test/dt05_bus_real',
                    '/media/maria/1E86F68359E110D6/CHiME3/tools/ASR_6ch_track/enhan/beamformit_test/dt05_caf_real',
                    '/media/maria/1E86F68359E110D6/CHiME3/tools/ASR_6ch_track/enhan/beamformit_test/dt05_ped_real',
                    '/media/maria/1E86F68359E110D6/CHiME3/tools/ASR_6ch_track/enhan/beamformit_test/dt05_str_real',
                    '/media/maria/1E86F68359E110D6/CHiME3/tools/ASR_6ch_track/enhan/beamformit_test/et05_str_real'

                 ]
""" '/media/maria/1E86F68359E110D6/CHiME3/tools/ASR_6ch_track/enhan/beamformit_test/et05_bus_real',
                    '/media/maria/1E86F68359E110D6/CHiME3/tools/ASR_6ch_track/enhan/beamformit_test/et05_ped_real',
                    '/media/maria/1E86F68359E110D6/CHiME3/tools/ASR_6ch_track/enhan/beamformit_test/et05_str_real'"""
untouched_file_path ='/media/maria/1E86F68359E110D6/chans'
compare_file_path = '/media/maria/1E86F68359E110D6/CHiME3/tools/ASR_6ch_track/enhan/beamformit_compare'



for p in test_file_path:
    for root, dirs, files in os.walk(p):
        for filename in files:

            """
            src = os.path.join(untouched_file_path+'/'+p.split('/')[-1]+'/'+filename)
            data, samplerate = sf.read(src)
            data = np.mean(data,1)
            """

            data1,samplerate = sf.read(os.path.join(untouched_file_path + '/' + p.split('/')[-1] + '/' + filename.split('.')[0]+'.CH1.wav'))
            data3,samplerate = sf.read(os.path.join(
                untouched_file_path  + '/' + p.split('/')[-1] + '/' +
                filename.split('.')[0] + '.CH3.wav'))
            data4,samplerate = sf.read(os.path.join(
                untouched_file_path  + '/' + p.split('/')[-1] + '/' +
                filename.split('.')[0] + '.CH4.wav'))
            data5,samplerate = sf.read(os.path.join(
                untouched_file_path  + '/' + p.split('/')[-1] + '/' +
                filename.split('.')[0] + '.CH5.wav'))
            data6,samplerate = sf.read(os.path.join(
                untouched_file_path  + '/' + p.split('/')[-1] + '/' +
                filename.split('.')[0] + '.CH6.wav'))

            data = (data1+data3+data4+data5+data6)/5




            print(data.shape)
            dst = os.path.join(compare_file_path + '/' + p.split('/')[-1] + '/' + filename)
            sf.write(dst, data, samplerate)





#./local/run_gmm_recog_2105.sh beamformit_test /media/maria/1E86F68359E110D6/CHiME3/tools/ASR_6ch_track/enhan/beamformit_compare ../../tools/ASR_models


"""
test
dt05_real WER: 18.03% (Average), 17.65% (BUS), 12.50% (CAFE), 52.94% (PEDESTRIAN), 16.46% (STREET)
compare
dt05_real WER: 19.26% (Average), 23.53% (BUS), 15.00% (CAFE), 23.53% (PEDESTRIAN), 18.99% (STREET)


"""