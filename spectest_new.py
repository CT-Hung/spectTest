from scipy.io import wavfile as wav
from scipy import signal
import matplotlib.pyplot as plt 
import time
import numpy as np
import math
import itertools

fs, data = wav.read('./whistle_file/20171113_220000.wav_3.25_3.26.wav')
N = int(round(float(fs/45.875)))
overlap = 0.9*N
window = ('hann')
time_start = time.time()
f, t, Sxx = signal.spectrogram(data, fs, window, N, overlap)
timesp = time.time()
print("spect = ", timesp - time_start)
#plt.figure()
#plt.pcolormesh(t, f, 10.0*np.log(Sxx))
#plt.colorbar()

def whistle_detector(input_data, fs, N, overlap, SNR_threshold, lowCutOff, highCutOff):
    time_1 = time.time()
    tmp = median_filter(input_data, 3)
    time_2 = time.time()
    print("median = ", time_2-time_1)
    tmp = edge_detector(tmp, SNR_threshold)
    time_3 = time.time()
    print("edge = ", time_3-time_2)
    tmp = moving_square(tmp, fs, N, overlap, lowCutOff, highCutOff)
    time_4 = time.time()
    print("square = ", time_4-time_3)
    return tmp

def median_filter(input_data, median_size):
    medfilter_result = input_data.copy()
    median_len = int(median_size/2)
    medfilter_tmp = np.zeros([input_data.shape[0]-median_len*2, input_data.shape[1]-median_len*2, median_size**2])
    i_start, i_end = median_len, int(input_data.shape[0]-median_len)
    j_start, j_end = median_len, int(input_data.shape[1]-median_len)
    xy_list = range(-median_len, median_len+1)
    k = 0
    for xx, yy in itertools.product(xy_list, xy_list):
        for yy in xy_list:
            medfilter_tmp[i_start:i_end, j_start:j_end, k] = input_data[i_start+xx:i_end+xx-1, j_start+yy:j_end+yy-1]
            k = k+1
            if k >= median_size**2:
                k = 0
    medfilter_result[i_start:i_end, j_start:j_end] = np.median(medfilter_tmp, axis=2)

    return medfilter_result

def edge_detector(input_data, SNR_threshold, jump_number=3):
    detect_result = np.zeros(input_data.shape)
    tmp = input_data[jump_number:input_data.shape[0]-jump_number, :]
    row_len = tmp.shape[0]
    moving_jump_number = input_data[:row_len, :]*0.5+input_data[jump_number*2:, :]*0.5
    #ignore the error of divid by 0
    with np.errstate(all='ignore'):
        SNR = 10.0*np.log(tmp/(moving_jump_number)) 
        SNR_i, SNR_j  = np.where(SNR > SNR_threshold)
    SNR_i = SNR_i+jump_number
    detect_result[SNR_i, SNR_j] = 1

    return detect_result

def moving_square(input_data, fs, N, overlap, lowCutOff, highCutOff):
    moving_result = np.zeros(input_data.shape)
    time_width = 0.01
    percent_threshold = 0.5
    bandwidth_sample = 5
    time_width_sample = 11

    if(bandwidth_sample%2 == 0):
        bandwidth_sample = bandwidth_sample-1
    if(time_width_sample%2 == 0):
        time_width_sample = time_width_sample+1

    real_number_threshold = percent_threshold* bandwidth_sample* time_width_sample

    i_start, i_end = int(N* lowCutOff/fs), int(N* highCutOff/fs-(bandwidth_sample-1)/2)
    j_start, j_end = int((time_width_sample-1)/2), int(input_data.shape[1]-((time_width_sample-1)/2))
    square_area = np.zeros([i_end-i_start, j_end-j_start])
    xx_list = range(-(bandwidth_sample-1)/2, ((bandwidth_sample-1)/2)+1)
    yy_list = range(-(time_width_sample-1)/2, ((time_width_sample-1)/2)+1)

    for xx, yy in itertools.product(xx_list, yy_list):
        square_area = square_area+input_data[i_start+xx:i_end+xx, j_start+yy:j_end+yy]

    #find the index of elements are bigger than threshold
    threshold_i, threshold_j = np.where(square_area >= real_number_threshold)
    threshold_i = threshold_i+i_start
    threshold_j = threshold_j+j_start

    for xx, yy in itertools.product(xx_list, yy_list):
        moving_result[threshold_i+xx, threshold_j+yy] = input_data[threshold_i+xx, threshold_j+yy]

    return moving_result 


aa = whistle_detector(Sxx, fs, N, overlap, 15.0, 3000, 10000)
#plt.figure()
#plt.pcolormesh(t, f, aa)
#plt.colorbar()
#plt.ylabel('Hz')
#plt.xlabel('T')
time_end = time.time()
print("time use = ", time_end-time_start)

plt.show()
