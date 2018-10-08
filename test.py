from scipy.io import wavfile as wav
import matplotlib.pyplot as plt 
import time
import signalProcess.whistleDetector as sp

fs, data = wav.read('./whistle_file/20171113_220000.wav_3.25_3.26.wav')
N = int(round(float(fs/45.875)))
overlap = 0.9*N
window = ('hann')
time_start = time.time()
#plt.figure()
#plt.pcolormesh(t, f, 10.0*np.log(Sxx))
#plt.colorbar()
t, f, aa = sp.whistle_detector(data, fs, window, N, overlap, 15.0, 3000, 10000)
#plt.figure()
#plt.pcolormesh(t, f, aa)
#plt.colorbar()
#plt.ylabel('Hz')
#plt.xlabel('T')
time_end = time.time()
print("time use = ", time_end-time_start)

plt.show()
