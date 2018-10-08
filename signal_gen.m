clc; clear;
fs = 96000;
t = 1/fs:1/fs:1;
y = chirp(t,30000,1,40000);
pspectrum(y,fs,'spectrogram','TimeResolution',0.1, ...

audiowrite('30k_40k.wav', y, fs)