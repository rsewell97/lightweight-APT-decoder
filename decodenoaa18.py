from pylab import*
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz,square
from scipy.misc import imresize
import numpy as np
kernel = [-1,-1,-1,-1,-1,1,1,1,1,1]

File = "C:/Users/Robbie Sewell/Documents/Satellite/testfiles/noaa-18-1.wav"
approxxbot = 900

sampFreq, raw = wavfile.read(File)

def meanvalue(wav):
    ave = 0
    for i in wav:
        ave+=i
    return ave/len(wav)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def scan(line,thresh=11000):
    for i in range(5,len(line)-5):
        x= np.dot(line[i-5:i+5],kernel)
        
        if x > thresh:
            return i
    return -1

cutoff = 1000 # desired cutoff frequency of the filter, Hz
width = 5513
timelimit = 100

data = np.asarray(raw)
t = np.arange(0,step=1/sampFreq,stop=len(data)/sampFreq)

print(sampFreq)
data = data - np.average(data)
data[data < 0] = 0
data = butter_lowpass_filter(data, cutoff, sampFreq)


data = data[approxxbot:]
dataimg = data[:int(width*floor(len(data)/width))]
dataimg = dataimg.reshape(-1,width)

start = []
for line in range(len(dataimg)-1,0,-1):
    p = scan(dataimg[line])
    start.append(p)

adjust = np.zeros(len(dataimg),dtype=int)
start = start[::-1]

i = 0
for pix in start: 
    i+=1
    if pix < start[i-1]+5 and pix > start[i-1]-5:
        adjust[i-1] = pix
    else: 
        adjust[i-1] = start[i-1]

for line in range(len(dataimg)):
    first = dataimg[line,:adjust[line]]
    second = dataimg[line,adjust[line]:]
    dataimg[line] = np.append(second,first)

dataimg = np.flip(dataimg,0)
dataimg = np.flip(dataimg,1)

dataimg = imresize(dataimg,[3200,4000])


plt.imshow(dataimg,cmap='gray')
plt.show()

plt.savefig('decoded_noaa18.png')
