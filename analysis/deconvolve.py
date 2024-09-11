import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
from scipy.signal import deconvolve

# -----------------------------------------------------------
# NOW WE DO THE ABOVE AND AVERAGE FOR 64 SIMS
# -----------------------------------------------------------

import sys
sys.path.insert(0, "../")
import src.io as io

# --- Open outputs into array ---

modeldir = "../512512/clutter/"
files = glob(f"{modeldir}outputs/*.rsf")
outputs = [io.rsffile(f"{f}").amps for f in files]

# --- Open targets ---

modeldir = "../../../mars2024/mltrSPSLAKE/T/"
ids = [f[-7:-4] for f in files]
targets = [io.rsffile(f"{modeldir}dsyT2-r{i}.rsf").amps for i in ids]

# --- Deconvolve all ---

rsffile = io.rsffile(files[0])
x = np.arange(rsffile.nx)*rsffile.dx

scalars = []
#pnglim = True
pnglim = False
for j, o in enumerate(outputs):
    
    if pnglim:
        ps = 255 # max val for arrays
        typ = np.uint8
    else:
        ps = 1 
        typ = np.float64

    # scale target to match png
    t = targets[j]
    t = (ps * (t - t.min()) / (t.max() - t.min())).astype(typ)

    # scale output to match png
    o = (ps * (o - o.min()) / (o.max() - o.min())).astype(typ)

    filt = []
    for i in range(len(x)):
        filt_est, remainder = deconvolve(o[i, :], t[i, :])
        filt.append(filt_est)
    scalars.append(np.array(filt).flatten())

# --- Plot result --- 

plt.figure(figsize=(15, 4))
for s in scalars:
    plt.plot(x, s)
plt.plot(x, np.average(scalars, axis=0), color="black")
plt.plot(x, np.median(scalars, axis=0), color="black", linestyle='dashed')
plt.xlabel("x (m)")
plt.ylabel("Scalar Filter")
plt.title("Deconvolution of target/output pair (set of 64)")
plt.show()

# --- Look at Frequency Composition ---

plt.figure(figsize=(15, 4))

# Plot Fourier transform of each scalar array
freqss = []
for s in scalars:
    fft_s = np.fft.fft(s-np.mean(s))
    freqs = np.fft.fftfreq(rsffile.nx, d=rsffile.dx)
    fft_s = np.fft.fftshift(fft_s)
    freqs = np.fft.fftshift(freqs)
    freqss.append(fft_s)
    plt.plot(freqs[1:], fft_s[1:])

# Average and median Fourier transforms
avg_fft = np.average(freqss, axis=0)[1:]
med_fft = np.median(freqss, axis=0)[1:]

# Plot average and median Fourier transforms
plt.plot(freqs[1:], avg_fft, color="black", label="Average FFT")
plt.plot(freqs[1:], med_fft, color="black", linestyle='dashed', label="Median FFT")

plt.xlabel("Frequency (1/meter)")
plt.ylabel("Amplitude")
plt.title("Fourier Transform of Scalar Filter (set of 64)")
plt.legend()
plt.show()

plt.figure(figsize=(15, 4))

# Plot Fourier transform of each scalar array
freqss = []
for s in scalars:
    fft_s = np.fft.fft(s-np.mean(s))
    freqs = np.fft.fftfreq(rsffile.nx, d=rsffile.dx)
    fft_s = np.fft.fftshift(fft_s)
    freqs = np.fft.fftshift(freqs)
    freqss.append(fft_s)
    plt.plot(freqs[1:], fft_s[1:])

# Average and median Fourier transforms
avg_fft = np.average(freqss, axis=0)[1:]
med_fft = np.median(freqss, axis=0)[1:]

# Plot average and median Fourier transforms
plt.plot(freqs[1:], avg_fft, color="black", label="Average FFT")
plt.plot(freqs[1:], med_fft, color="black", linestyle='dashed', label="Median FFT")

plt.xlabel("Frequency (1/meter)")
plt.xlim(-1, 1)
plt.ylabel("Amplitude")
plt.title("Fourier Transform of Scalar Filter (set of 64)")
plt.legend()
plt.show()

# --- Plot LPF result ---

def moving_average(signal, window_size=20, mode='reflect'):
    pad_width = window_size // 2
    padded_signal = np.pad(signal, pad_width, mode=mode)[:-1]
    return np.convolve(padded_signal, np.ones(window_size)/window_size, mode='valid')

# Apply moving average filter to each s in scalars
filtered_scalars = [moving_average(s) for s in scalars]

plt.figure(figsize=(15, 4))
for s in filtered_scalars:
    plt.plot(range(512), s)
plt.plot(range(512), np.average(filtered_scalars, axis=0), color="black")
plt.plot(range(512), np.median(filtered_scalars, axis=0), color="black", linestyle='dashed')
plt.xlabel("Trace #")
plt.ylabel("Scalar Filter (Low-pass filtered)")
plt.title("Deconvolution of target/output pair (set of 64, low-pass filtered)")
plt.show()