import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

# --- Open output and target ---

direc = "../4096512/overlap/"

oimg = Image.open(f"{direc}output.png")
timg = Image.open(f"{direc}target.png")

o = np.array(oimg)
t = np.array(timg)

# --- Deconvolve ---

from scipy.signal import deconvolve

filt = []
for i in range(512):
    filt_est, remainder = deconvolve(o[:, i], t[:, i])
    filt.append(filt_est)
filt = np.array(filt)

# the results of the deconvolution are entirely scalars
# this leads me to believe that the difference between the
# input and output signal has nothing to do with phase at
# all and is just the thing screwing up the amplitudes

# --- Show image ---

plt.plot(range(512), filt.flatten())
plt.xlabel("Trace #")
plt.ylabel("Scalar Filter")
plt.title("Deconvolution of target/output pair")
plt.show()


# -----------------------------------------------------------
# NOW WE DO THE ABOVE AND AVERAGE FOR 64 SIMS
# -----------------------------------------------------------

import sys
sys.path.insert(0, "../")
import src.io as io

# --- Open target ---

rsftarget = io.rsffile("../data/4096512/overlap/valid/dat2.rsf")
target = np.squeeze(rsftarget.amps)

# --- Open outputs into array ---

modeldir = "../4096512/overlap/"
outputs = [io.rsffile(f"{modeldir}outputs/output-"+"%02d"%i+".rsf").amps for i in range(64)]

# --- Deconvolve all ---

scalars = []
for j, o in enumerate(outputs):

    # scale target to match png
    t = target[j, :, :]
    t = (255 * (t - t.min()) / (t.max() - t.min())).astype(np.uint8)

    # scale output to match png
    o = (255 * (o - o.min()) / (o.max() - o.min())).astype(np.uint8)

    filt = []
    for i in range(512):
        filt_est, remainder = deconvolve(o[i, :], t[i, :])
        filt.append(filt_est)
    scalars.append(np.array(filt).flatten())

# --- Plot result --- 

plt.figure(figsize=(15, 4))
for s in scalars:
    plt.plot(range(512), s)
plt.plot(range(512), np.average(scalars, axis=0), color="black")
plt.plot(range(512), np.median(scalars, axis=0), color="black", linestyle='dashed')
plt.xlabel("Trace #")
plt.ylabel("Scalar Filter")
plt.title("Deconvolution of target/output pair (set of 64)")
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