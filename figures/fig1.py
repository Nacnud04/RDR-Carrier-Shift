import tensorflow as tf
import sys
sys.path.insert(0, "../")
import src.io as io
import numpy as np

# --- LOAD MODEL ---
model = tf.keras.models.load_model("../512512/overlap/models/hl_overlap_100.keras")

# --- LOAD DATA ---
data = io.dataset(dir="../data/overlap/valid", pair=(0, 2)).tf_dataset(64)
sources = np.squeeze(np.array([x for x, y in data]))

# --- PREDICT ---
output = model.predict(data)

# --- MAKE PLOT --- 
import matplotlib.pyplot as plt
i = 12
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(sources[i, :, :].T, cmap="gray", vmin=-1, vmax=1)
ax[0].set_title("Input")

ax[1].imshow(np.squeeze(output[i]).T, cmap="gray", vmin=-1, vmax=1)
ax[1].set_title("Output")

for a in ax:
    a.set_xticks([])
    a.set_yticks([])

plt.subplots_adjust(wspace=0)

plt.show()
