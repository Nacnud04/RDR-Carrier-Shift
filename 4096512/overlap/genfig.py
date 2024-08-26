import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# --- Open source and target image and put into arrays ---
simg = Image.open("source.png")
timg = Image.open("target.png")

s = np.array(simg)
t = np.array(timg)

# -- Crop to desired region ---
s = s[1536:2561, :]
t = t[1536:2561, :]

# -- Create and save figure
fig, ax = plt.subplots(1, 2, figsize=(14,14))

ax[0].set_title("Input", fontsize=24)
ax[0].imshow(s, cmap="gray")
ax[0].axis('off')

ax[1].set_title("Output", fontsize=24)
ax[1].imshow(t, cmap="gray")
ax[1].axis('off')

plt.savefig("Figure.png")
