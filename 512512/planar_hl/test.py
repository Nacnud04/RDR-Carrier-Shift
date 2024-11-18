import tensorflow as tf
from rich import print

import sys
sys.path.insert(0, "../../")
import src.io as io

import numpy as np
import glob

import os
# run model on gpu 2
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# find all model filepaths
files = np.sort(glob.glob("models/planar_hl_*.keras"))
batch_size = 32

dataset = io.dataset(dir='../../data/512512/random_planes/valid', pair=(0, 2))
size = (512, 512, 1)

datasets = []
preds = []
for f in np.sort(files):
    print(f"[bold yellow]Loading model:[/bold yellow] [bright_cyan]{f}[/bright_cyan]")
    model = tf.keras.models.load_model(f)

    val_dataset = dataset.tf_dataset(batch_size, size)
    print(f"[green]Generating predictions...[/green]")
    datasets.append(val_dataset)
    preds.append(model.predict(val_dataset))

i = 10

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
sources = np.concatenate([x for x, y in datasets[0]], axis=0)
targets = np.concatenate([y for x, y in datasets[0]], axis=0)

sc = ax[0].imshow(sources[i], cmap="gray", vmin=-1, vmax=1)
ax[0].set_title("Source")
tg = ax[1].imshow(targets[i], cmap="gray", vmin=-1, vmax=1)
ax[1].set_title("Target")

result = preds[0][i]
print(f"\n\nMIN:{np.min(result)}\nMAX:{np.max(result)}\n\n")
re = ax[2].imshow(result, cmap="gray", vmin=-1, vmax=1)
ax[2].set_title("Result")

difference = targets[i] - result
dif = ax[3].imshow(difference, cmap="gray", vmin=-1, vmax=1)
ax[3].set_title("Difference")

suptitle = plt.suptitle(f"Epoch: {0:03d}")

def update(j):

    sources = np.concatenate([x for x, y in datasets[j]], axis=0)
    targets = np.concatenate([y for x, y in datasets[j]], axis=0)
    result = preds[j][i]
    difference = targets[i] - result
    
    sc.set_data(sources[i])
    tg.set_data(targets[i])
    re.set_data(result)
    dif.set_data(difference)

    suptitle.set_text(files[j])
    
    return sc, tg, re, dif, suptitle

# Create the animation
ani = FuncAnimation(fig, update, frames=len(files))

ani.save('train_animation_80.gif', writer='imagemagick', fps=3)

plt.show()
