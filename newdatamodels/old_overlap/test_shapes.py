import tensorflow as tf
import sys
sys.path.insert(0, "../../")
import src.io as io
import numpy as np
from rich import print
import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--index", default=10, type=int, help="Index to start at")
args = parser.parse_args()

model_dirs = ("../../512512/overlap/", "../../512512/overlap/")
models = ("models/hm_overlap_200.keras", "models/hl_overlap_200.keras")

print(f"[yellow]Loading models...[/yellow]")
models = [tf.keras.models.load_model(f'{d}/{m}') for d, m in zip(model_dirs, models)]
print(f"[bold green]Loaded models![/bold green]")

print(f"[yellow]Loading testing datasets...[/yellow]")

n = 30

# create first dataset with pair 0, 1
cs1 = io.ClutterSim(dir="/home/byrne/WORK/research/mars2024/mltrSPSLAKE/M", maxfiles=n)
cs1.set_max_dims(512, 512)
cs1.load(pair=(0, 1))

# now do same for pair 0, 2
cs2 = io.ClutterSim(dir="/home/byrne/WORK/research/mars2024/mltrSPSLAKE/M", maxfiles=n)
cs2.set_max_dims(512, 512)
cs2.load(pair=(0, 2))

# move into array
val_datasets = [cs.tf_dataset(64, testing=1)[1] for cs in (cs1, cs2)]

sources = np.concatenate([x for x, y in val_datasets[0]], axis=0)
targets1 = np.concatenate([y for x, y in val_datasets[0]], axis=0)
targets2 = np.concatenate([y for x, y in val_datasets[1]], axis=0)

print(f"[bold green]Loaded datasets![/bold green]")

print(f"[yellow]Generating predictions...[/yellow]")
preds = [m.predict(v) for m, v in zip(models, val_datasets)]
print(f"[bold green]Done with model predictions and initalization![/bold green]")


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

i = 0

fig, ax = plt.subplots(3,3, figsize=(12,12))

sc = ax[0, 0].imshow(sources[i], cmap="gray", vmin=-1, vmax=1)
ax[0, 0].set_title("HF Source")
ax[0, 0].set_ylabel("Target")

tg_mf = ax[0, 1].imshow(targets1[i], cmap="gray", vmin=-1, vmax=1)
ax[0, 1].set_title("MF")

tg_lf = ax[0, 2].imshow(targets2[i], cmap="gray", vmin=-1, vmax=1)
ax[0, 2].set_title("LF")

ax[1, 0].axis('off')

pd_mf = ax[1, 1].imshow(preds[0][i], cmap="gray", vmin=-1, vmax=1)
ax[1, 1].set_ylabel("Result")

pd_lf = ax[1, 2].imshow(preds[1][i], cmap="gray", vmin=-1, vmax=1)

ax[2, 0].axis('off')

df_mf = ax[2, 1].imshow(targets1[i] - preds[0][i], cmap="gray", vmin=-1, vmax=1)
ax[2, 1].set_ylabel("Difference")

df_lf = ax[2, 2].imshow(targets2[i] - preds[1][i], cmap="gray", vmin=-1, vmax=1)

for a in ax.flatten():
    a.set_xticks([])
    a.set_yticks([])

suptitle = plt.suptitle(i)
plt.subplots_adjust(wspace=0, hspace=0)

def update(i):
    sc.set_data(np.squeeze(sources[i]).T)
    tg_mf.set_data(np.squeeze(targets1[i]).T)
    tg_lf.set_data(np.squeeze(targets2[i]).T)
    pd_mf.set_data(np.squeeze(preds[0][i]).T)
    pd_lf.set_data(np.squeeze(preds[1][i]).T)
    df_mf.set_data(np.squeeze(targets1[i]-preds[0][i]).T)
    df_lf.set_data(np.squeeze(targets2[i]-preds[1][i]).T)
    suptitle.set_text(f"Example: {i:03d}")
    return sc, tg_mf, tg_lf, pd_mf, pd_lf, df_mf, df_lf, suptitle

ani = FuncAnimation(fig, update, frames=len(targets1))
ani.save('overlaps.gif', writer='pillow', fps=0.2)
