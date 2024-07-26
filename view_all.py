import tensorflow as tf
import src.io as io
import numpy as np
from rich import print
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--index", default=10, type=int, help="Index to start at")
parser.add_argument("-c", "--count", default=1, type=int, help="How many results to show after start index")
args = parser.parse_args()

model_dirs = ("camp_hm", "camp_hl")
models = ("models/camp_hm_100.keras", "models/camp_hl_100.keras")

print(f"[yellow]Loading models...[/yellow]")
models = [tf.keras.models.load_model(f'{d}/{m}') for d, m in zip(model_dirs, models)]
print(f"[bold green]Loaded models![/bold green]")

print(f"[yellow]Loading testing datasets...[/yellow]")
datasets = [io.dataset(dir='data/curved_amp/valid', pair=p) for p in ((0, 1), (0, 2))]
val_datasets = [dataset.tf_dataset(32) for dataset in datasets]
sources = np.concatenate([x for x, y in val_datasets[0]], axis=0)
targets1 = np.concatenate([y for x, y in val_datasets[0]], axis=0)
targets2 = np.concatenate([y for x, y in val_datasets[1]], axis=0)
print(f"[bold green]Loaded datasets![/bold green]")

print(f"[yellow]Generating predictions...[/yellow]")
preds = [m.predict(v) for m, v in zip(models, val_datasets)]
print(f"[bold green]Done with model predictions and initalization![/bold green]")


import matplotlib.pyplot as plt

i = args.index

fig, ax = plt.subplots(3,3, figsize=(12,12))

ax[0, 0].imshow(sources[i], cmap="gray")
ax[0, 0].set_title("HF Source")
ax[0, 0].set_ylabel("Target")

ax[0, 1].imshow(targets1[i], cmap="gray")
ax[0, 1].set_title("MF")

ax[0, 2].imshow(targets2[i], cmap="gray")
ax[0, 2].set_title("LF")

ax[1, 0].axis('off')

ax[1, 1].imshow(preds[0][i], cmap="gray")
ax[1, 1].set_ylabel("Result")

ax[1, 2].imshow(preds[1][i], cmap="gray")

ax[2, 0].axis('off')

ax[2, 1].imshow(targets1[i] - preds[0][i], cmap="gray", vmin=-1, vmax=1)
ax[2, 1].set_ylabel("Difference")

ax[2, 2].imshow(targets2[i] - preds[1][i], cmap="gray", vmin=-1, vmax=1)

for a in ax.flatten():
    a.set_xticks([])
    a.set_yticks([])

plt.suptitle(i)
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()