"""

Code to just view a single example.

"""

import tensorflow as tf
import sys, glob, os
sys.path.insert(0, "../")

import src.io as io
import numpy as np
from rich import print

# operate using gpu 2
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# load dataset
print(f"[bold yellow]Loading dataset...[/bold yellow]")
dataset = io.dataset(dir="../data/curved/valid", pair=(0, 2))
val_dataset = dataset.tf_dataset(32)

# load model
model_path = "models/curved_hl_100.keras"
print(f"[bold yellow]Loading model:[/bold yellow] [bright_cyan]{model_path}[/bright_cyan]")
model = tf.keras.models.load_model(model_path)

# generate predictions
print(f"[green]Generating predictions...[/green]")
preds = model.predict(val_dataset)

# example to choose
i = int(sys.argv[1])

# plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
sources = np.concatenate([x for x, y in val_dataset], axis=0)
targets = np.concatenate([y for x, y in val_dataset], axis=0)

sc = ax[0].imshow(sources[i], cmap="gray", vmin=-1, vmax=1)
ax[0].set_title("Source")
tg = ax[1].imshow(targets[i], cmap="gray", vmin=-1, vmax=1)
ax[1].set_title("Target")

result = preds[i]
re = ax[2].imshow(result, cmap="gray", vmin=-1, vmax=1)
ax[2].set_title("Result")

difference = targets[i] - result
dif = ax[3].imshow(difference, cmap="gray", vmin=-1, vmax=1)
ax[3].set_title("Difference")

suptitle = plt.suptitle(f"Example: {i:03d}")

plt.show()