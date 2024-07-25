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
dataset = io.dataset(dir="../data/curved/valid")
val_dataset = dataset.tf_dataset(32)

# load model
model_path = "models/curved_hm_100.keras"
print(f"[bold yellow]Loading model:[/bold yellow] [bright_cyan]{model_path}[/bright_cyan]")
model = tf.keras.models.load_model(model_path)

# generate predictions
print(f"[green]Generating predictions...[/green]")
preds = model.predict(val_dataset)

# plot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
sources = np.concatenate([x for x, y in val_dataset], axis=0)
targets = np.concatenate([y for x, y in val_dataset], axis=0)

sc = ax[0].imshow(sources[0], cmap="gray", vmin=-1, vmax=1)
ax[0].set_title("Source")
tg = ax[1].imshow(targets[0], cmap="gray", vmin=-1, vmax=1)
ax[1].set_title("Target")

result = preds[0]
print(f"\n\nMIN:{np.min(result)}\nMAX:{np.max(result)}\n\n")
re = ax[2].imshow(result, cmap="gray", vmin=-1, vmax=1)
ax[2].set_title("Result")

difference = targets[0] - result
dif = ax[3].imshow(difference, cmap="gray", vmin=-1, vmax=1)
ax[3].set_title("Difference")

suptitle = plt.suptitle(f"Epoch: {0:03d}")

def update(i):

    result = preds[i]
    difference = targets[i] - result
    
    sc.set_data(sources[i])
    tg.set_data(targets[i])
    re.set_data(result)
    dif.set_data(difference)

    suptitle.set_text("")
    
    return sc, tg, re, dif, suptitle

# Create the animation
ani = FuncAnimation(fig, update, frames=len(preds))

ani.save('many_examples.gif', writer='pillow', fps=2)