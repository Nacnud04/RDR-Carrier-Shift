import sys
sys.path.insert(0, "../")
import src.io as io
from src.model import *
import numpy as np
from rich import print

# read in all rsf files
batch_size = 64
img_size = (512, 512, 1)

import keras

# build and summarize model
print("[yellow]Building model...[/yellow]")
model = get_model(img_size)
model.summary()
print("[bold green]Finished building model![/bold green]")

import random

val_samples = 15

# subset data into training and validation
print("[yellow]Loading training dataset...[/yellow]")
dataset = io.dataset(dir='../data/random_planes')
train_dataset = dataset.tf_dataset(batch_size)
print("[yellow]Loading validation dataset...[/yellow]")
dataset = io.dataset(dir='../data/random_planes/valid')
valid_dataset = dataset.tf_dataset(batch_size)
print("[bold green]Finished loading datasets![/bold green]")

# configure model
print("[yellow]Compiling model...[/yellow]")
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mean_squared_error")

callbacks = [
    #keras.callbacks.ModelCheckpoint("carrier_shift.keras", save_best_only=True)
    keras.callbacks.ModelCheckpoint(
        filepath="models/planar_hm_{epoch:03d}.keras",
        save_best_only=False,
        save_freq='epoch'  # Save every epoch and manually control saving frequency
    )
]

print("[bold green]TRAINING![/bold green]")
# train model with validation at end of epochs
epochs = 100
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=valid_dataset,
    callbacks=callbacks,
    verbose=2,
)

import pickle

with open('training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

# generate predictions for all in validation dataset
dataset = io.dataset(dir='../data/random_planes/valid')
val_dataset = dataset.tf_dataset(batch_size)
val_preds = model.predict(val_dataset)

i = 10

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
sources = np.concatenate([x for x, y in val_dataset], axis=0)
targets = np.concatenate([y for x, y in val_dataset], axis=0)

ax[0].imshow(sources[i], cmap="gray")
ax[0].set_title("Source")
ax[1].imshow(targets[i], cmap="gray")
ax[1].set_title("Target")

result = val_preds[i]
print(f"\n\nMIN:{np.min(result)}\nMAX:{np.max(result)}\n\n")
ax[2].imshow(result, cmap="gray")
ax[2].set_title("Result")

difference = targets[i] - result
ax[3].imshow(difference, cmap="gray")
ax[3].set_title("Difference")

plt.show()
