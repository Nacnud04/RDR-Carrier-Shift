import sys
sys.path.insert(0, "../")
import src.io as io

from rich import print
import shutil
import numpy as np

def yprint(t):
    print(f"[bold yellow]{t}[/bold yellow]")

batch_size = 64

# first load datasets
yprint("Loading training dataset...")
dataset = io.dataset(dir='../data/curved_amp')
train_dataset = dataset.tf_dataset(batch_size)
yprint("Loading validation dataset...")
dataset = io.dataset(dir='../data/curved_amp/valid')
valid_dataset = dataset.tf_dataset(batch_size)
print("[bold green]Finished loading datasets![/bold green]")

# copy model from planar example to start training based on
yprint("Generating copy of planar model...")
shutil.copyfile("../curved_hm/models/curved_hm_100.keras", "./camp_hm.keras")

import keras

# load model
yprint(f"Loading and compiling model...")
model = keras.saving.load_model("./camp_hm.keras")
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mean_squared_error")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="models/camp_hm_{epoch:03d}.keras",
        save_best_only=False,
        save_freq='epoch'
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
    verbose=1,
)

import pickle

with open('training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)
