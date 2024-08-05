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
dataset = io.dataset(dir='../data/curved_amp', pair=(0, 2))
train_dataset = dataset.tf_dataset(batch_size)
yprint("Loading validation dataset...")
dataset = io.dataset(dir='../data/curved_amp/valid', pair=(0, 2))
valid_dataset = dataset.tf_dataset(batch_size)
print("[bold green]Finished loading datasets![/bold green]")

yprint("Generating copy of prior model...")
shutil.copyfile("../curved_hl/models/curved_hl_100.keras", "./camp_hl.keras")

import keras

# load model
yprint(f"Loading and compiling model...")
model = keras.saving.load_model("./camp_hl.keras")
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mean_squared_error")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="models/camp_hl_{epoch:03d}.keras",
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
