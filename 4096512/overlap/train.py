import sys
import pickle
sys.path.insert(0, "../../")
import src.io as io
from src.model import *

from rich import print
import shutil

batch_size = 8
img_size = (512, 4096, 1)

# --- LOAD HL DATASETS ---

print("[yellow]Loading datasets...[/yellow]")

direc = "../../data/4096512/overlap"
dataset = io.dataset(dir=direc, pair=(0, 2))
hl_train_dataset = dataset.tf_dataset(batch_size, img_size)
dataset = io.dataset(dir=f'{direc}/valid', pair=(0, 2))
hl_valid_dataset = dataset.tf_dataset(batch_size, img_size)

print("[bold green]Finished loading datasets![/bold green]")


# --- BUILD MODEL ---
import keras

print("[yellow]Building model...[/yellow]")

model = get_model(img_size)
model.summary()

model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mean_squared_error")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="models/hl_overlap_{epoch:04d}.keras",
        save_best_only=False,
        save_freq='epoch'
    )
]

print("[bold green]Finished building model![/bold green]")

epochs = 500

# --- TRAIN HL MODEL ---

print("[bold green]TRAINING MODEL![/bold green]")

history = model.fit(
    hl_train_dataset,
    epochs=epochs,
    validation_data=hl_valid_dataset,
    callbacks=callbacks,
    verbose=1,
)

# --- EXPORT HL TRAINING HISTORY ---

with open('hl_training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)
