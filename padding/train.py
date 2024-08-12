import sys
import pickle
sys.path.insert(0, "../")
import src.io as io

from rich import print
import shutil

batch_size = 64

def yprint(t):
    print(f"[bold yellow]{t}[/bold yellow]")

# --- LOAD HL DATASETS ---
yprint("Loading HL datasets...")
dataset = io.dataset(dir='../data/sizes', pair=(0, 2))
hl_train_dataset = dataset.tf_dataset(batch_size)
dataset = io.dataset(dir='../data/sizes/valid', pair=(0, 2))
hl_valid_dataset = dataset.tf_dataset(batch_size)

# --- LOAD HM DATASETS ---
yprint("Loading HM datasets...")
dataset = io.dataset(dir='../data/sizes', pair=(0, 1))
hm_train_dataset = dataset.tf_dataset(batch_size)
dataset = io.dataset(dir='../data/sizes/valid', pair=(0, 1))
hm_valid_dataset = dataset.tf_dataset(batch_size)

print("[bold green]Finished loading datasets![/bold green]")

# --- COPY EXISTING MODELS ---
yprint("Generating copy of prior models...")
shutil.copyfile("../noise/models/hl_noise_050.keras", "./hl_pad.keras")
shutil.copyfile("../noise/models/hm_noise_050.keras", "./hm_pad.keras")

# --- LOAD HL MODEL ---
import keras

yprint(f"Loading and compiling model...")
hl_model = keras.saving.load_model("./hl_pad.keras")
hl_model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mean_squared_error")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="models/hl_pad_{epoch:03d}.keras",
        save_best_only=False,
        save_freq='epoch'
    )
]

epochs = 50

# --- TRAIN HL MODEL ---
print("[bold green]TRAINING HL MODEL![/bold green]")
# train model with validation at end of epochs
history = hl_model.fit(
    hl_train_dataset,
    epochs=epochs,
    validation_data=hl_valid_dataset,
    callbacks=callbacks,
    verbose=1,
)

# --- EXPORT HL TRAINING HISTORY ---
with open('hl_training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)


# --- LOAD HM MODEL
yprint(f"Loading and compiling model...")
hm_model = keras.saving.load_model("./hm_pad.keras")
hm_model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mean_squared_error")

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="models/hm_pad_{epoch:03d}.keras",
        save_best_only=False,
        save_freq='epoch'
    )
]

# --- TRAIN HM MODEL ---
print("[bold green]TRAINING HM MODEL![/bold green]")
# train model with validation at end of epochs
history = hm_model.fit(
    hm_train_dataset,
    epochs=epochs,
    validation_data=hm_valid_dataset,
    callbacks=callbacks,
    verbose=1,
)

# --- EXPORT HM TRAINING HISTORY ---
with open('hm_training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)