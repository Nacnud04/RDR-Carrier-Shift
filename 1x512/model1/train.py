
# --- INITIAL IMPORTS ---

import sys
sys.path.insert(0, "../../")
import src.io as io
import src.model as m
from src.plotting import *

from rich import print

import tensorflow as tf

# --- LOAD DATA INTO TF --- 

batch_size = 128

# create cluttersim object
cs = io.ClutterSim(maxfiles=400)
print(f"{cs.N} cluttersims detected")

# define model dims
cs.set_max_dims(1, 512)

# load all cluttersims
cs.load()

# move onto gpu
tf_train, tf_test = cs.tf_dataset(batch_size, 0.1, bitdepth=32)

# --- check last N input output pairs --- 

# find batches in dataset
batches = tf_train.cardinality().numpy()
print(f"Batches: {batches}")
print(f"{(batches * batch_size) // cs.sections} input clutter sims as predicted by batch size")

view_only = False

i = 0
for source, target in tf_train.take(50):

    # Convert the TensorFlow tensors to NumPy arrays
    source_np = source.numpy()
    target_np = target.numpy()
    
    for j in range(source.shape[0]):
        
        if j % 500 == 0:

            filenum = cs.trainids[i // cs.sections]
            sectnum = int(i % cs.sections)

            print(f"Exporting image figs/{filenum:03d}-{sectnum}.png")

            export_trace_img(source_np[j,0,:], target_np[j,0,:], f"figs/{filenum:03d}-{sectnum}.png")

        i += 1

if view_only:
    sys.exit()

# --- BUILD MODEL ---

print("[yellow]Building model...[/yellow]")

par = {
    "kernel_size":3,
    "filt_depth":3,
    "clipnorm":1e-4
}

model, callbacks = m.new_model(cs.img_size, "clutter", lr=1e-4, par=par, summary=True)

print("[bold green]Finished building model![/bold green]")

# --- TRAIN MODEL ---

epochs = 500
print("[bold green]TRAINING MODEL![/bold green]")
m.train(model, tf_train, tf_test, epochs, callbacks)

