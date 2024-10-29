
# --- INITIAL IMPORTS ---

import sys
sys.path.insert(0, "../../")
import src.io as io
import src.model as m
from src.plotting import export_gif

from rich import print

import os
# run model on gpu 2
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# --- LOAD DATA INTO TF --- 

batch_size = 64

# create cluttersim object
cs = io.ClutterSim(dir="/home/byrne/WORK/research/mars2024/mltrSPSLAKE/M", combinerandom=True)
print(f"{cs.N} cluttersims detected")

# define model dims
cs.set_max_dims(512, 512)

# load all cluttersims
cs.load()

# move onto gpu
tf_train, tf_test = cs.tf_dataset(batch_size, testing=0.1, randomtrainsize=1000)

# --- PLOT MODEL IO ---

# find batches in dataset
batches = tf_train.cardinality().numpy()

print(f"{(batches * batch_size) // cs.sections} input clutter sims as predicted by batch size")

# grab last N batches
N = 1
last_N_pairs = tf_train.skip(batches - N)

view_only = False

i = (batches - N) * batch_size
for source, target in last_N_pairs:

    for j in range(source.shape[0]):

        filenum = cs.trainids[(i // cs.sections) % len(cs.trainids)]
        sectnum = int(i % cs.sections)

        # Convert the TensorFlow tensors to NumPy arrays
        source_np = source.numpy()
        target_np = target.numpy()

        print(f"Exporting image figs/{filenum:03d}-{sectnum}.gif")

        export_gif(source_np[j,:,:].T, target_np[j,:,:].T, f"figs/{filenum:03d}-{sectnum}.gif", seismic=True)

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

model, callbacks = m.new_model(cs.img_size, "clutter", lr=1e-3, par=par)

print("[bold green]Finished building model![/bold green]")

# --- TRAIN MODEL ---

epochs = 500
print("[bold green]TRAINING MODEL![/bold green]")
m.train(model, tf_train, tf_test, epochs, callbacks)

