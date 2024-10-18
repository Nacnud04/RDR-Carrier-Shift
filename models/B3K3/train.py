
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
cs = io.ClutterSim()
print(f"{cs.N} cluttersims detected")

# define model dims
cs.set_max_dims(512, 512)

# load all cluttersims
cs.load()

# move onto gpu
tf_train, tf_test = cs.tf_dataset(batch_size, testing=0.1)

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

