
# --- INITIAL IMPORTS ---

import sys
sys.path.insert(0, "../../../")
import src.io as io
import src.model as m
from src.plotting import export_image

from rich import print

# --- LOAD DATA INTO TF --- 

batch_size = 32

# create cluttersim object
cs = io.ClutterSim()
print(f"{cs.N} cluttersims detected")

# define model dims
cs.set_max_dims(512, 512)

# load all cluttersims
cs.load()

# move onto gpu
tf_train, tf_test = cs.tf_dataset(batch_size, testing=0.1)

# check input output make sense
i = 10
for source, target in tf_train.take(i):

    # Convert the TensorFlow tensors to NumPy arrays
    source_np = source.numpy()
    target_np = target.numpy()

    print(f"Exporting image {i}")

    export_image(source_np[0,:,:], f"figs/s-r{i:03d}.png", seismic=True)
    export_image(target_np[0,:,:], f"figs/t-r{i:03d}.png", seismic=True)
    i += 1

# --- BUILD MODEL ---

print("[yellow]Building model...[/yellow]")
par = {
    "kernel_size":9
}
model, callbacks = m.new_model(cs.img_size, "clutter", lr=1e-3, par=par)
print("[bold green]Finished building model![/bold green]")

# --- TRAIN MODEL ---

epochs = 200
print("[bold green]TRAINING MODEL![/bold green]")
m.train(model, tf_train, tf_test, epochs, callbacks)

