
# --- INITIAL IMPORTS ---

import sys
sys.path.insert(0, "../../")
import src.io as io
import src.model as m

from rich import print

# --- LOAD DATA INTO TF --- 

batch_size = 8

# create cluttersim object
cs = io.ClutterSim()
print(f"{cs.N} cluttersims detected")

# define model dims
cs.set_max_dims(512, 4096)

# load all cluttersims
cs.load()

# move onto gpu
tf_train, tf_test = cs.tf_dataset(batch_size, testing=0.1)

# --- BUILD MODEL ---

print("[yellow]Building model...[/yellow]")
model, callbacks = m.existing_model("../overlap/models/hl_overlap_0500.keras", "clutter")
print("[bold green]Finished building model![/bold green]")

# --- TRAIN MODEL ---

epochs = 100
print("[bold green]TRAINING MODEL![/bold green]")
m.train(model, tf_train, tf_test, epochs, callbacks)

