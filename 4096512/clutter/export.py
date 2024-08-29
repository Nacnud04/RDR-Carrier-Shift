
# --- INITIAL IMPORTS ---

import sys
sys.path.insert(0, "../../")
import src.io as io

import os

# run model on gpu 2
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from rich import print

# --- LOAD DATA INTO TF --- 

batch_size = 8
testing = 0.1

# create cluttersim object
cs = io.ClutterSim()
print(f"{int(cs.N*testing)} testing cluttersims")

# define model dims
cs.set_max_dims(512, 4096)

# load all cluttersims
cs.load()

# move onto gpu
_, tf_test = cs.tf_dataset(batch_size, testing=testing)

# --- LOAD MODEL ---

import tensorflow as tf
file = "models/clutter_0100.keras"
model = tf.keras.models.load_model(file)

# run model
preds = model.predict(tf_test)

# --- EXPORT AS RSF ---

# stitch model output
rdrgrms = cs.stitch(preds)

sourcedir = "/home/byrne/WORK/research/mars2024/mltrSPSLAKE/T"

for i, rdr in zip(cs.testids, rdrgrms):

    print(f"Writing output #{i+1}", end="    \r")

    io.save_rsf_2D(rdr, "outputs/mdl-"+"%03d"%i+".rsf",
                   f=f"{sourcedir}/dsyT0-r"+"%03d"%i+".rsf")