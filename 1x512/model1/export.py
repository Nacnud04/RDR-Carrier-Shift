
# --- INITIAL IMPORTS ---

import sys
sys.path.insert(0, "../../")
import src.io as io

import os

# run model on gpu 2
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from rich import print

# --- LOAD DATA INTO TF --- 

batch_size = 128
testing = 0.1

# create cluttersim object
cs = io.ClutterSim(maxfiles=400)
print(f"{int(cs.N*testing)} testing cluttersims")

# define model dims
cs.set_max_dims(1, 512)

# load all cluttersims
cs.load()

# move onto gpu
_, tf_test = cs.tf_dataset(batch_size, testing=testing, bitdepth=32)

print(tf_test.cardinality().numpy())

# --- LOAD MODEL ---

import tensorflow as tf
file = "models/clutter_0118.keras"
model = tf.keras.models.load_model(file)

print(model.summary())

# run model
preds = model.predict(tf_test)

# --- EXPORT AS RSF ---

# stitch model output
rdrgrms = cs.stitch(preds)

#import matplotlib.pyplot as plt
#for rdr in rdrgrms:
#    plt.imshow(rdr.T, cmap="seismic", origin="upper")
#    plt.show()

sourcedir = "/home/byrne/WORK/research/mars2024/mltrSPSLAKE/T"

for i, rdr in zip(cs.testids, rdrgrms):

    print(f"Writing output #{i}", end="    \r")

    io.save_rsf_2D(rdr, "outputs/mdl-"+"%03d"%i+".rsf",
                   f=f"{sourcedir}/dsyT0-r"+"%03d"%i+".rsf")
