import tensorflow as tf
import sys
sys.path.insert(0, "../")
import src.io as io
from src.plotting import *

import numpy as np
from rich import print
import glob
import os

# run model on gpu 2
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# find all model filepaths
files = np.sort(glob.glob("models/camp_hl_*.keras"))
print(f"{len(files)} models found")
batch_size = 32

dataset = io.dataset(dir='../data/curved_amp/valid', pair=(0, 2))

datasets = []; preds = []
for f in np.sort(files):
    # Load the saved model
    print(f"[bold yellow]Loading model:[/bold yellow] [bright_cyan]{f}[/bright_cyan]")
    model = tf.keras.models.load_model(f)

    val_dataset = dataset.tf_dataset(batch_size)
    print(f"[green]Generating predictions...[/green]")
    datasets.append(val_dataset)
    preds.append(model.predict(val_dataset))

i = 57

plt_training(files, datasets, preds, i)
