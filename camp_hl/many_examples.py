import tensorflow as tf
import sys, glob, os
sys.path.insert(0, "../")

import src.io as io
from src.plotting import *
from rich import print

# operate using gpu 2
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# load dataset
print(f"[bold yellow]Loading dataset...[/bold yellow]")
dataset = io.dataset(dir="../data/curved_amp/valid", pair=(0, 2))
val_dataset = dataset.tf_dataset(32)

# load model
model_path = "models/camp_hl_100.keras"
print(f"[bold yellow]Loading model:[/bold yellow] [bright_cyan]{model_path}[/bright_cyan]")
model = tf.keras.models.load_model(model_path)

# generate predictions
print(f"[green]Generating predictions...[/green]")
preds = model.predict(val_dataset)

plt_examples(val_dataset, preds)