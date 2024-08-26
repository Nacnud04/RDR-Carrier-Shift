import tensorflow as tf
import sys
sys.path.insert(0, "../../")
import src.io as io
import numpy as np
from rich import print
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--index", default=10, type=int, help="Index to start at")
args = parser.parse_args()

model = "models/hl_overlap_0500.keras"

print(f"[yellow]Loading models...[/yellow]")
model = tf.keras.models.load_model(model)
print(f"[bold green]Loaded models![/bold green]")

print(f"[yellow]Loading testing datasets...[/yellow]")
dataset = io.dataset(dir="../../data/4096512/overlap/valid", pair=(0, 2)) 
val_dataset = dataset.tf_dataset(8, (512, 4096, 1))
sources = np.concatenate([x for x, y in val_dataset], axis=0)
targets = np.concatenate([y for x, y in val_dataset], axis=0)
print(f"[bold green]Loaded datasets![/bold green]")

print(f"[yellow]Generating predictions...[/yellow]")
preds = model.predict(val_dataset)
print(f"[bold green]Done with model predictions and initalization![/bold green]")

i = args.index 

from src.plotting import export_image as image

image(sources[i], "source.png")
image(targets[i], "target.png")
image(preds[i], "output.png")
picmax = np.max(preds[i])
picmin = np.min(preds[i])
diff = preds[i]-targets[i]
diff[0, 0] = picmax
diff[-1, -1] = picmin
image(diff, "difference.png")

