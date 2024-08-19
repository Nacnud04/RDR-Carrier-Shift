import sys
sys.path.insert(0, "../../../../")
import src.io as io

import numpy as np
from PIL import Image

dataset = io.dataset(dir='.', pair=(0, 2))
h, l = dataset.cs.grab_pair(int(sys.argv[1]), pair=(0, 2))

h = np.squeeze(h)
normalized_data = (255 * (h - h.min()) / (h.max() - h.min())).astype(np.uint8)

image = Image.fromarray(normalized_data.T, mode='L')
image.save('highcarrier.png')

l = np.squeeze(l)
normalized_data = (255 * (l - l.min()) / (l.max() - l.min())).astype(np.uint8)

image = Image.fromarray(normalized_data.T, mode='L')
image.save('lowcarrier.png')