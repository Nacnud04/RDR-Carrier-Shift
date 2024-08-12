import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the history object
with open('training_history.pkl', 'rb') as file:
    loaded_history = pickle.load(file)

# Now you can use the loaded history to plot the graphs
plt.figure(figsize=(12, 4))

plt.subplot(1, 1, 1)
plt.plot(loaded_history['loss'], label='Training loss')
plt.plot(loaded_history['val_loss'], label='Validation loss')
plt.ylim(np.min(loaded_history['val_loss']), np.max(loaded_history['val_loss']))
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()