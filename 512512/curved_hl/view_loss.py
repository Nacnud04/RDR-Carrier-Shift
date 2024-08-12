import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the history object
with open('training_history.pkl', 'rb') as file:
    loaded_history = pickle.load(file)

# load previous history object as well
with open('../planar_hm/training_history.pkl', 'rb') as file:
    prev_history = pickle.load(file)

loss = np.concatenate((loaded_history['loss'], prev_history['loss']))
val_loss = np.concatenate((loaded_history['val_loss'], prev_history['val_loss']))

# Now you can use the loaded history to plot the graphs
plt.figure(figsize=(12, 4))

plt.subplot(1, 1, 1)
plt.plot(loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.ylim(np.min(val_loss), np.max(val_loss))
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()