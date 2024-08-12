import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the history object
with open('training_history.pkl', 'rb') as file:
    loaded_history = pickle.load(file)

# load previous history objects as well
with open('../planar_hm/training_history.pkl', 'rb') as file:
    prev_history1 = pickle.load(file)
    
with open('../curved_hm/training_history.pkl', 'rb') as file:
    prev_history2 = pickle.load(file)

loss = np.concatenate((prev_history1['loss'], prev_history2['loss'], loaded_history['loss']))
val_loss = np.concatenate((prev_history1['val_loss'], prev_history2['val_loss'], loaded_history['val_loss']))

# Now you can use the loaded history to plot the graphs
plt.figure(figsize=(12, 4))

plt.subplot(1, 1, 1)
plt.plot(loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.ylim(np.min(val_loss), np.max(val_loss))
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()