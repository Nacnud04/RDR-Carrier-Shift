import pickle
import matplotlib.pyplot as plt
import numpy as np

variance = 0.1

# Load the history object
with open('training_history.pkl', 'rb') as file:
    loaded_history = pickle.load(file)

loss = loaded_history['loss']
val_loss = loaded_history['val_loss']

# Now you can use the loaded history to plot the graphs
plt.figure(figsize=(12, 4))

plt.subplot(1, 1, 1)
plt.plot(loss, label='HM Training loss', color='red')
plt.plot(val_loss, label='HM Validation loss', color='pink')
#plt.ylim(min((np.min(hm_val_loss),np.min(hl_val_loss))), max((np.max(hm_val_loss),np.max(hl_val_loss))))
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()