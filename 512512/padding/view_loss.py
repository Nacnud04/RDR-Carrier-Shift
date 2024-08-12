import pickle
import matplotlib.pyplot as plt
import numpy as np

variance = 0.1

# Load the history object
with open('hm_training_history.pkl', 'rb') as file:
    loaded_history = pickle.load(file)

# load previous history objects as well
with open('../planar_hm/training_history.pkl', 'rb') as file:
    prev_history1 = pickle.load(file)
with open('../curved_hm/training_history.pkl', 'rb') as file:
    prev_history2 = pickle.load(file)
with open('../camp_hm/training_history.pkl', 'rb') as file:
    prev_history3 = pickle.load(file)
with open('../noise/hm_training_history.pkl', 'rb') as file:
    prev_history4 = pickle.load(file)

hm_loss = np.concatenate((prev_history1['loss'], prev_history2['loss'], 
                          prev_history3['loss'], prev_history4['loss'],
                          loaded_history['loss']))
hm_val_loss = np.concatenate((prev_history1['val_loss'], prev_history2['val_loss'], 
                              prev_history3['val_loss'], prev_history4['val_loss'],
                              loaded_history['val_loss']))

# Load the history object
with open('hl_training_history.pkl', 'rb') as file:
    loaded_history = pickle.load(file)

# load previous history objects as well
with open('../planar_hl/training_history.pkl', 'rb') as file:
    prev_history1 = pickle.load(file)
with open('../curved_hl/training_history.pkl', 'rb') as file:
    prev_history2 = pickle.load(file)
with open('../camp_hl/training_history.pkl', 'rb') as file:
    prev_history3 = pickle.load(file)
with open('../noise/hl_training_history.pkl', 'rb') as file:
    prev_history4 = pickle.load(file)

hl_loss = np.concatenate((prev_history1['loss'], prev_history2['loss'], 
                          prev_history3['loss'], prev_history4['loss'],
                          loaded_history['loss']))
hl_val_loss = np.concatenate((prev_history1['val_loss'], prev_history2['val_loss'], 
                              prev_history3['val_loss'], prev_history4['val_loss'],
                              loaded_history['val_loss']))

# Now you can use the loaded history to plot the graphs
plt.figure(figsize=(12, 4))

plt.subplot(1, 1, 1)
plt.plot(hm_loss, label='HM Training loss', color='red')
plt.plot(hm_val_loss, label='HM Validation loss', color='pink')
plt.plot(hl_loss, label='HL Training loss', color='blue')
plt.plot(hl_val_loss, label='HL Validation loss', color='cyan')
plt.ylim(min((np.min(hm_val_loss),np.min(hl_val_loss))), max((np.max(hm_val_loss),np.max(hl_val_loss))))
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()