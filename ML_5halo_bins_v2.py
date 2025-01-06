#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits, ascii
import h5py
from scipy.interpolate import interp2d, RegularGridInterpolator
from matplotlib.ticker import AutoMinorLocator,MaxNLocator
import astropy.table as tab
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import os 


# In[35]:


np.random.seed(1234)
h5f_test= 'ML_training_z2.0_BinLabel_test_1200_6000.hdf5'
# readin the data
with h5py.File(h5f_test, 'r') as hf:
    flux_lya = np.array(hf['flux_halo'])
    flux_NO_halo = np.array(hf['flux_NO_halo'])
    mask_halo = np.array(hf['mask_halo'])
    mask_NO_halo = np.array(hf['mask_NO_halo'])
    wavelength = np.array(hf['wavelength_range'])

flux_all = np.append(np.concatenate(flux_lya), flux_NO_halo, 0)
mask_all = np.append(np.concatenate(mask_halo), mask_NO_halo, 0)


# In[62]:


h5f_test= 'ML_test_z2.0BinLabel_test_data_500.hdf5'
# readin the data
with h5py.File(h5f_test, 'r') as hf:
    test_flux = np.concatenate(np.array(hf['test_flux']))
    test_mask = np.concatenate(np.array(hf['test_mask']))


# In[66]:


# Assuming mask_halo is already defined
# Identify the 4 different values in mask_halo (excluding 0)
unique_values = np.unique(mask_all)


# In[172]:


mask_all_new = []
unique_values = np.unique(mask_all)

for i in np.arange(0,len(unique_values[unique_values!=0])):
    mask_new = np.zeros_like(mask_all)
    label_value_i = unique_values[unique_values!=0][i]
    mask_new[mask_all==label_value_i] = i+1
    mask_all_new.append(mask_new)


# In[124]:


all_label = []
for i in np.arange(0,len(unique_values[unique_values!=0])):
    label_value_i = unique_values[unique_values!=0][i]
    ind_label_i = np.where(mask_halo[i]==label_value_i) 
    label_i = np.zeros_like(mask_halo[i])
    label_i[ind_label_i] = i+1
    all_label.append(label_i)


# In[177]:


np.unique(mask_all_new[2])


# In[110]:


#test_flux = np.array(hf['flux_lya'][test_ind])
#test_LOS_MASK = np.array(hf['LOS_MASK'][test_ind])
# Example Dataset Preparation
# Generate random data: 1000 samples, each with a sequence of 50 features
n_samples = len(flux_lya)
sequence_length = len(wavelength)

N_epochs =40
version =f'spec{len(flux_lya.flatten())}_{len(flux_NO_halo)}_allM_E{N_epochs:d}_v1'

# Define file paths for saving the model and training history
model_file = 'cnn_model_'+version+'_4_labels.keras'
log_file = 'training_log_'+version+'_4_labels.csv'


# In[178]:


# Check if the model file exists
if os.path.exists(model_file):
    # Load the model
    model = load_model(model_file)
    print("Model loaded from", model_file)
else:
        # Build the 1D CNN Model
    input_layer = Input(shape=(sequence_length, 1), name="Wavelength_Input")

    # Convolutional and pooling layers
    conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', name="Conv1")(input_layer)
    pool1 = MaxPooling1D(pool_size=2, name="Pooling1")(conv1)
    conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', name="Conv2")(pool1)
    pool2 = MaxPooling1D(pool_size=2, name="Pooling2")(conv2)

    # Flatten layer
    flatten = Flatten()(pool2)

    # Dense layers
    dense1 = Dense(128, activation='relu', name="Dense_Shared")(flatten)
    dense1 = Dropout(0.5, name="Dropout")(dense1)

    # Output layer for the first set of labels
    output1 = Dense(sequence_length, activation='linear', name='LOS_MASK_Output1')(dense1)

    # Output layer for the second set of labels
    output2 = Dense(sequence_length, activation='linear', name='LOS_MASK_Output2')(dense1)

    # Combine all into a single model
    model = Model(inputs=input_layer, outputs=[output1, output2])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss={'LOS_MASK_Output1': 'mean_squared_error', 'LOS_MASK_Output2': 'mean_squared_error'},
                  metrics={'LOS_MASK_Output1': 'mae', 'LOS_MASK_Output2': 'mae'})

        # Define the callbacks for saving the model and logging the training history
    checkpoint = ModelCheckpoint(model_file, save_best_only=True, monitor='val_loss', mode='min')
    csv_logger = CSVLogger(log_file)

    # Train the model
    history = model.fit(
        flux_all,
        {'LOS_MASK_Output1': mask_all_new[0], 'LOS_MASK_Output2': mask_all_new[1]},
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[checkpoint, csv_logger]
    )

    print("Model trained and saved to", model_file)


# In[136]:


mask_all_categorical.T[0]


# In[48]:


# Evaluate the model
results = model.evaluate(flux_all, mask_all)
print("Evaluation Results:", results)


# In[63]:


# Use the model to predict the label array
predicted_labels = model.predict(test_flux[0:5])


# In[56]:


# Plot the loss and metric curves
plt.figure(figsize=(12, 4))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.show()


# In[64]:


fig = plt.figure(figsize=(16, 5))
ax = fig.add_subplot(111)

#make a new wavelength array that has 10 times the length
wavelength_all = np.linspace(0,5,5 * len(wavelength))

ax.plot(wavelength_all, test_flux[0:5].flatten(), color = 'black',alpha =0.75,label='orig_spec')

    # Adding Twin Axes to plot using dataset_2
ax2 = ax.twinx() 
     
color = 'tab:green'
ax2.tick_params(axis ='y', labelcolor = color) 

ax2.plot(wavelength_all, test_mask[0:5].flatten(), linestyle='--',color = 'blue',alpha =0.5,label='true label')
ax2.plot(wavelength_all, predicted_labels[0:5].flatten(), linestyle=':',color = 'red',alpha =0.75,label='predict label')

ax2.set_ylabel('Halo Mass', color = color) 
ax2.set_ylim(-0.5, 13.5)

plt.legend(loc='upper left')

ax.set_ylabel('Transmission')
ax.set_xlabel('spec_pixel')

ax.set_ylim(-0.1, 1.25)

#ax.set_xlim(wavelength_all[0], wavelength_all[-1])
plt.show
#plt.savefig('ML_halomass11.0_11.5_1000halo_5000nonhalo.pdf')

#plt.close()


# In[ ]:




