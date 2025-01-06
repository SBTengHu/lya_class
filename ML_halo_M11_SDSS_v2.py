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


# In[3]:


np.random.seed(1234)
h5f_test= 'ML_SDSS_training_z2.0sdss_M11_test_1200_6000.hdf5'
# readin the data
with h5py.File(h5f_test, 'r') as hf:
    flux_lya = np.array(hf['flux_halo'])
    flux_NO_halo = np.array(hf['flux_NO_halo'])
    mask_halo = np.array(hf['mask_halo'])
    mask_NO_halo = np.array(hf['mask_NO_halo'])
    wavelength = np.array(hf['wavelength_range'])
    test_flux = np.array(hf['test_flux'])
    test_mask = np.array(hf['test_mask'])

flux_all = np.append(np.concatenate(flux_lya), flux_NO_halo, 0)
mask_all = np.append(np.concatenate(mask_halo), mask_NO_halo, 0)


# In[4]:


# Assuming mask_halo is already defined
# Identify the 4 different values in mask_halo (excluding 0)
unique_values = np.unique(mask_all)


# In[5]:


mask_all_new = []
unique_values = np.unique(mask_all)

for i in np.arange(0,len(unique_values[unique_values!=0])):
    print(unique_values[unique_values!=0][i])
    mask_new = np.zeros_like(mask_all)
    label_value_i = unique_values[unique_values!=0][i]
    mask_new[mask_all==label_value_i] = 1
    mask_all_new.append(mask_new)


# In[6]:


all_label = []
for i in np.arange(0,len(unique_values[unique_values!=0])):
    label_value_i = unique_values[unique_values!=0][i]
    ind_label_i = np.where(mask_halo[i]==label_value_i) 
    label_i = np.zeros_like(mask_halo[i])
    label_i[ind_label_i] = i+1
    all_label.append(label_i)


# In[8]:


#test_flux = np.array(hf['flux_lya'][test_ind])
#test_LOS_MASK = np.array(hf['LOS_MASK'][test_ind])
# Example Dataset Preparation
# Generate random data: 1000 samples, each with a sequence of 50 features
n_samples = len(flux_lya)
sequence_length = len(wavelength)

N_epochs =40
version =f'spec{len(flux_lya.flatten())}_{len(flux_NO_halo)}_M11_E{N_epochs:d}_v3'

# Define file paths for saving the model and training history
model_file = 'cnn_model_'+version+'_4_labels.keras'
log_file = 'training_log_'+version+'_4_labels.csv'


# In[13]:


{f'Output_{i}': mask_all_new[i] for i in range(len(mask_all_new))}


# In[14]:


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

    outputs = []
    for i in range(len(mask_all_new)):
        output = Dense(sequence_length, activation='sigmoid', name=f'Output_{i}')(dense1)
        outputs.append(output)

    # Combine all into a single model
    model = Model(inputs=input_layer, outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss={f'Output_{i}': 'binary_crossentropy' for i in range(len(mask_all_new))},
                  metrics={f'Output_{i}': 'accuracy' for i in range(len(mask_all_new))})

        # Define the callbacks for saving the model and logging the training history
    checkpoint = ModelCheckpoint(model_file, save_best_only=True, monitor='val_loss', mode='min')
    csv_logger = CSVLogger(log_file)

    # Train the model
    history = model.fit(
        flux_all,
       #{f'Output_{i}': mask_all_new[i] for i in range(len(mask_all_new))},
        mask_all_new,
        epochs=25,
        batch_size=32,
        validation_split=0.2,
        callbacks=[checkpoint, csv_logger]
    )

    print("Model trained and saved to", model_file)


# In[18]:


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

#plt.subplot(1, 2, 2)
#for i in range(len(mask_all_new)):
#    plt.plot(history.history[f'Output_{i}_accuracy'])
#    plt.plot(history.history[f'val_Output_{i}_accuracy'])
#plt.title('Model Accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend([f'Train Output_{i}' for i in range(len(mask_all_new))] + [f'Validation Output_{i}' for i in range(len(mask_all_new))], loc='upper right')


# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history[f'accuracy'])
plt.plot(history.history[f'val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend([f'Train Output_' for i in range(len(mask_all_new))] + [f'Validation Output_' for i in range(len(mask_all_new))], loc='upper right')


# In[19]:


# Evaluate the model
results = model.evaluate(flux_all, {f'Output_{i}': mask_all_new[i] for i in range(len(mask_all_new))})
print("Evaluation Results:", results)


# In[20]:


N_test = 10
#test_ind= np.random.choice(   np.arange(0,len(test_flux)), N_test, replace=False)

# Use the model to predict the label array
predicted_labels = model.predict(test_flux)


# In[28]:


import itertools
fig = plt.figure(figsize=(20, 5))
ax = fig.add_subplot(111)

#make a new wavelength array that has 10 times the length
wavelength_all = np.linspace(0,N_test,N_test * len(wavelength))

ax.step(wavelength_all, test_flux[:N_test].flatten(), color = 'black',alpha =0.75,label='orig_spec')

ax.axhline(y=0.25,ls=':', color = 'gray',alpha=0.8)


    # Adding Twin Axes to plot using dataset_2
ax2 = ax.twinx() 
     
color = 'tab:green'
ax2.tick_params(axis ='y', labelcolor = color) 

ax2.plot(wavelength_all, test_mask[:N_test].flatten(), linestyle='--',color = 'yellow',alpha =0.5,label='true label')

color_tuple = ( 'blue','red', 'darkgoldenrod', 'purple', 'darkolivegreen', 'darkorange', 'deeppink', 'cyan', 'dimgray')
colors = itertools.cycle(color_tuple)

for i in np.arange(0,len(mask_all_new)):
    color_i = next(colors)
    ax2.plot(wavelength_all, predicted_labels[:N_test].flatten(), linestyle=':',color = color_i,alpha =0.75,label=f'predict label {i:d}')
    #ax2.plot(wavelength_all, predicted_labels[i][:N_test].flatten(), linestyle=':',color = color_i,alpha =0.75,label=f'predict label {i:d}')

ax2.set_ylabel('Halo Mass', color = color) 
ax2.set_ylim(-0.1, 1.25)

plt.legend(loc='upper left')

ax.set_ylabel('Transmission')
ax.set_xlabel('spec_pixel')

ax.set_ylim(-0.1, 1.25)

#ax.set_xlim(wavelength_all[0], wavelength_all[-1])
#plt.show
#plt.savefig('ML_mass_bins_testv0.pdf')
plt.show
#plt.close()


# In[ ]:





# In[ ]:




