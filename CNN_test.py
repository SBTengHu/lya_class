import os
import numpy as np
import h5py
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow as tf
import matplotlib.pyplot as plt


# Reproducibility
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed()

test_N=10
# Read in the data
with h5py.File('output.hdf5', 'r') as hf:
    flux_lya = np.array(hf['flux_lya'][0:-test_N])
    LOS_MASK = np.array(hf['LOS_MASK'][0:-test_N])
    test_flux = np.array(hf['flux_lya'][-test_N:])
    test_LOS_MASK = np.array(hf['LOS_MASK'][-test_N:])
    wavelength = np.array(hf['wavelength_range'])

# Example Dataset Preparation
n_samples = len(flux_lya)
sequence_length = len(wavelength)

# Define file paths for saving the model and training history
model_file = 'cnn_model.keras'
log_file = 'training_log.csv'

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

    # Output layer for binary classification
    output = Dense(sequence_length, activation='sigmoid', name='LOS_MASK_Output')(dense1)

    # Combine all into a single model
    model = Model(inputs=input_layer, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Define the callbacks for saving the model and logging the training history
    checkpoint = ModelCheckpoint(model_file, save_best_only=True, monitor='val_loss', mode='min')
    csv_logger = CSVLogger(log_file)

    # Train the model
    history = model.fit(
        flux_lya,
        LOS_MASK,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[checkpoint, csv_logger]
    )

    print("Model trained and saved to", model_file)

# Evaluate the model
results = model.evaluate(flux_lya, LOS_MASK)
print("Evaluation Results:", results)

# Use the model to predict the label array
predicted_labels = model.predict(test_flux)

# If the output layer uses sigmoid activation, you might want to threshold the predictions
# to get binary labels (0 or 1)
threshold = 0.5
predicted_labels_binary = (predicted_labels > threshold).astype(int)

print(predicted_labels_binary)

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(111)

#make a new wavelength array that has 10 times the length
wavelength_all = np.linspace(0, 10 * len(wavelength),1)

plt.plot(wavelength_all, test_flux[0:10].flatten(), label='flux',color = 'black',alpha =0.75)
plt.plot(wavelength_all, test_LOS_MASK[0:10], label='True Label', linestyle='--',color = 'blue',alpha =0.75)
plt.plot(wavelength_all, predicted_labels[0:10], label='predicted Label', linestyle=':',color = 'red',alpha =0.75)

plt.xlabel('Wavelength')
plt.ylabel('Value')
plt.show
#plt.legend()
##plt.savefig('ML_test_spec.pdf')
#plt.close()