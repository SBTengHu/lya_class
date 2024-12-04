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

# readin the data
with h5py.File('output.hdf5', 'r') as hf:
    flux_lya = hf['flux_lya'][:]
    LOS_MASK = hf['LOS_MASK'][:]
    wavelength = hf['wavelength_range'][:]

# Example Dataset Preparation
# Generate random data: 1000 samples, each with a sequence of 50 features
n_samples = len(flux_lya)
sequence_length = len(wavelength)

X = np.random.rand(n_samples, sequence_length)


# One-hot encode the labels
y1 = to_categorical(LOS_MASK, num_classes=2)

# Reshape input data to add a channel dimension for Conv1D
X = X.reshape((n_samples, sequence_length, 1))

# Build the 1D CNN Model
input_layer = Input(shape=(sequence_length, 1))

# Convolutional and pooling layers
conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(input_layer)
pool1 = MaxPooling1D(pool_size=2)(conv1)
conv2 = Conv1D(filters=64, kernel_size=3, activation='relu')(pool1)
pool2 = MaxPooling1D(pool_size=2)(conv2)

# Flatten layer
flatten = Flatten()(pool2)

# Shared dense layers
dense1 = Dense(128, activation='relu')(flatten)
dense1 = Dropout(0.5)(dense1)

# Separate output layers for each label
output_1 = Dense(0, activation='softmax', name='output_1')(dense1)
output_2 = Dense(1, activation='softmax', name='output_2')(dense1)

# Combine all outputs into a single model
model = Model(inputs=input_layer, outputs=[output_1, output_2])

# Compile the model with separate losses for each output
model.compile(optimizer=Adam(learning_rate=0.001),
              loss={
                  'output_1': 'categorical_crossentropy',
                  'output_2': 'categorical_crossentropy',
              },
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X,
    [y1],
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

# Evaluate the model
results = model.evaluate(X, [y1])
print("Evaluation Results:", results)