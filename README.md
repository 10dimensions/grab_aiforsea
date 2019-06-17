# AIForSEAsia | Grab | 2019

A Python notebook for building a model representing traffic intensities across different locations at different time periods. 

Dataset provided by Grab as part of [aiforsea.com](www.aiforsea.com) challenge.

## Pre-requisites
Google Colab (.ipynb) 

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
!pip install pygeohash
```

## Steps

### Import Libraries and Initialize Datasets

```python
import pandas as pd
import numpy as np
import datetime
import pygeohash

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

url = 'https://raw.githubusercontent.com/10dimensions/grab_aiforsea/master/training.csv'
df = pd.read_csv(url, chunksize = 10000)

```

### Pre-Processing

```python
def gh_decode(hash):
    lat, lon = pygeohash.decode(hash)
    return pd.Series({"latitude":float(lat), "longitude":float(lon)})

for chunk_df in df:
    
    chunk_df = chunk_df.join(chunk_df["geohash6"].apply(gh_decode))
    chunk_df['day'] = chunk_df['day'] % 7
    chunk_df['timestamp'] = chunk_df['timestamp'].str.split(':').apply(lambda x: ( int(x[0])*60 + int(x[1])) / 10 )
    chunk_df['demand'] = chunk_df['demand'] * 1000
    
```

### Model Definition

```python
def build_model():
  model = keras.Sequential([
    layers.Dense(32, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])
  
```

### Model Training

```python
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 25

history = model.fit(
  train_dataset, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
```

## Prediction
