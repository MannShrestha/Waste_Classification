# Waste_Classification using CNN
Waste DataSet = (https://www.kaggle.com/datasets/techsash/waste-classification-data)

## Data Description:

Dataset is divided into train data (85%) and test data (15%).

Training data - 22564 images  Test data - 2513 images. Both Training and Testing data has two class "Organic" and "Non-Organic".

## Objective:
The aim of this project is to classify an image that is either organic or Non-organic.


# Project Description
### Importing Libraries:
```
 import numpy as np
 import matplotlib
 import matplotlib.pyplot as plt
 import matplotlib.image as img
 from PIL import Image
 from sklearn.metrics import classification_report, confusion_matrix
 import keras
 from keras.models import load_model
 import tensorflow
 from tensorflow.keras.preprocessing import image
 from tensorflow.keras.utils import load_img
 from tensorflow.keras.preprocessing.image import ImageDataGenerator
 from tensorflow.keras.models import Sequential
 from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
 from tensorflow.keras.layers import Conv2D, MaxPooling2D
 from tensorflow.keras.optimizers import Nadam, SGD
 from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#To Ignore Warnings
import warnings
warnings.filterwarnings('ignore')
```

### Image Data Generator
- Image Size = 224 x 224
- Batch Size = 32 
- Epochs = 50
- Data Augmentation and its parameters

### Model BUilding
- model = Sequential()
- Conv2D
- Padding = 'same'
- kernel size = 3 x 3
- MaxPooling2D(pool_size=(2, 2)
- Dropout(0.25)
- Activation Function = relu, softmax
- callbacks = [earlystop, checkpoint, reduce_lr]
- loss Function = categorical_crossentropy
- Optimizer = Nadam
  
#### model Train Accuracy: 0.9118  & Test Accuracy: 0.9300

## Plot training & validation accuracy
## Confusion Matrix

## Evaluate the model
## Test the model using Images

