#!/usr/bin/env python
# coding: utf-8

# In[2]:


# STEP 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import tensorflow as tf



# In[3]:


# STEP 2: Load and Prepare Dataset
def load_dataset(data_dir):
    image_paths = []
    labels = []
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, img_file))
                    labels.append(label)
    return pd.DataFrame({'image_path': image_paths, 'label': labels})

train_dir = '/kaggle/input/train-dataset'
test_dir = '/kaggle/input/test-dataset'

train_df = load_dataset(train_dir)
test_df = load_dataset(test_dir)

print("Train size:", len(train_df), "| Test size:", len(test_df))
print(train_df['label'].value_counts())


# In[4]:


# STEP 3: Data Generator Setup (Stronger Augmentation)
IMG_SIZE = 224
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=(0.8, 1.2)
)

train_gen = datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    subset='validation',
    shuffle=True
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = test_datagen.flow_from_dataframe(
    test_df,
    x_col='image_path',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='categorical',
    batch_size=1,
    shuffle=False
)


# In[6]:


# STEP 4: Build the Model with Regularization
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.4)(x)
num_classes = len(train_gen.class_indices)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False  # Freeze initial layers

model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# In[8]:


# STEP 5: Train the Model
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint("best_skin_model.h5", monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2, verbose=1)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks
)


# In[9]:


# STEP 6: Fine-Tune Top Layers (Optional)
# Unfreeze last 30 layers for fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks
)


# In[10]:


# STEP 7: Evaluate the Model
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))


# In[11]:


# STEP 8: Predict on Single Image
def predict_image(path):
    img = Image.open(path).resize((IMG_SIZE, IMG_SIZE))
    img_array = preprocess_input(np.expand_dims(np.array(img), axis=0))
    prediction = model.predict(img_array)
    class_label = list(train_gen.class_indices.keys())[np.argmax(prediction)]
    confidence = np.max(prediction)
    print(f"Predicted class: {class_label} ({confidence:.2f})")


# In[12]:


predict_image('/kaggle/input/test-dataset/Eczemaa/10_eczema-lids-9.jpg')


# In[13]:


predict_image('/kaggle/input/test-dataset/Acne/40_07AcnePittedScars1.jpg')


# In[14]:


predict_image('/kaggle/input/test-dataset/normal/188_Selfie_12.jpg')


# In[15]:


predict_image('/kaggle/input/test-dataset/Basal Cell Carcinoma/155_basal-cell-carcinoma-lid-22.jpg')


# In[16]:


# STEP 9: Save the Final Trained Model
model.save("final_skin_disease_model.h5")
print("✅ Final model saved as 'final_skin_disease_model.h5'")    

