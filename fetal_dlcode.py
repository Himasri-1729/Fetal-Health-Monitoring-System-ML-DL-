import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Dataset Path
dataset_path = "labeled_fetal_images"

# Image Data Augmentation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load Training & Validation Data
train_data = datagen.flow_from_directory(dataset_path, target_size=(224, 224), batch_size=32,
                                         class_mode='categorical', subset='training')
validation_data = datagen.flow_from_directory(dataset_path, target_size=(224, 224), batch_size=32,
                                               class_mode='categorical', subset='validation')

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: Normal, Mild Risk, Severe Risk
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model & Save .h5 File
history = model.fit(train_data, validation_data=validation_data, epochs=10)
model.save("fetal_health_model.h5")
print("✅ Model saved as fetal_health_model.h5")

# Plot Accuracy & Loss Graphs
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss')
plt.legend()

plt.show()
