from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os

# Load data generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
       'C:\\Users\\navya\\Desktop\\Train',
       target_size=(224, 224),
       batch_size=32,
       class_mode='binary',
       subset='training')

validation_generator = train_datagen.flow_from_directory(
       'C:\\Users\\navya\\Desktop\\Train',
       target_size=(224, 224),
       batch_size=32,
       class_mode='binary',
       subset='validation')

# Load and set up model
base_model = MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model with initial training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Fine-tuning: Unfreeze last few layers
for layer in base_model.layers[-100:]:
    layer.trainable = True

# Recompile with a low learning rate
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Evaluate on test data
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
       'C:\\Users\\navya\\Desktop\\Test',
       target_size=(224, 224),
       batch_size=32,
       class_mode='binary')

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Visualize single image prediction
def visualize_prediction(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    plt.imshow(img)
    plt.title(f"Prediction: {'Fresh' if prediction[0] < 0.5 else 'Rotten'}")
    plt.show()

visualize_prediction('C:\\Users\\navya\\Desktop\\b_f001.png')
