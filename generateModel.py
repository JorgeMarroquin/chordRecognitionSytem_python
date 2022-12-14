from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
import json
tf.config.list_physical_devices('GPU')

train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

print("Image Shape", cv2.imread("./assets/cmaj.png").shape)

train_dataset = train.flow_from_directory("./chromagramDataset/train", batch_size=3, class_mode='binary', target_size=(480, 640))
test_dataset = test.flow_from_directory("./chromagramDataset/test", batch_size=3, class_mode='binary', target_size=(480, 640))

print(train_dataset.class_indices)
json_object = json.dumps(train_dataset.class_indices, indent=4)
with open("class_indices.json", "w") as outfile:
    outfile.write(json_object)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(480, 640, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(25))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_dataset, steps_per_epoch=3, epochs=65, validation_data=test_dataset)

if(not os.path.exists("./saved_model")):
    os.mkdir("./saved_model")
model.save('./saved_model')

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig("lastModelAccuracy.png")