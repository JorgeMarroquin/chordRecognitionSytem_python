import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
model = tf.keras.models.load_model('../saved_model')

model.summary()
test = ImageDataGenerator(rescale=1/255)
test_dataset = test.flow_from_directory("../chromagramDataset/test", batch_size=3, class_mode='binary', target_size=(480, 640))
loss, acc = model.evaluate(test_dataset, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))