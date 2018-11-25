
import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()


# Returns a short sequential model
def create_model():
  model = tf.keras.models.Sequential([
      keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  
  model.compile(optimizer=tf.keras.optimizers.Adam(), 
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  
  return model



model = create_model()


model.fit(train_images, train_labels, epochs=3)


model.save('my_model.h5')


 # Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('my_model.h5')

predictions = new_model.predict(test_images)

loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))




print(np.argmax(predictions[0]))


plt.imshow(test_images[0],cmap=plt.cm.binary)
plt.show()






