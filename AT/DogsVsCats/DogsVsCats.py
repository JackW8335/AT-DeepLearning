import cv2
import tensorflow as tf
import numpy as np

CATEGORIES = ["Dogs", "Cats", "Humans"]


def prepare(filepath):
    IMG_SIZE = 200
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

model = tf.keras.models.load_model("CatsVDogs.model")


prediction = model.predict([prepare('dog3.jpg')])
print(CATEGORIES[np.argmax(prediction[0])])

###############################################
prediction = model.predict([prepare('cat2.jpeg')])
print(CATEGORIES[np.argmax(prediction[0])])

##############################################

prediction = model.predict([prepare('dog2.jpeg')])
print(CATEGORIES[np.argmax(prediction[0])])


f = open("save.txt","w+")

f.write(CATEGORIES[int(prediction[0][0])])

f.close()