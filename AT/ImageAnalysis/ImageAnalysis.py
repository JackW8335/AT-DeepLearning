import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


DATADIR = "F:\AT - TensorFlowDeepLearning\Images"
CATAGORIES = ["Dogs", "Cats", "Humans"]



IMG_SIZE = 200


training_data = []

def create_training_data():
   for category in CATAGORIES:
    path = os.path.join(DATADIR, category) #path to cats or dogs directory
    class_num = CATAGORIES.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)

            new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
            training_data.append([new_array, class_num])

            
        except Exception as e:
            pass 

create_training_data()


import random
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])


x = []
y = []


for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,1)



import pickle

pickle_out = open("x.pickle","wb")
pickle.dump(x,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()
