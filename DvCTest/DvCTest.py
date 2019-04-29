import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


import os  
from tqdm import tqdm
import cv2


import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 250
LR = 1e-3

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 5, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)

model.load("model.tfl")

# test_data = np.load('test_data.npy')

fig=plt.figure()

# for num,data in enumerate(test_data[:49]):
#     # cat: [1,0]
#     # dog: [0,1]
    
#     img_num = data[1]
#     img_data = data[0]
    
#     y = fig.add_subplot(7,7,num+1)
#     orig = img_data
#     data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
#     #model_out = model.predict([data])[0]
#     model_out = model.predict([data])[0]
    
#     if np.argmax(model_out) == 0: str_label='Cat'
#     elif np.argmax(model_out) == 1:str_label='Dog'+
#     elif np.argmax(model_out) == 2:str_label='Man'
#     elif np.argmax(model_out) == 3:str_label='City'
#     elif np.argmax(model_out) == 4:str_label='Forest'

        
#     y.imshow(orig,cmap='gray')
#     plt.title(str_label)
#     y.axes.get_xaxis().set_visible(False)
#     y.axes.get_yaxis().set_visible(False)
# plt.show()

test_data = []
for img in tqdm(os.listdir("test")):
	path = os.path.join("test",img)
	img_num = img.split()[0]
	img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
	test_data.append([np.array(img), img_num])


d = test_data[1]
img_data, img_num = d

data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
prediction = model.predict([data])[0]

plt.imshow(img_data, cmap = 'gray')
plt.show()

print(f"cat: {prediction[0]}, dog: {prediction[1]}, human: {prediction[2]}, city: {prediction[3]}, forest: {prediction[4]}")

f = open("save.txt", "w+")
f.write(f"cat: {prediction[0]}\ndog: {prediction[1]}\nhuman: {prediction[2]}\ncity: {prediction[3]}\nforest: {prediction[4]}")
f.close() 