import os
import cv2
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow.keras.utils import to_categorical
BASE_DIR = '/content/drive/MyDrive/Cs/'
image_dir = os.path.join(BASE_DIR, 'Mask', 'images')
annot_dir = os.path.join(BASE_DIR, 'Mask', 'annotations')
abel2category = {'without_mask': 0, 'with_mask': 1, 'mask_weared_incorrect': 2}
category2label = {v: k for k, v in label2category.items()}
datas = []

for root, dirs, files in os.walk(annot_dir):
    for file in files:
        tree = ET.parse(os.path.join(root, file))
        data = {'path': None, 'objects': []}
        data['path'] = os.path.join(image_dir, tree.find('filename').text)
        for obj in tree.findall('object'):
            label = label2category[obj.find('name').text]
            # top left co-ordinates
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            # bottom right co-ordinates
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            data['objects'].append([label, xmin, ymin, xmax, ymax])
        datas.append(data)

print('Total images :', len(datas))
index = np.random.randint(0, len(datas))
img = cv2.imread(datas[index]['path'])
for (category, xmin, ymin, xmax, ymax) in datas[index]['objects']:
    # Draw bounding boxes
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    cv2.putText(img, str(category), (xmin+2, ymin-3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 2)
# Show image
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
img_size = (100, 100)
X = []
Y = []

for data in datas:
    img = cv2.imread(data['path'])
    for (category, xmin, ymin, xmax, ymax) in data['objects']:
        roi = img[ymin : ymax, xmin : xmax]
        roi = cv2.resize(roi, (100, 100))
        data = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        target = to_categorical(category, num_classes=len(category2label))
        X.append(data)
        Y.append(target)
        
X = np.array(X)
Y = np.array(Y)
np.save('/content/drive/My Drive/data/X', X)
np.save('/content/drive/My Drive/data/Y', Y)
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
pre_trained_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(100, 100, 3))

pre_trained_model.summary()
for layer in pre_trained_model.layers:
    layer.trainable = False
    
last_layer = pre_trained_model.get_layer('mixed7')
print('Last layer output shape :', last_layer.output_shape)
last_output = last_layer.output
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
#The Final layer with 3 outputs for 3 categories
x = tf.keras.layers.Dense(3, activation='softmax')(x)

model = tf.keras.models.Model(inputs=pre_trained_model.input, outputs=x)

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
X = np.load('/content/drive/My Drive/data/X.npy')
Y = np.load('/content/drive/My Drive/data/Y.npy')

print(X.shape, Y.shape)
ax = sns.countplot(np.argmax(Y, axis=1), palette="Set1", alpha=0.8)
ax.set_xticklabels(['without_mask', 'with_mask', 'mask_weared_incorrect'], rotation=30, ha="right", fontsize=15)
plt.show()
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='data/model-{epoch:03d}.ckpt',
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True, 
    verbose=0)

history = model.fit(X_train, 
                    Y_train, 
                    epochs=20, 
                    callbacks=[checkpoint], 
                    validation_split=0.1)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(acc, label='Training')
plt.plot(val_acc, label='Validation')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.subplot(122)
plt.plot(loss, label='Training')
plt.plot(val_loss, label='Validation')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
model.evaluate(X_test, Y_test)
Y_pred = np.argmax(model.predict(X_test), axis=1)
import time
import pickle
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model

import tensorflow as tensorFlow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Flatten, Dropout, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import PIL
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
detector=MTCNN()
ImagePath=r'/content/drive/MyDrive/Colab Notebooks/Mask/images/maksssksksss352.png'
img=cv2.imread(ImagePath)
faces = detector.detect_faces(img)
if faces:
  for face in faces:
        try:
            x, y, w, h = face['box'] 
            # Predict
            roi =  img[y : y+h, x : x+w]
            data = cv2.resize(roi, img_size)
            data = data / 255.
            data = data.reshape((1,) + data.shape)
            scores = model.predict(data)
            target = np.argmax(scores, axis=1)[0]
            # Draw bounding boxes
            #x=cv2.rectangle(img,(x, y),(x+w, y+h),(0,155,255),10)
            if target==1:
              cv2.rectangle(img=img, pt1=(x, y), pt2=(x+w, y+h), color=colors[target], thickness=2)
              print(target)
              print("mask Detected")

            elif target==0:
              
              print("No mask Detected")
            plt.imshow(img) 
        except Exception as e:
            print(e)
else:
  print("No face Detected")
detector = MTCNN()
colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (0, 255, 255)}
img_size = (100, 100)
ImagePath=r'/content/drive/MyDrive/Colab Notebooks/Mask/images/maksssksksss352.png'
img=cv2.imread(ImagePath)
#rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
faces = detector.detect_faces(img)
face_cascade = cv2.CascadeClassifier('/content/drive/MyDrive/data/haarcascade_frontalface_default.xml')
if faces:
  print("MTCNN")
  for face in faces:
        try:
            x, y, w, h = face['box'] 
            # Predict
            roi =  img[y : y+h, x : x+w]
            data = cv2.resize(roi, img_size)
            data = data / 255.
            data = data.reshape((1,) + data.shape)
            scores = model.predict(data)
            target = np.argmax(scores, axis=1)[0]
            # Draw bounding boxes
            #x=cv2.rectangle(img,(x, y),(x+w, y+h),(0,155,255),10)
            if target==1:
              cv2.rectangle(img=img, pt1=(x, y), pt2=(x+w, y+h), color=colors[target], thickness=2)
            
        except Exception as e:
            print(e)
            print(roi.shape)

  plt.imshow(img)

else:
  print("haar")
  faces = face_cascade.detectMultiScale(img, 1.3, 5)
  for (x, y, w, h) in faces:
        # Predict
        roi =  img[y : y+h, x : x+w]
        data = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), img_size)
        data = data / 255.
        data = data.reshape((1,) + data.shape)
        scores = model.predict(data)
        target = np.argmax(scores, axis=1)[0]
        # Draw bounding boxes
        cv2.rectangle(img=img, pt1=(x, y), pt2=(x+w, y+h), color=colors[target], thickness=2)
        text = "{}: {:.2f}".format(category2label[target], scores[0][target])
        cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
  plt.imshow(img)
