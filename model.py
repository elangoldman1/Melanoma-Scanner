from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report
import os
import os                       # for working with files
import numpy as np              # for numerical computationss
import pandas as pd             # for working with dataframes
import seaborn as sns
import torch                    # Pytorch module
# for plotting informations on graph and images using tensors
import matplotlib.pyplot as plt
import torch.nn as nn           # for creating  neural networks
from torch.utils.data import DataLoader  # for dataloaders
import torch.nn.functional as F  # for functions for calculating loss
# for transforming images into tensors
import torchvision.transforms as transforms
from torchvision.utils import make_grid       # for data checking
# for working with classes and images
from torchvision.datasets import ImageFolder
# for getting the summary of our model
from torchsummary import summary
import tensorflow as ts
from tensorflow import keras
import itertools
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

training_dir = "./melanoma_cancer_dataset/train"
mole_types = os.listdir(training_dir)


nums_train = {}
for m in mole_types:
    path = os.path.join(training_dir, m)
    if os.path.isdir(path):
        nums_train[m] = len(os.listdir(path))

train_img_class = pd.DataFrame(
    nums_train.values(), index=nums_train.keys(), columns=["no. of images"])
print('Train data distribution :')
train_img_class

plt.figure(figsize=(10, 10))
plt.title('data distribution ', fontsize=30)
plt.ylabel('Number of images', fontsize=20)
plt.xlabel('Type of skin cancer', fontsize=20)

mel_or_ben = list(nums_train.keys())
num_images = list(nums_train.values())
sns.barplot(x=mel_or_ben, y=num_images)

aug_train = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                         rotation_range=20,
                                                         horizontal_flip=True,
                                                         validation_split=0.25
                                                         )
aug_valid = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, validation_split=0.25)
train = aug_train.flow_from_directory(aug_train, subset='training', target_size=(224, 224), batch_size=64, color_mode='rgb',
                                      class_mode='categorical', shuffle=True)

test = aug_valid.flow_from_directory(aug_train, subset='validation', target_size=(224, 224), batch_size=64, color_mode='rgb',
                                     class_mode='categorical', shuffle=False)

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(
    32, 3, activation='relu', input_shape=(224, 224, 3)))

model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Conv2D(64, 3, activation='relu'))
model.add(keras.layers.Dropout(0.15))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(train,
                    validation_data=test,
                    epochs=10)

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.title("Train and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlim(0, 10)
plt.ylim(0.0, 1.0)
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Train and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.xlim(0, 9.25)
plt.ylim(0.75, 1.0)
plt.legend()
plt.tight_layout()


Y_pred = model.predict(test)
y_pred = np.argmax(Y_pred, axis=1)

print(classification_report(test.classes, y_pred))

# calculating and plotting the confusion matrix
confusion_mat = confusion_matrix(test.classes, y_pred)
plot_confusion_matrix(conf_mat=confusion_mat, show_absolute=True,
                      show_normed=True,
                      colorbar=True)
plt.show()
