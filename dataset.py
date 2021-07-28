import os
import numpy as np
from numpy.lib.npyio import savez_compressed
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

size = (256, 256)
path = 'horse2zebra/'
name = 'horse2zebra.npz'

def fixImgs(path, size):
  data = []
  for img in os.listdir(path):
    imgArray = load_img(path+img, size)
    imgArray = img_to_array(imgArray)
    data.append(imgArray)
  return np.asarray(data)

horseTrain = fixImgs(path+'trainA/', size)
horseTest = fixImgs(path+'testA/', size)
zebraTrain = fixImgs(path+'trainB/', size)
zebraTest = fixImgs(path+'trainB/', size)

savez_compressed(name, horseTrain, horseTest, zebraTrain, zebraTest)
print('Dataset converted to NPZ and prepped')

