import cv2
import tensorflow as tf
import numpy as np

cata = ["Cat","Dog"]

def prepare(f):
  img_size=100
  gray = cv2.imread(f,cv2.IMREAD_GRAYSCALE)
  resized=cv2.resize(gray,(img_size,img_size))
  normalized = resized/255.0  
  reshaped=np.reshape(normalized,(1,img_size,img_size,1))
  return reshaped

model= tf.keras.models.load_model("model-008.model")

prediction = model.predict([prepare("image")])#path for the image for which you want to predict
final=prediction.flatten()
final=final.tolist()
i=final.index(max(final))
print(cata[i])
print(prediction)
