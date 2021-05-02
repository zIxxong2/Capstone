import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5_5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open('6.jpg')

#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)

# display the resized image
image.show()

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)

a = prediction.argmax()       #최대값 index 찾기
print("max_index", prediction.argmax(), "max", prediction.max())




if (a == 0):
    print("당신은 계란형 얼굴 입니다.")
elif(a == 1):
    print("당신은 각진형 얼굴 입니다.")
elif(a == 2):
    print("당신은 마름모형 얼굴 입니다.")
elif (a == 3):
    print("당신은 둥근 얼굴 입니다.")
elif (a == 4):
    print("당신은 하트형 얼굴 입니다.")

print("예측: ", prediction)