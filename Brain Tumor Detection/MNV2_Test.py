import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("MNV2_Model.h5")

# Ask the user for the path of the MRI image
path = input("Please enter the path of the MRI image: ")

data = []  # initialize empty list to store image data

image = cv2.imread(path)  # read the image data into a numpy array
image = cv2.resize(image, (224, 224))  # resize the image to (224,224)
data.append(image)  # add the image data to the list of data

data = np.array(data)

# Make a prediction using the model
prediction = model.predict(data)

# Check if the image contains a tumor or not
if prediction > 0.5:
    print("The MRI scan contains a tumor.")
else:
    print("The MRI scan does not contain a tumor.")
