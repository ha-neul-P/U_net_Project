import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt

def load_data(data_dir, img_size=(128, 128)):
    images = []
    masks = []
    for img_file in os.listdir(data_dir):
        if '_mask' not in img_file:
            img_path = os.path.join(data_dir, img_file)
            mask_path = os.path.join(data_dir, img_file.replace('.jpg', '_mask.jpg'))
            
            img = load_img(img_path, target_size=img_size)
            mask = load_img(mask_path, target_size=img_size, color_mode='grayscale')
            
            img = img_to_array(img) / 255.0
            mask = img_to_array(mask) / 255.0
            
            images.append(img)
            masks.append(mask)
    return np.array(images), np.array(masks)

test_dir = './split_datasets/test'
X_test, y_test = load_data(test_dir)

TFLITE_MODEL_PATH = './Desktop/dcv.tflite'
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
print(f"Input shape: {input_shape}")

def preprocess_image(image, input_shape):
    image = load_img(image, target_size=(input_shape[1], input_shape[2]))
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

sample_image = X_test[7]
sample_mask = y_test[7]

interpreter.set_tensor(input_details[0]['index'], np.expand_dims(sample_image, axis=0))

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

predicted_mask = (output_data[0, :, :, 0] > 0.5).astype(np.float32)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(sample_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(sample_mask[:, :, 0], cmap="gray")
plt.title("True Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(predicted_mask, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")
plt.show()
