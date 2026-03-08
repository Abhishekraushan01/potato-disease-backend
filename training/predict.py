import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
DATASET_PATH = os.path.join(BASE_DIR, "dataset")

IMAGE_SIZE = 256

# -----------------------------
# LOAD LATEST MODEL VERSION
# -----------------------------
versions = [int(f.split(".")[0]) for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]
model_version = max(versions)

print("Loading model version:", model_version)

model = load_model(os.path.join(MODEL_DIR, f"{model_version}.keras"))

# -----------------------------
# LOAD DATASET (FOR CLASS NAMES)
# -----------------------------
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=32
)

class_names = dataset.class_names
print("Classes:", class_names)

# -----------------------------
# DATASET SPLIT
# -----------------------------
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1):

    ds_size = len(ds)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict(img):

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0   # normalization
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array, verbose=0)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    return predicted_class, confidence


# -----------------------------
# PREDICT DATASET IMAGES
# -----------------------------
plt.figure(figsize=(15, 15))

for images, labels in test_ds.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        img = images[i].numpy().astype("uint8")

        plt.imshow(img)

        predicted_class, confidence = predict(img)
        actual_class = class_names[labels[i]]

        plt.title(f"Actual: {actual_class}\nPredicted: {predicted_class}\nConfidence: {confidence}%")

        plt.axis("off")

plt.show()

# -----------------------------
# PREDICT IMAGE FROM PC
# -----------------------------
def predict_image(img_path):

    img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))

    predicted_class, confidence = predict(img)

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence}%")
    plt.axis("off")
    plt.show()


# Example usage
# predict_image("leaf.jpg")