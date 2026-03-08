import tensorflow as tf
from tensorflow.keras import models, layers
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
MODEL_PATH = os.path.join(BASE_DIR, "potato_model.keras")
HISTORY_PATH = os.path.join(BASE_DIR, "history.pkl")

BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS = 3
EPOCHS = 50

# -----------------------------
# LOAD DATASET
# -----------------------------
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names
n_classes = len(class_names)

print("Classes:", class_names)

# -----------------------------
# DATASET SPLIT FUNCTION
# -----------------------------
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):

    assert (train_split + test_split + val_split) == 1

    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# -----------------------------
# PREPROCESSING
# -----------------------------
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1./255)
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

# -----------------------------
# BUILD MODEL
# -----------------------------
model = models.Sequential([
    layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),

    resize_and_rescale,
    data_augmentation,

    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
print("\nTraining started...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)

# -----------------------------
# CREATE MODEL DIRECTORY
# -----------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# CREATE VERSION NUMBER
# -----------------------------
versions = [int(v) for v in os.listdir(MODEL_DIR) if v.isdigit()]
model_version = max(versions + [0]) + 1

print(f"\nSaving model version: {model_version}")

# -----------------------------
# SAVE MODEL FOR API (FOLDER FORMAT)
# -----------------------------
model.export(f"{MODEL_DIR}/{model_version}")

# -----------------------------
# SAVE LATEST MODEL (.keras)
# -----------------------------
model.save(MODEL_PATH)

# -----------------------------
# SAVE TRAINING HISTORY
# -----------------------------
with open("../history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("\nTraining completed and saved successfully.")