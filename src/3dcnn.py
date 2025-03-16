import os
import random
import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from loader import get_files_and_labels_oasis_1, get_files_and_labels_oasis_2
from src.preprocessing.preprocessing import process_scan, train_preprocessing, val_preprocessing
from models import get_3DCNN_model, visualize_model
import keras


random.seed(42)

# directory name for model
version = 1
model_prefix = "Oasis-1-PROC"
while os.path.isdir("./models/" + model_prefix + "-V" + str(version)):
    version += 1
model_path = "./models/" + model_prefix + "-V" + str(version)

# Oasis 2 RAW
dataset_path = "/home/kamran/projects/final-project/Datasets/Oasis-2"
label_path = "oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx"
data_path = "OAS2_RAW"

# Oasis 1
# dataset_path = "/home/kamran/projects/final-project/Datasets/Oasis-1"
# label_path = "/home/kamran/projects/final-project/Datasets/Oasis-1/oasis_cross-sectional-5708aa0a98d82080.xlsx"
# data_path = "/home/kamran/projects/final-project/Datasets/Oasis-1/MRI/Data"

# Reads the file paths and labels of MRI scans
files, labels = get_files_and_labels_oasis_1(dataset_path, label_path, data_path)

# Split data in the ratio 80-20 for training and validation.
# splitting is done using file paths for efficiency
X_train, X_test, y_train, y_test = train_test_split(np.array(files),
                                                    np.array(labels),
                                                    test_size=0.2,
                                                    stratify=np.array(labels),
                                                    random_state=42)

# Read and process the scans.
# Each scan is resized across height, width, and depth and rescaled.
# X_train = np.array([process_scan(path) for path in X_train.tolist()])
# X_test = np.array([process_scan(path) for path in X_test.tolist()])
with ThreadPoolExecutor(max_workers=4) as executor:
    X_train = list(executor.map(process_scan, X_train))
with ThreadPoolExecutor(max_workers=4) as executor:
    X_test = list(executor.map(process_scan, X_test))

X_train = np.array(X_train)
X_test = np.array(X_test)

print(
    "Number of samples in train and validation are %d and %d."
    % (X_train.shape[0], X_test.shape[0])
)

# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((X_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((X_test, y_test))


batch_size = 2
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(X_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(X_test))
    .map(val_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

# Build model.
model = get_3DCNN_model(width=128, height=128, depth=64)
model.summary()

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    model_path + "/" + model_path + "_best.keras", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 100
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)

model.save(model_path + "/" + model_path + ".keras")
visualize_model(model, model_path)

#Plot graph of accuracy and loss
fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

fig.savefig('figure 3.png')



