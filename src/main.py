import nibabel as nib
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import layers, models, Input
import matplotlib.pyplot as plt
import pandas as pd

maxinput = 0

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

dataset_path = "/home/kamran/Datasets"
label_path = "oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx"
image_path = "OAS2_RAW"

# # Function to get file paths and labels (Specific to OASIS-2)
# def get_files_and_labels(dataset_path, label_path):
#     labels = []
#     image_files = []
#     df = pd.read_excel(os.path.join(dataset_path, label_path))
#     # Excludes converted
#     for session, label in zip(df['MRI ID'], (df[df['Group'] != "Converted"])["Group"]):
#         # Path of session MRI files
#         base_path = os.path.join(dataset_path, image_path, session, "RAW")
#         # Find all image files for session and apply session label
#         for file in [f for f in os.listdir(base_path) if f.endswith(".img")]:
#             image_files.append(os.path.join(base_path, file))
#             labels.append(int(label == "Demented"))
            
#     return image_files, labels

# Get Sessions and labels
def get_sessions(dataset_path, label_path):
    # Read patient data
    df = pd.read_excel(os.path.join(dataset_path, label_path))
    # Exclude converted
    df = df[df['Group'] != "Converted"]

    g = df.groupby('Subject ID')['MRI ID'].unique()
    df2 = df[['Subject ID', 'Group']].drop_duplicates()
    df2['MRI IDs'] = df2['Subject ID'].map(g)
    print(df2)
    return df2

# Function to load NIfTI image
def load_nifti_image(file_path):
    img = nib.load(file_path)
    # print(img.header.get_data_shape())
    return img.get_fdata()

# Resize the images (e.g., to 128x128x128 or 64x64 depending on your dataset)
def resize_image(image_data, target_shape=(128, 128, 128)):
    return np.resize(image_data, target_shape)

# Normalize image data
def normalize_image(image_data):
    min = np.min(image_data)
    max = np.max(image_data)
    image_data = (image_data - min) / (max - min)
    global maxinput
    if max > maxinput:
       maxinput = max
    return image_data.astype('float32')

def create_cnn_model(input_shape):
    model = models.Sequential()
    model.add(Input(shape=input_shape))
    model.add(layers.Conv3D(32, (3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Conv3D(128, (3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout layer for regularization
    model.add(layers.Dense(1, activation='sigmoid'))  # For binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_images(X, y):
    images = []
    labels = []
    
    i = len(X)
    for subject, label in zip(X, y):
        for session in subject[1]:
            # Path of session MRI files
            base_path = os.path.join(dataset_path, image_path, session, "RAW")
            # Find all image files for session and apply session label
            # for file in [f for f in os.listdir(base_path) if f.endswith(".img")]:
            img_data = load_nifti_image(os.path.join(base_path, [f for f in os.listdir(base_path) if f.endswith(".img")][0]))
            img_data = resize_image(img_data)
            img_data = normalize_image(img_data)
            images.append(img_data)
            labels.append(label)
            # exit(0)
        i -= 1
        break
        if i % 10 == 0:
            print(i)
    return np.array(images), np.array(labels)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

sessions = get_sessions(dataset_path, label_path)

X_train, X_test, y_train, y_test = train_test_split(np.array(sessions.drop(['Group'], axis=1)),
                                                    np.array([int(x == "Demented") for x in list(sessions['Group'])]),
                                                    test_size=0.2,
                                                    stratify=np.array([int(x == "Demented") for x in list(sessions['Group'])]),
                                                    random_state=42)

X_train, y_train = load_images(X_train, y_train)
X_test, y_test = load_images(X_test, y_test)

# Create the model
model = create_cnn_model((128, 128, 128, 1))  # Input shape: (height, width, depth, channels)
model.summary()

print(maxinput)
print(X_train[0].shape)
print(X_train[0][64][64][64])
# Reshape data to include the channel dimension (necessary for CNN input)
X_train = X_train[..., np.newaxis]  # Add channel dimension
X_test = X_test[..., np.newaxis]
print(X_train[0][64][64][64])
print(X_train[0].shape)

# # Train the model
# history = model.fit(X_train, y_train, epochs=50, batch_size=2, validation_data=(X_test, y_test))


# model.save("./model-oas2-2.keras")

# # Evaluate the model
# test_loss, test_acc = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {test_acc}")

# # Plot the training and validation accuracy
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig('figure1.png')
# plt.show()
# plt.savefig('figure1-1.png')

# # Plot the training and validation loss
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('figure2.png')
# plt.show()
# plt.savefig('figure2-2.png')
