import os
import pandas as pd

# Function to get file paths and labels (Specific to OASIS-2)
def get_files_and_labels_oasis_2(dataset_path, label_path, data_path):
    labels = []
    image_files = []
    df = pd.read_excel(os.path.join(dataset_path, label_path))
    # Excludes converted
    for session, label in zip(df['MRI ID'], (df[df['Group'] != "Converted"])["Group"]):
        # Path of session MRI files
        base_path = os.path.join(dataset_path, data_path, session, "RAW")
        # Find all image files for session and apply session label
        for file in [f for f in os.listdir(base_path) if f.endswith(".img")]:
        # image_files.append(os.path.join(base_path, [f for f in os.listdir(base_path) if f.endswith(".img")][0]))
            image_files.append(os.path.join(base_path, file))
            labels.append(int(label == "Demented"))
            
    return image_files, labels

# Function to get file paths and labels (Specific to OASIS-1)
def get_files_and_labels_oasis_1(dataset_path, label_path, data_path, raw=False):
    df = pd.read_excel(os.path.join(label_path))
    df.drop(df.tail(20).index, inplace=True)
    df.dropna(subset=['CDR'], inplace=True)
    df.loc[df['CDR'] >= 0.5, 'CDR'] = 1
    labels = []
    image_files = []
    for Id, label in zip(df.ID, df.CDR):
        if raw:
            base_path = os.path.join(data_path, Id, "RAW")
            image_path = [f for f in os.listdir(base_path) if f.endswith(".img")][0]
        else:
            base_path = os.path.join(data_path, Id, "PROCESSED/MPRAGE/T88_111")
            image_path = [f for f in os.listdir(base_path) if f.endswith("masked_gfc.img")][0]
            image_files.append(os.path.join(base_path, image_path))
            labels.append(int(label))
    return image_files, labels