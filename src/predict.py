import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

from models import get_3DCNN_model
from utils.preprocessing import process_scan


if __name__ == "__main__":
    dataset_path = "/home/kamran/projects/final-project/Datasets/Oasis-1"
    label_path = "/home/kamran/projects/final-project/Datasets/Oasis-1/oasis_cross-sectional-5708aa0a98d82080.xlsx"
    data_path = "/home/kamran/projects/final-project/Datasets/Oasis-1/MRI/Data"

    # dataset_path = "/home/kamran/projects/final-project/Datasets/Oasis-2"
    # label_path = "/home/kamran/projects/final-project/Datasets/Oasis-2/oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx"
    # data_path = "/home/kamran/projects/final-project/Datasets/Oasis-2/OAS2_RAW"

    df = pd.read_excel(os.path.join(label_path))
    df.drop(df.tail(20).index, inplace=True)
    df.dropna(subset=['CDR'], inplace=True)
    df.loc[df['CDR'] >= 0.5, 'CDR'] = 1
    x_test = list(df['ID'])
    y_test = list(df.CDR)

    # Build model.
    model = get_3DCNN_model(width=128, height=128, depth=64)
    # model.summary()

    # Load best weights.
    model.load_weights("/home/kamran/projects/final-project/models/oasis-1-proc-v1/model-oas1-1.keras")

    y_pred = []
    target_names = ['Non-Demented', 'Demented']
    for x in range(len(x_test)):
        # base_path = os.path.join(data_path, x_test[x], "RAW")
        base_path = os.path.join(data_path, x_test[x], "PROCESSED/MPRAGE/T88_111")
        # image_path = [f for f in os.listdir(base_path) if f.endswith(".img")][0]
        image_path = [f for f in os.listdir(base_path) if f.endswith("masked_gfc.img")][0]
        image = process_scan(os.path.join(base_path, image_path))[..., np.newaxis]
        prediction = model.predict(np.expand_dims(image, axis=0))[0][0]

        y_pred.append(prediction)
        # scores = [1 - prediction[0], prediction[0]]

        # class_names = ["normal", "abnormal"]
        # for score, name in zip(scores, class_names):
        #     print(
        #         "This model is %.2f percent confident that MRI scan is %s"
        #         % ((100 * score), name)
        #     )

    y_pred = np.array(y_pred)
    y_pred = y_pred > 0.5
    print(classification_report(np.array(y_test), y_pred, target_names=target_names))

    cm = confusion_matrix(np.array(y_test), y_pred)

    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Demented", "Demented"])

    cm_display.plot()
    plt.savefig("cm.png") 
    
