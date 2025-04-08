import cv2
import numpy as np
import tensorflow as tf
import nibabel as nib
from scipy import ndimage
import random
from sklearn.model_selection import train_test_split
import subprocess

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Return raw data
    return scan.get_fdata()

class Utils:
    def read_nifti_file(filepath):
        """Read and load volume"""
        # Read file
        scan = nib.load(filepath)
        # Return raw data
        return scan.get_fdata()
    
    def skullstrip(filepath, threshold=0.5):
        subprocess.run(["bet", f"{filepath} {filepath}_stripped -B -f {threshold} -g 0"])

    def normalize_volume(volume):
        """Normalize the volume"""
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
        return volume.astype('float32')

    def resize_volume(img, target_shape=(128, 128, 64)):
        img = img.squeeze()
        """Resize across z-axis"""
        # Set the desired depth
        desired_width = target_shape[0]
        desired_height = target_shape[1]
        desired_depth = target_shape[2]
        # Get current width, height, and depth
        current_depth = img.shape[-1]
        current_width = img.shape[0]
        current_height = img.shape[1]
        # Compute depth factor
        width_factor = desired_width / current_width
        height_factor = desired_height / current_height
        depth_factor = desired_depth / current_depth
        # Rotate
        img = ndimage.rotate(img, 90, reshape=False)
        # Resize across z-axis
        img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
        return img

    def rotate(volume):
        """Rotate the volume by a few degrees"""

        def scipy_rotate(volume):
            # define some rotation angles
            # angles = [-20, -10, -5, 5, 10, 20]
            angles = [180, 90, 0, -90]
            # pick angles at random
            angle = random.choice(angles)
            # rotate volume
            volume = ndimage.rotate(volume, angle, reshape=False)
            volume[volume < 0] = 0
            volume[volume > 1] = 1
            return volume

        augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
        return augmented_volume

    def find_brain_bounding_box(image):
        x, y, w, h = (image.shape[0] + 1, image.shape[1] + 1, -1, -1)
        for slice in image.T:
            blurred = cv2.GaussianBlur(slice.T, (13, 13), 150)
            blurred *= 255.0 / blurred.max()
            blurred[np.where(blurred < 0)] = 0
            blurred = blurred.astype(np.uint8)
            ret, thres = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            blurred_thres = cv2.medianBlur(thres, 5)
            contours, hierarchy = cv2.findContours(blurred_thres, 1, 2)
            curr_x, curr_y, curr_w, curr_h = cv2.boundingRect(contours[0])
            w = max(w, curr_x + curr_w)
            h = max(h, curr_y + curr_h)
            x = min(x, curr_x)
            y = min(y, curr_y)
        # provide a small padding around the found box:
        x = max(x - 10, 0)
        y = max(y - 10, 0)
        w = min(w + 10, image.shape[0])
        h = min(h + 10, image.shape[1])
        return x, y, w, h


def process_scan(path):
    """Read and resize volume"""
    print(path)
    # Read scan
    volume = Utils.read_nifti_file(path)
    # Average PET scans over 4th dimension -> 3D
    if len(volume.shape) == 4:
            volume = np.mean(volume, axis=3)
    # Normalize
    volume = Utils.normalize_volume(volume)
    # Resize width, height and depth
    volume = Utils.resize_volume(volume)
    # # Augment volume by rotation
    # volume = rotate(volume)
    # Add channel
    # volume = volume[..., np.newaxis]
    return volume

def train_preprocessing(volume, label):
    # Augment volume by rotation
    volume = Utils.rotate(volume)
    # Add channel
    volume = volume[..., np.newaxis]
    return volume, label

def val_preprocessing(volume, label):
    # Add channel
    volume = volume[..., np.newaxis]
    return volume, label