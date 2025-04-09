import numpy as np
import matplotlib.pyplot as plt
import keras
import visualkeras
from scipy import ndimage

# Visualize a central slice of the masked PET data
def plot_central_slices(volume, title, cmap='hot'):
    """Plot central slices of a 3D volume in all three orientations"""
    # Get central slice indices
    x_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    z_mid = volume.shape[2] // 2
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(volume[x_mid, :, :], cmap=cmap)
    plt.title(f"{title} - Sagittal")
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.face
    plt.imshow(volume[:, :, z_mid], cmap=cmap)
    plt.imshow(ndimage.rotate(volume[:, :, y_mid], -90, reshape=False), cmap=cmap)
    plt.title(f"{title} - Coronal")
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(volume[:, y_mid, :], cmap=cmap)
    plt.imshow(ndimage.rotate(volume[:, z_mid, :], 90, reshape=False), cmap=cmap)
    plt.title(f"{title} - Axial")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()


def slice(volume):
    print("Dimension of the MRI scan is:", volume.shape)
    if len(volume.shape) == 4:
        volume = np.squeeze(volume)
    plt.imsave('figure 1.png', volume[:, :, 30], cmap="gray")
    plt.show()
