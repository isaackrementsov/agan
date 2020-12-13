import os
import cv2
import numpy as np
from PIL import Image
from image_metrics import kinetic_energy, potential_energy, disorder, average

paths = os.listdir('../assets')

for path in paths:
    try:
        img = Image.open('../assets/' + path)
        img.load()
        img = np.asarray(img, dtype="int32")
        # Convert to rgb if rgba
        img = img[:,:,:3]

        K = kinetic_energy(img, 1)
        V = potential_energy(img, 1)
        D = disorder(img, 1)
        A = average(img, 1)
        metric_vec = np.array([K,V,D,A])

        name = path.split('.')[0]
        with open('../metrics/' + name + '.npy', 'wb') as file:
            np.save(file, metric_vec)

    except Exception as e:
        print(path, e)
