import os
import numpy as np

paths = os.listdir('../metrics')

Kmax = (0, '')
Vmax = (0, '')
Dmax = (0, '')
Amax = (0, '')

for path in paths:
    try:
        name = path.split('.npy')[0]
        with open('../metrics/' + path, 'rb') as file:
            metric_vec = np.load(file)
            (K,V,D,A) = metric_vec

            if name != 'imglazy-load-placeholder':
                if K > Kmax[0]:
                    Kmax = (K, path)

                if V > Vmax[0]:
                    Vmax = (V, path)

                if D > Dmax[0]:
                    Dmax = (D, path)

                if A > Amax[0]:
                    Amax = (A, path)

    except Exception as e:
        print(path, e)

print(Kmax)
print(Vmax)
print(Dmax)
print(Amax)
