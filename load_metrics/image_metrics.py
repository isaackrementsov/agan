from scipy.integrate import simps
import numpy as np

def double_simps(integrand):
    return simps([simps(slice) for slice in integrand])

def histogram(u):
    u = u.flatten()
    return np.histogram(u, bins=30, density=True)[0]

def shannon_entropy(P):
    H = 0

    for P_ui in P:
        if P_ui > 0:
            H -= P_ui*np.log(P_ui)

    return H

def kinetic_energy(img, weight):
    c_K = (252,107,3)

    k = weight*np.dot(img, c_K)
    K = double_simps(k)

    return K

def potential_energy(img, weight):
    c_V = (228,228,242)

    v = weight*np.dot(img, c_V)
    V = double_simps(v)

    return V

def disorder(img, weight):
    total_color = np.dot(img, (1,1,1))
    D = shannon_entropy(histogram(total_color))

    return D

def average(img, weight):
    c_A = (248,255,112)

    a = weight*np.dot(img, c_A)
    A = double_simps(a)

    return A
