""" MA3.py

Student: Sara Vilén
Mail: sara.vilen.3882@student.uu.se
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math 
import concurrent.futures as future
import numpy as np
from statistics import mean 
from time import perf_counter as pc
from numba import njit


# Exc1
n = {1000, 10000, 100000}   # n is the number of points

def approximate_pi(n):
    
    x_in, y_in= [], []
    x_out, y_out = [], []
    n_c = 0             # antal punkter i cirkeln 

    for i in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if math.sqrt(x**2+y**2) <= 1:
            x_in.append(x)
            y_in.append(y)
            n_c += 1
        else:
            x_out.append(x)
            y_out.append(y)

    approx_pi = 4 * (n_c/n)
    plt.scatter(x_in, y_in, color = 'red', s=1)
    plt.scatter(x_out, y_out, color = 'blue', s=1) 
    plt.savefig(f"pi_{n}.png")
    plt.show()
    plt.close()
    print('Värdet av n:\n', n, 'Approximerat värdet av pi:\n', approx_pi)
    # Write your code here
    return approx_pi


# Exc2, approximation

# # första 
# def sphere_volume(n, d, r=1):  # n is the number of points, d is the number of dimensions of the sphere 
#     X = np.random.uniform(-r, r, size=(n, d))   # skapar en matris med en rad för varje dimension d, varje rad är en punkt och varje kolonn är en koordinat 
    
#     in_sphere = []
#     out_sphere = []

#     distance_to_origo = math.sqrt(np.sum((X**2), axis=1))       # summera varje rad för sig, höj upp i två 
    
#     for i in distance_to_origo:
#         if i <= r:
#             in_sphere.append(distance_to_origo)
#         else:
#             out_sphere.append(distance_to_origo)
    
#     share_in_sphere = (np.sum(in_sphere)) / n     # beräknar andelen punkten som ligger i sfären genm att summera alla punkter och dela på totala mängden punkter

#     volume = share_in_sphere * (2*r) ** d
    
#     return volume


# # förbättrad
# def sphere_volume(n, d, r=1):
#     # n punkter, d koordinater vardera, likformigt i kuben [-r, r]^d
#     X = np.random.uniform(-r, r, size=(n, d))

#     # avstånd till origo för varje punkt (en rad = en punkt)
#     distance_to_origo = np.sqrt(np.sum(X**2, axis=1))

#     # True för punkter innanför sfären
#     in_sphere = distance_to_origo <= r

#     # andel träffar
#     share_in_sphere = np.sum(in_sphere) / n

#     # andel * kubens volym
#     volume = share_in_sphere * (2 * r) ** d
#     return volume


# med tre av funktionerna, klar

def sphere_volume(n, d, r=1):   
   
    X = np.random.uniform(-r, r, size=(n, d))  # n punkter, d koordinater vardera, likformigt i kuben [-r, r]^d

    # avstånd till origo för varje punkt (en rad = en punkt), med lambda som funktion 
    distance_to_origo = lambda X: np.sqrt(np.sum(X**2, axis=1))

    # True för punkter innanför sfären, anväder list comprehension
    in_sphere = [distance <= r for distance in distance_to_origo(X)]

    #motsvarar tidigare np.sum(in_sphere )
    hits = len(list(filter(lambda inside: inside, in_sphere)))

    # andel träffar
    share_in_sphere = hits / n

    # andel * kubens volym
    volume = share_in_sphere * (2 * r) ** d
    print(f'Den approximerade storleken är:\n {volume}')
    return volume


#Exc2, real value
def hypersphere_exact(n, d):        # n is the number of points, d is the number of dimensions of the sphere 
    return ((math.pi)**(d/2)) / math.gamma((d/2) + 1)


#Exc3: numba version
# Precis som innan 
@njit
def sphere_volume_numba(n:int, d:int)->float:
    X = np.random.uniform(-r, r, size=(n, d))  # n punkter, d koordinater vardera, likformigt i kuben [-r, r]^d

    # avstånd till origo för varje punkt (en rad = en punkt), med lambda som funktion 
    distance_to_origo = lambda X: np.sqrt(np.sum(X**2, axis=1))

    # True för punkter innanför sfären, anväder list comprehension
    in_sphere = [distance <= r for distance in distance_to_origo(X)]

    #motsvarar tidigare np.sum(in_sphere )
    hits = len(list(filter(lambda inside: inside, in_sphere)))

    # andel träffar
    share_in_sphere = hits / n

    # andel * kubens volym
    volume = share_in_sphere * (2 * r) ** d
    print(f'Den approximerade storleken är:\n {volume}')
    return volume

@njit
    # Anpassad för numba kanske?
def sphere_volume_numba(n:int, d:int)->float:
    hits = 0
    for i in range(n):
        s = 0.0
        for j in range(d):
            x = np.random.uniform(-1, 1)
            s += x * x
        if s <= 1:
            hits += 1
    return hits / n * 2 ** d
    

#Exc4: parallel code - parallelize actual computations by splitting data

def sphere_volume_parallel(n, d, np=10):
    with future.ProcessPoolExecutor(max_workers=np) as ex:
        volumes = list(ex.map(sphere_volume, [n//np] * np, [d] * np))
    return sum(volumes) / len(volumes)


    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    

    # Exc2
    n = 100000
    d = 2
    start_time = pc()
    sphere_volume(n, d)
    stop_time = pc()
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")
    

    n = 10**6
    d = 11
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")
    print(f"With n = {n}, time for python version: {stop_time - start_time}")
    # Exc3
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc3: Sequential time of {d} and {n}: {stop-start}")
    print("What is numba time?")

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")
    print("What is parallel time?")

    
    

if __name__ == '__main__':
	main()

