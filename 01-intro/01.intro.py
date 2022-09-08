from tkinter import N
import numpy as np

np.zeros(5)
np.ones(10)
np.full(10, 2.5)

a = np.array([1,2,3,5,7,12])
a[2]
a[2]=10
a

np.arange(10)
np.arange(3,10)

np.linspace(0,100, 10)

np.zeros((5,2))

n = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

n[0,1]=20
n[2] = [1,1,1]
n[:,2]= [0,1,2]
np.random.seed(2)
100*np.random.rand(5 ,2) #uniform dist
np.random.randint(low=0, high=100, size=(5 ,2)) #normal dist

a = np.arange(5)
a + 1 # no need for for loops

b = (10 + (a * 2 )) ** 2 / 100

a + b
a>b
a[a>b] 
a.mean()
a.std()
n.sum()
