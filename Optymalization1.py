# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
import numpy as np
import math


x1 = np.linspace(-5, 5, 1000)
x2 = np.linspace(-5, 5, 1000)


def f1(x1, x2):
    return (x1**2 + x2 - 11)**2+(x1+x2**2-7)**2

r_min, r_max = -5.0, 5.0
xaxis = arange(r_min, r_max, 0.1)
yaxis = arange(r_min, r_max, 0.1)
x, y = np.meshgrid(xaxis, yaxis)
results = f1(x, y)
figure = plt.figure()
axis = figure.gca(projection='3d')
axis.plot_surface(x, y, results, cmap='jet', shade= "false")
axis.set_xlabel('x1')
axis.set_ylabel('x2')
axis.set_zlabel('z');
plt.show()
plt.contour(x,y,results, [0,1,2,5,10,20,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800])
plt.show()


x3 = np.linspace(-35, 35, 1000)

def ackley(variables: np.array):
    sum1 = np.sum(variables ** 2)
    sum2 = np.sum(cos(2*pi*variables))
    return -20.0 * exp(-0.02 * sqrt(1/len(variables)) * sum1) - exp(1/len(variables) * sum2) + e + 20

def f2(x1, x2):
    return -20.0 * exp(-0.02 * sqrt(0.5 * (x1**2 + x2**2)))-exp(0.5 * (cos(2 * pi * x1)+cos(2 * pi * x2))) + e + 20


r_min, r_max = -35.0, 35.0
xaxis = arange(r_min, r_max, 2.0)
yaxis = arange(r_min, r_max, 2.0)
x, y = meshgrid(xaxis, yaxis)
results = f2(x, y)
figure = plt.figure()
axis = figure.gca(projection='3d')
axis.plot_surface(x, y, results, cmap='jet', shade= "false")
axis.set_xlabel('x1')
axis.set_ylabel('x2')
axis.set_zlabel('f1(x1,x2)');
plt.show()
plt.contour(x,y,results, [1,2,3,4,5,6,7,8,9])
plt.show()


print('Wartosc funkcji f1 w punkcie (3,2):    f1(3,2) = ', f1(3,2))


print('\n\nWartosc funkcji f1 w punkcie (3,2):    f2(0,0) = ', f2(0,0))


print('\n\nWartosc funkcji f1 w punkcie (3,2):    f2(0,0) = ', ackley(np.array([0, 0])))