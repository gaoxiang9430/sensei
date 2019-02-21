from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata as gd
import time

def func(x,y,z):
    return 0.5*(3)**(1/2)-((x-0.5)**2+(y-0.5)**2+(z-0.5)**2)**(1/2)
x = np.random.rand(10)
y = np.random.rand(10)
z = np.random.rand(10)
v = func(x,y,z)

print("Generate new grid...")
start_time=time.clock()
xi,yi,zi=np.ogrid[0:1:11j, 0:1:11j, 0:1:11j]
X1=xi.reshape(xi.shape[0],)
Y1=yi.reshape(yi.shape[1],)
Z1=zi.reshape(zi.shape[2],)
ar_len=len(X1)*len(Y1)*len(Z1)
X=np.arange(ar_len,dtype=float)
Y=np.arange(ar_len,dtype=float)
Z=np.arange(ar_len,dtype=float)
l=0
for i in range(0,len(X1)):
    for j in range(0,len(Y1)):
        for k in range(0,len(Z1)):
            X[l]=X1[i]
            Y[l]=Y1[j]
            Z[l]=Z1[k]
            l=l+1
print ('time needed: ', time.clock()-start_time, ' seconds')
print("")

#interpolate "data.v" on new grid "X,Y,Z"
print("Interpolate...")
start_time=time.clock()
V = gd((x,y,z), v, (X,Y,Z), method='linear')
print ('time needed: ', time.clock()-start_time, ' seconds')
print("")

#Plot original values
fig1 = plt.figure()
ax1=fig1.gca(projection='3d')
sc1=ax1.scatter(x, y, z, c=v, cmap=plt.hot())
plt.colorbar(sc1)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

#Plot interpolated values
fig2 = plt.figure()
ax2=fig2.gca(projection='3d')
sc2=ax2.scatter(X, Y, Z, c=V, cmap=plt.hot())
plt.colorbar(sc2)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

#Show plots
plt.show()
