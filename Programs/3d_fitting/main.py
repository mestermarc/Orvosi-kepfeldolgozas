import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sliceDataset as sD
from ellipsoid_fit import ellipsoid_plot, data_regularize, ellipsoid_fit, ellipsoid_fit2

one = sD.get_one()
two = sD.get_two()
three = sD.get_three()
four = sD.get_four()

from skimage import filters

tumor = []

tumor.append(one)
tumor.append(two)
tumor.append(three)
tumor.append(four)

edges = []

for slices in tumor:
    edge = filters.roberts(slices)
    edge[edge > 0] = 1
    int_array = edge.astype(int)
    edges.append(int_array)

pontfelho = []
slicecounter = 1
for edge in edges:
    middlex = int(round(len(edge)/2,0))
    kieg = 0
    if middlex< len(edge)/2:
        kieg=1
    for row in range(0,len(edge)):
        middley = int(round(len(edge[slicecounter])/2,0))
        for column in range(0,len(edge[slicecounter])):
            #print(row, column, slicecounter)
            if edge[row][column]>0:
                pontfelho.append((row-middlex,column-middley,slicecounter*2.31))
    slicecounter+=1

print("pontfelho hossza:", len(pontfelho))

xp=[]
yp=[]
zp=[]

for pont in pontfelho:
    xp.append(pont[0])
    yp.append(pont[1])
    zp.append(pont[2])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xp, yp, zp, zdir='z', s=20, c='b',rasterized=True)
ax.set_xlabel('x')

plt.show()

import numpy as np
import plotly.offline as go_offline
import plotly.graph_objects as go

# CREATING 3D TERRAIN MODEL

#fig=go.Figure()
#fig.add_trace(go.Surface(z=zp,x=xp,y=yp))
#fig.update_layout(scene=dict(aspectratio=dict(x=2, y=2, z=0.5),xaxis = dict(range=[-3,3],),yaxis = dict(range=[-3,3])))

#3d fit

data = np.array(pontfelho)

data2 = data_regularize(data, divs=8)

center, evecs, radii, v = ellipsoid_fit(data2)

data_centered = data - center.T
data_centered_regularized = data2 - center.T

a, b, c = radii
r = (a * b * c) ** (1. / 3.)
D = np.array([[r / a, 0., 0.], [0., r / b, 0.], [0., 0., r / c]])
# http://www.cs.brandeis.edu/~cs155/Lecture_07_6.pdf
# affine transformation from ellipsoid to sphere (translation excluded)
TR = evecs.dot(D).dot(evecs.T)
data_on_sphere = TR.dot(data_centered_regularized.T).T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# hack  for equal axes
# ax.set_aspect('equal')
# for direction in (-1, 1):
#     for point in np.diag(direction * np.max(data) * np.array([1, 1, 1])):
#         ax.plot([point[0]], [point[1]], [point[2]], 'w')

ax.scatter(data_centered[:, 0], data_centered[:, 1], data_centered[:, 2], marker='o', color='b')
# ax.scatter(data_centered_regularized[:, 0], data_centered_regularized[:, 1],
#            data_centered_regularized[:, 2], marker='o', color='b')
ellipsoid_plot([0, 0, 0], radii, evecs, ax=ax, plot_axes=True, cage_color='g')

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_on_sphere[:, 0], data_on_sphere[:, 1], data_on_sphere[:, 2], marker='o', color='r')
ellipsoid_plot([0, 0, 0], [r, r, r], evecs, ax=ax, plot_axes=True, cage_color='orange')

# ax.plot([r],[0],[0],color='r',marker='o')
# ax.plot([radii[0]],[0],[0],color='b',marker='o')

plt.show()

def plotCubeAt2(positions,sizes=None,colors=None, **kwargs):
    if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append( data )
    return Poly3DCollection(np.concatenate(g),
                            facecolors=np.repeat(colors,6), **kwargs)

