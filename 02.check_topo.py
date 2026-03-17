#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 20:58:33 2025

@author: chingchen
"""


import vtk
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.interpolate import griddata



model1='1072'
model2='1076'
model3='1001'
model4='0924'

path = '/Users/chingchen/Desktop/Rhea/rhea_model/'
# path='/home/x-jchen64/rhea/scratch/07_test_sinker/vtk/'
savepath = '/Users/chingchen/Desktop/Rhea/data/input/'
# savepath = '/home/x-jchen64/rhea/scratch/07_test_sinker/input/'

# mesh1 = pv.read(path+'sinker'+str(model1)+"_input.pvtu")
# mesh2 = pv.read(path+'sinker'+str(model2)+"_input.pvtu")


mesh_face1 = pv.read(path+'sinker'+str(model1)+"_solution.face1.pvtu")
mesh_face2 = pv.read(path+'sinker'+str(model2)+"_solution.face1.pvtu")
mesh_face3 = pv.read(path+'sinker'+str(model3)+"_solution.face1.pvtu")
mesh_face4 = pv.read(path+'sinker'+str(model4)+"_solution.face1.pvtu")

surf1 = mesh_face1.points
surf2 = mesh_face2.points
surf3 = mesh_face3.points
surf4 = mesh_face4.points

# topo1 = surf1[:,2]
topo2 = surf2[:,2]
topo3 = surf3[:,2]
topo4 = surf4[:,2]

uz1 = mesh_face1.point_data['velocity'][:,2]
uz2 = mesh_face2.point_data['velocity'][:,2]
uz3 = mesh_face3.point_data['velocity'][:,2]
uz4 = mesh_face4.point_data['velocity'][:,2]

def build_point_locator(points: np.ndarray):
    poly = pv.PolyData(points)
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(poly)
    loc.BuildLocator()
    return loc

locator = build_point_locator(surf2)
idx_curr_for_ref = np.array([locator.FindClosestPoint(p) for p in surf1], dtype=int)
delta_h = 1e-10 * uz2[idx_curr_for_ref]

new_update_topo = delta_h
new_update_topo -= new_update_topo.mean()

# delta_topo2 = 1e-10 * uz2

# update_topo = delta_topo2 + topo2
# update_topo = update_topo-np.mean(update_topo)

mesh_surf = mesh_face2.points
# data_surf = update_topo
data_surf = new_update_topo
print(uz2)


import matplotlib.tri as mtri
pts = mesh_face1.points
x, y, z = pts[:,0], pts[:,1], pts[:,2]
h = z -1
tri = mtri.Triangulation(x, y)

fig, ax = plt.subplots(figsize=(7,7))
cf = ax.tricontourf(tri, h, levels=30, cmap="magma")
cbar = fig.colorbar(cf, ax=ax, shrink=1.0, aspect=25, pad=0.02)
cbar.set_label("topography" )
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(x.min(), x.max()); ax.set_ylim(y.min(), y.max())
ax.set_xlabel("x" )
ax.set_ylabel("y" )
ax.set_title('topography')



fig4, ax4 = plt.subplots(figsize=(7,7))
pts = mesh_face1.points
x = pts[:, 0]
y = pts[:, 1]
ngrid=200
xi = np.linspace(x.min(), x.max(), ngrid)
yi = np.linspace(y.min(), y.max(), ngrid)
Xi, Yi = np.meshgrid(xi, yi)
levels = np.linspace(-2800, 300, 21)

Uz_grid = griddata((x, y), uz1, (Xi, Yi), method='cubic')
im = ax4.contourf(Xi, Yi, Uz_grid, levels=levels,cmap='magma',vmax=300,vmin=-2800)
cbar = plt.colorbar(im, ax=ax4)
ticks = np.linspace(-2800, 300, 6)
cbar.set_ticks(ticks)   
ax4.set_aspect('equal')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title(model1)

fig2, ax2 = plt.subplots(figsize=(7,7))
pts = mesh_face2.points
x = pts[:, 0]
y = pts[:, 1]
ngrid=200
xi = np.linspace(x.min(), x.max(), ngrid)
yi = np.linspace(y.min(), y.max(), ngrid)
Xi, Yi = np.meshgrid(xi, yi)

Uz_grid = griddata((x, y), uz2, (Xi, Yi), method='cubic')
im = ax2.contourf(Xi, Yi, Uz_grid, levels=levels,cmap='magma',vmax=300,vmin=-2800)
cbar = plt.colorbar(im, ax=ax2)
ticks = np.linspace(-2800, 300, 6)
cbar.set_ticks(ticks)  
# plt.colorbar(im, ax=ax2)
ax2.set_aspect('equal')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title(model2)

 

fig3, ax3 = plt.subplots(figsize=(7,7))
pts = mesh_face3.points
x = pts[:, 0]
y = pts[:, 1]
ngrid=200
xi = np.linspace(x.min(), x.max(), ngrid)
yi = np.linspace(y.min(), y.max(), ngrid)
Xi, Yi = np.meshgrid(xi, yi)

Uz_grid = griddata((x, y), uz3, (Xi, Yi), method='cubic')
im = ax3.contourf(Xi, Yi, Uz_grid, 20,cmap='magma')
plt.colorbar(im, ax=ax3)
ax3.set_aspect('equal')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title(model3)



