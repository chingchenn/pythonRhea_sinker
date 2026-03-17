#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 02:20:59 2026

@author: chingchen
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

fontsize=12
step=700
scale=300
model_number  = 'slab0204'
mesh = pv.read('/Users/chingchen/Desktop/Rhea/rhea_model/'+str(model_number)+'_input.pvtu')
FIELD = 'temperature'   
nx, ny, nz = 101, 101, 101       
xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

# velocity
mesh_vel = pv.read("/Users/chingchen/Desktop/Rhea/rhea_model/"+str(model_number)+"_solution_primary.pvtu")
kk = np.array(mesh_vel.point_data.get('velocity'))
tol=0.04
points = np.array(mesh_vel.points)


fig,(ax) = plt.subplots(1,1,figsize=(10,12))

y0 = 0.1
dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)
dz = (zmax - zmin) / (nz - 1)
y0 = np.clip(y0, ymin, ymax)  

img = pv.ImageData(dimensions=(nx, 1, nz))
img.origin  = (xmin, y0, zmin)
img.spacing = (dx,  1.0,  dz)
sampled = img.sample(mesh)
arr = np.asarray(sampled.point_data[FIELD]).reshape(nz, nx)


qq = ax.imshow(arr, origin='lower', extent=[xmin, xmax, zmin, zmax], cmap='jet',alpha=0.2)
ax.set_title(f'{FIELD} at y={y0}',fontsize=fontsize+4)

mask  = np.abs(points[:,1] - y0) < tol
slc = mesh_vel.slice(normal="y", origin=(0, y0, 0)) 

x = points[mask, 0]
z = points[mask, 2]
u = kk[mask, 0]
w = kk[mask, 2]
step = step
up   = w >  0
down = w <  0


speed=ax.quiver(x[::step], z[::step], u[::step], w[::step],
             color='k', scale=scale, width=0.002, headwidth=6, headlength=8,
             scale_units='xy', angles='xy')

v_ref = 10
ax.quiverkey(
    speed, X=0.95, Y=1.02, U=v_ref,
    label=f'{v_ref} mm/yr', labelpos='E',
    coordinates='axes')

ax.set_xlabel('x',fontsize=fontsize+4)
ax.set_ylabel('z',fontsize=fontsize+4)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3.5%", pad=0.10)  # size/pad 你只需決定一次
cbar = fig.colorbar(qq, cax=cax)
cbar.set_label(FIELD, fontsize=fontsize)
cbar.ax.tick_params(labelsize=fontsize)

ax.set_xlim(1.7,3)
ax.set_ylim(0.6,1)