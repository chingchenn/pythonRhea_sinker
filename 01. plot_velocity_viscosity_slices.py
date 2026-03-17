#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 14:21:46 2025

@author: chingchen
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

fontsize=20
factor=5e-7
step = 8
model_number  = '1121'
# mesh = pv.read("/Users/chingchen/Desktop/Rhea/rhea_model/04_test_sinker/vtk/sinker"+str(model_number)+"_input.pvtu")
mesh = pv.read("/Users/chingchen/Desktop/Rhea/rhea_model/sinker"+str(model_number)+"_input.pvtu")
FIELD = "viscosity"   
nx, ny, nz = 101, 101, 101       
xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

# velocity
mesh_vel = pv.read("/Users/chingchen/Desktop/Rhea/rhea_model/sinker"+str(model_number)+"_solution_primary.pvtu")
# mesh = pv.read("/Users/chingchen/Desktop/Rhea/rhea_model/03_test_sinker/vtk/sinker0"+str(model_number)+"_solution_primary.pvtu")
kk = np.array(mesh_vel.point_data.get("velocity"))
tol=0.04
points = np.array(mesh_vel.points)

slices_list = [0.1,0.3,0.5,1.0]
labels = ['(a)', '(b)', '(c)', '(d)']
z0 = 0.5
y0 = 0.5

fig,(ax,ax3,ax5) = plt.subplots(3,2,figsize=(20,26))
for kkk, mmm in enumerate(slices_list):
    y0 = mmm
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    dz = (zmax - zmin) / (nz - 1)
    y0 = np.clip(y0, ymin, ymax)  

    img = pv.ImageData(dimensions=(nx, 1, nz))
    img.origin  = (xmin, y0, zmin)
    img.spacing = (dx,  1.0,  dz)
    sampled = img.sample(mesh)
    arr = np.asarray(sampled.point_data[FIELD]).reshape(nx, nz)
    arr = np.log10(arr)
    
    if kkk==0:
        aa = ax[0]
    elif kkk ==1:
        aa = ax[1]
    elif kkk ==2:
        aa = ax3[0]
    elif kkk ==3:
        aa = ax3[1]
    
    qq = aa.imshow(arr, origin='lower', extent=[xmin, xmax, zmin, zmax], cmap='jet',alpha=0.2, vmax=22.5, vmin = 19)
    aa.set_title(f'{FIELD} at y={y0}',fontsize=fontsize+4)
    
    mask  = np.abs(points[:,1] - y0) < tol
    slc = mesh_vel.slice(normal="y", origin=(0, y0, 0)) 

    x = points[mask, 0]
    z = points[mask, 2]
    u = kk[mask, 0]*factor
    w = kk[mask, 2]*factor
    up   = w >  0
    down = w <  0

    # aa.quiver(x[down][::step], z[down][::step], u[down][::step], w[down][::step], scale=2, width=0.002,
    #             headwidth=6, headlength=8,scale_units='xy', angles='xy')
    # aa.quiver(x[up][::step], z[up][::step], u[up][::step], w[up][::step],
    #            color='#D14309', scale=1, width=0.002, headwidth=6, headlength=8,
    #             scale_units='xy', angles='xy')
    aa.quiver(x[::step], z[::step], u[::step], w[::step],
               color='k', scale=1, width=0.002, headwidth=6, headlength=8,
                scale_units='xy', angles='xy')
    
    aa.set_xlabel('x',fontsize=fontsize+4)
    aa.set_ylabel('z',fontsize=fontsize+4)
    aa.text(0.02, 0.95, labels[kkk], transform=aa.transAxes,
        fontsize=fontsize+6, fontweight='bold', va='top', ha='left')

cbar_ax1 = fig.add_axes([0.48, 0.4, 0.015, 0.4])  
cbar = fig.colorbar(qq, cax=cbar_ax1, orientation='vertical', label=FIELD)
cbar_ax1.xaxis.set_label_position('top')
cbar_ax1.xaxis.set_ticks_position('top')
cbar.ax.tick_params(labelsize=fontsize+3)     
cbar_ax1.yaxis.label.set_size(fontsize+3) 

# --------------------- x-y -------------------------
slices_z_list = [0.0,1.0]
labels = ['(e)', '(f)']
for qqq, iii in enumerate(slices_z_list):
    if qqq==0:
        aa = ax5[0]
    elif qqq ==1:
        aa = ax5[1]
    z0 = iii
    
    z0 = float(np.clip(z0, zmin, zmax))                         
    mask = np.abs(points[:, 2] - z0) < tol
    
    x = points[mask, 0]
    y = points[mask, 1]
    u = kk[mask, 0] * factor
    v = kk[mask, 1] * factor

    aa.quiver(x[::step], y[::step], u[::step], v[::step],
              scale=1.2, width=0.002, headwidth=6, headlength=8)
    
    aa.set_xlabel('x',fontsize=fontsize+4)
    aa.set_ylabel('y',fontsize=fontsize+4)
    aa.set_ylim(0,1);aa.set_xlim(0,1)
    aa.set_aspect('equal', adjustable='box')
    aa.text(0.02, 0.95, labels[qqq], transform=aa.transAxes,
        fontsize=fontsize+6, fontweight='bold', va='top', ha='left',color='darkblue')
    aa.set_title(f"Velocity field on x–y plane at z={z0}",fontsize=fontsize+4)
    
