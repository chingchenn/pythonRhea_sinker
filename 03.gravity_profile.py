#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 12:48:13 2025

@author: chingchen
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

fontsize=18
bwidth=2

model_number  = '0801'
path = '/Users/chingchen/Desktop/Rhea/rhea_model/'
vol = pv.read(path+'sinker'+str(model_number)+"_input.pvtu")
face = pv.read(path+'sinker'+str(model_number)+"_solution.face1.pvtu")

temperature  = np.array(vol.point_data.get('temperature'))
xmin, xmax, ymin, ymax, zmin, zmax = vol.bounds
def to_cell_scalar_from_point(grid, point_array, name="tmp_point"):
    grid = grid.copy()
    grid.point_data.set_array(point_array, name)
    grid = grid.point_data_to_cell_data(pass_point_data=False)
    return np.asarray(grid.cell_data[name]), grid

def gravity_gz_mgal(obs_pts_m, centers_m, drho_cell, vol_cell_m3, chunk=3000):
    Xc, Yc, Zc = centers_m[:,0], centers_m[:,1], centers_m[:,2]
    W = drho_cell * vol_cell_m3
    out = np.zeros(len(obs_pts_m))
    for i0 in range(0, len(obs_pts_m), chunk):
        i1 = min(i0+chunk, len(obs_pts_m))
        xo = obs_pts_m[i0:i1,0][:,None]
        yo = obs_pts_m[i0:i1,1][:,None]
        zo = obs_pts_m[i0:i1,2][:,None]
        rx = xo - Xc[None,:]
        ry = yo - Yc[None,:]
        rz = zo - Zc[None,:]
        r2 = rx*rx + ry*ry + rz*rz
        r3 = np.power(r2, 1.5)
        r3[r3==0] = np.inf
        gz = G * np.sum(W[None,:] * rz / r3, axis=1)
        out[i0:i1] = gz / 1e-5
    return out

G = 6.674e-11
rho0 = 3300.0      # kg/m^3
alpha =  2e-05      # 1/K
dT = 1400
Lscale = 6371e6


if isinstance(vol, pv.MultiBlock):
    vol = vol.combine()

T = np.asarray(vol.point_data.get("temperature"))
drho_point = -rho0 * alpha * T  # Already in Kelvin, no scaling needed

# convert to cell data
drho_cell, vol_cell_grid = to_cell_scalar_from_point(vol, drho_point, "drho")
vol_cell_grid = vol_cell_grid.compute_cell_sizes(length=False, area=False, volume=True)
cell_centers_unit = vol_cell_grid.cell_centers().points
cell_vol_unit     = vol_cell_grid["Volume"]

# scale to SI
cell_centers_m = cell_centers_unit * Lscale
cell_vol_m3    = cell_vol_unit * (Lscale ** 3)

obs_unit = face.points
obs_m    = obs_unit * Lscale

gz_mgal = gravity_gz_mgal(
    obs_m, cell_centers_m, drho_cell, cell_vol_m3
)


fig,(ax) = plt.subplots(1,1,figsize=(8,8))
x = obs_unit[:,0]
y = obs_unit[:,1]
zval = gz_mgal

xi = np.linspace(x.min(), x.max(), 200)
yi = np.linspace(y.min(), y.max(), 200)
XI, YI = np.meshgrid(xi, yi)
ZI = griddata((x, y), zval, (XI, YI), method='linear')

c = ax.contourf(XI, YI, ZI, 30, cmap='RdBu_r')
ax.set_aspect('equal')
cbar = fig.colorbar(c, ax=ax, shrink=0.85, aspect=20, pad=0.02)
cbar.set_label("g_z (mGal)")


y0 = 0.5
tol = 1e-3

y0 = 0.5
j = np.argmin(np.abs(yi - y0))  # 找最接近 y0 的那一列
x_line = xi
g_line = ZI[j, :]

# g_line  = g_line-np.mean(g_line)
fig2,(ax2) = plt.subplots(1,1,figsize=(8,8))
ax2.plot(x_line, g_line, lw=1.5,color='darkgreen')
ax2.set_xlabel("x",fontsize=fontsize); 
ax2.set_ylabel("g_z (mGal)",fontsize=fontsize)
ax2.set_title(f'Gravity anomaly profile (y = 0.5) for {model_number}',fontsize=fontsize)
ax.set_xlabel('x',fontsize=fontsize)
ax.set_ylabel('y',fontsize=fontsize)
ax.set_title(f'Gravity anomaly map (surface) for {model_number}',fontsize=fontsize)

ax2.set_xlim(0,1)
# ax.set_ylim(-1,1)

for aaa in [ax,ax2]:
    aaa.grid(True)
    aaa.tick_params(which='minor', length=5, width=2, direction='in')
    aaa.tick_params(labelsize=fontsize,width=bwidth,length=10,right=True, top=True,direction='in',pad=15)
    for axis in ['top','bottom','left','right']:
        aaa.spines[axis].set_linewidth(bwidth)