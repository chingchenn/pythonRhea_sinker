#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 15:45:39 2025

@author: chingchen
"""
bwith=2
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def half_space_cooling_T(z, Tsurf, Tmantle,  age_in_myrs):
    diffusivity = 1e-6
    myrs2sec = 86400 * 365.2425e6

    T = Tsurf + (Tmantle - Tsurf) * erf(z /
            np.sqrt(4 * diffusivity * age_in_myrs * myrs2sec) )
    return T


layerz = (0, 1.5e3, 7.5e3, 10e3)   # 1st elem must be 0
phase=[11,3,3,4]
tem=1

deepz = layerz[-1] * 10
z = np.linspace(0, deepz, num=50000)

T = half_space_cooling_T(z, 10, 1330, 40)

fig, (ax3) = plt.subplots(1,1,figsize=(15,12))

adiabatic = 3e-5*(1330-10)*10
temp = z/1000*adiabatic+T
ax3.plot(temp,-z/1000,color='#B22222',label='temperature',lw=10)
# ax3.set_xlim(0,2000)
# ax3.set_ylim(200,0)
# ax3.axes.yaxis.set_visible(False)
ax3.spines['bottom'].set_linewidth(bwith)
ax3.spines['top'].set_linewidth(bwith)
ax3.spines['right'].set_linewidth(bwith)
ax3.spines['left'].set_linewidth(bwith)
ax3.tick_params(axis='x', labelsize=26)
ax3.tick_params(axis='y', labelsize=26)
ax3.set_title('Temperature Profile',fontsize=30)
ax3.set_xlabel('Temperature ($^\circ$C)',fontsize=26)
ax3.grid()

Lx = 1000e3
Lz = 1000e3       
nx, nz = 401, 301

x = np.linspace(0, Lx, nx)
z = np.linspace(0, Lz, nz)

X, Z = np.meshgrid(x, z)   

adiabatic_grad = 0.3   # K/km
T_background = 10 + adiabatic_grad * (Z / 1e3) + 1330 * (Z / Lz) 


age_max = 40.0                
x_trench = 400e3              
age_1d = age_max * (x / x_trench)   
age_1d[x > x_trench] = age_max
AGE = age_1d[np.newaxis, :] * np.ones_like(Z)

T_plate = half_space_cooling_T(Z, 10, 1330, 40)
T_field = T_background.copy()
mask_litho = (Z < 120e3)
T_field[mask_litho] = np.minimum(T_field[mask_litho], T_plate[mask_litho])


theta = np.deg2rad(60)   
slab_thickness = 80e3
z_slab_max = 300e3
z_line = (X - x_trench) * np.tan(theta)
dist_normal = (Z - z_line) * np.cos(theta)


mask_slab = (
    (X >= x_trench) &       
    (dist_normal >= 0.0) &
    (dist_normal <= slab_thickness) &
    (z_line <= z_slab_max)         
)
s_along = (X - x_trench) / np.cos(theta)   
age_slab = 80.0
T_slab = half_space_cooling_T(dist_normal[mask_slab],
                              10, 1330, 60)
T_field[mask_slab] = np.minimum(T_field[mask_slab], T_slab)



fig2, ax = plt.subplots(figsize=(8, 8))
im = ax.pcolormesh(X/1e3, Z/1e3, T_field, shading='auto', cmap='jet', alpha = 0.6)
ax.invert_yaxis()        
ax.set_xlabel('x (km)', fontsize=16)
ax.set_ylabel('z (km)', fontsize=16)
ax.set_title('2-D Temperature with Subduction', fontsize=18)
for side in ['bottom','top','left','right']:
    ax.spines[side].set_linewidth(bwith)
ax.tick_params(axis='both', labelsize=14)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.05)

cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Temperature (°C)', fontsize=14)
ax.set_aspect('equal')
