#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 19:27:41 2025

@author: chingchen
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

C = 1
model_number  = '1050'
model_dataset = '09_test_sinker'

path = '/Users/chingchen/Desktop/Rhea/rhea_model/'
# path = f'/home/x-jchen64/rhea/scratch/{model_dataset}/vtk/'
savepath = '/Users/chingchen/Desktop/Rhea/data/input/'
# savepath = f'/home/x-jchen64/rhea/scratch/{model_dataset}/input/'



mesh = pv.read(path+'sinker'+str(model_number)+"_input.pvtu")
mesh_pri = pv.read(path+'sinker'+str(model_number)+"_solution_primary.pvtu")
mesh_sec = pv.read(path+'sinker'+str(model_number)+"_solution_secondary.pvtu")
mesh_face1 = pv.read(path+'sinker'+str(model_number)+"_solution.face1.pvtu")

stress = mesh_face1.point_data['stress_norm']
sigma_dyn = stress - stress.mean()


Ra     = 2.344042e9
alpha  = 2.0e-5
dT     = 1400.0
g = 9.81
rho = 3300.3
L = 6.371e6
#rho0g_nd = Ra / (alpha*dT)
rho0g_nd = Ra
DENOM_PA_TO_H_ND = rho * g * L
#h_update = - sigma_dyn / rho0g_nd 
h_update = - sigma_dyn  / DENOM_PA_TO_H_ND



mesh_surf = mesh_face1.points
data_surf2 = h_update * C

# np.savetxt(f'{savepath}{model_number}_mesh_xyz.txt', mesh_surf)
# np.savetxt(f'{savepath}{model_number}_test4_mesh_xyz_topo_{C}.txt', data_surf)
   
coords = mesh_face1.points
x = coords[:,0]
y = coords[:,1]

# --- (for contour) -------------------------------------------------
N = 200   # grid resolution 
xi = np.linspace(x.min(), x.max(), N)
yi = np.linspace(y.min(), y.max(), N)
Xi, Yi = np.meshgrid(xi, yi)

# 
Hi = griddata((x, y), data_surf2, (Xi, Yi), method='cubic')

# ---  contour ---------------------------------------------------------------
plt.figure(figsize=(7,6))
im = plt.contourf(Xi, Yi, Hi, 20, cmap='coolwarm')

plt.colorbar(im, label='Dynamic topography  h (nondimensional)')

plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Dynamic Topography (model {model_number})')

plt.tight_layout()
plt.show()