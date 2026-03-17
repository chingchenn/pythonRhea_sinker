#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 11:05:04 2025

@author: chingchen
"""

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
# from scipy.interpolate import LinearNDInterpolator
# from pathlib import Path

casename = 'shift_lateral'
# vtu_p = pv.read(f'/home/x-jqfang/scratch/rhea_2504/{casename}_solution_primary.pvtu')
# vtu_s = pv.read(f'/home/x-jqfang/scratch/rhea_2504/{casename}_solution_secondary.pvtu')

C = 1
model_number  = '0840'
model_dataset = '09_test_sinker'

path = '/Users/chingchen/Desktop/Rhea/rhea_model/'
# path = f'/home/x-jchen64/rhea/scratch/{model_dataset}/vtk/'
savepath = '/Users/chingchen/Desktop/Rhea/data/input/'
# savepath = f'/home/x-jchen64/rhea/scratch/{model_dataset}/input/'



mesh = pv.read(path+'sinker'+str(model_number)+"_input.pvtu")
mesh_pri = pv.read(path+'sinker'+str(model_number)+"_solution_primary.pvtu")
mesh_sec = pv.read(path+'sinker'+str(model_number)+"_solution_secondary.pvtu")
mesh_face1 = pv.read(path+'sinker'+str(model_number)+"_solution.face1.pvtu")



mesh = mesh_pri.points
arg_surf = np.where(np.isclose(mesh[:, 2], mesh[:, 2].max()))[0]
data = mesh_sec.point_data['stress_diag'][:, 2]-mesh_pri.point_data['pressure']
mesh_surf = mesh[arg_surf]
rhog = 3.23e4
data_surf = -(data[arg_surf]-data[arg_surf].mean())/rhog/6371e3
# data_surf = (data[arg_surf]-data[arg_surf].min())/rhog/6371e3
print(data_surf.size)

fig, ax = plt.subplots(figsize=(8, 2))
ax.scatter(mesh[arg_surf, 0], data_surf)
plt.tight_layout()
plt.show()

np.savetxt(f'/Users/chingchen/Desktop/Rhea/data/input/dy_{model_number}_points_xyz.txt', mesh_surf)
np.savetxt(f'/Users/chingchen/Desktop/Rhea/data/input/dy_{model_number}_points_xyz_factors.txt', data_surf)


reference_length = 6.371e6
reference_viscosity = 1.0e20
reference_diffusivity = 1e-6
reference_T1 = 1400.
reference_time = reference_length**2/reference_diffusivity
reference_velocity = reference_length/reference_time
reference_stress = reference_viscosity/reference_time
s_per_year = 3600.*24.*365.25
rhog = 3.23e4
reference_stress/rhog/6371e3


fig, ax = plt.subplots(figsize=(8, 6))

scatter = ax.scatter(
    mesh_surf[:, 0], mesh_surf[:, 1],
    c=data_surf, cmap='coolwarm',
    s=10             
)


cbar = fig.colorbar(scatter, ax=ax, label='Z-Axis Topography (Normalized $\\Delta h$)')

ax.set_xlabel('X Coordinate (Normalized)')
ax.set_ylabel('Y Coordinate (Normalized)')
ax.set_title('Topography on X-Y Plane (Z as Color)')

