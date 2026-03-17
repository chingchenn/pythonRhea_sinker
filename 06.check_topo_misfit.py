#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 13:20:24 2025

@author: chingchen
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
# from scipy.interpolate import griddata

fontsize=12
bwidth=2

model1='1111'
model2='1113'
model3='1115'
model4='1118'


model1='1111'
model2='1120'
model3='1122'
model4='1121'
# model5='1103'
# model6='1104'
path = '/Users/chingchen/Desktop/Rhea/rhea_model/'
savepath = '/Users/chingchen/Desktop/Rhea/data/input/'


# ---------- Read meshes ----------
meshes = [
    pv.read(path + f"sinker{model1}_solution.face1.pvtu"),
    pv.read(path + f"sinker{model2}_solution.face1.pvtu"),
    pv.read(path + f"sinker{model3}_solution.face1.pvtu"),
    pv.read(path + f"sinker{model4}_solution.face1.pvtu"),
    # pv.read(path + f"sinker{model5}_solution.face1.pvtu"),
    # pv.read(path + f"sinker{model6}_solution.face1.pvtu")
]

# ---------- Extract data ----------
surf_list = [m.points for m in meshes]
uz_list   = [m.point_data['velocity'][:,2] for m in meshes]
stress_list = [m.point_data['stress_norm'] for m in meshes]
C_values = np.array([30, 3, 15, 60 ])
C_values = np.array([1, 0.3, 30, 3 ])
labels = [f"viscosity={c} ref vis" for c in C_values]

fig,(ax) = plt.subplots(1,1,figsize=(10,8))
for surf, uz, stress, c, label in zip(surf_list, uz_list, stress_list, C_values, labels):
    
    reference_uz = pv.read(path + f"sinker{model1}_solution.face1.pvtu").point_data['velocity'][:,2]
    temp_stress = pv.read(path + f"sinker{model1}_solution.face1.pvtu").point_data['stress_norm']
    ref_stress =  temp_stress - temp_stress.mean()
    g = 9.81
    rho = 3300.0
    L = 6.371e6
    ref_topo = -ref_stress /(rho*g*L) 
    
    
    x = surf[:,0]
    y = surf[:,1]
    z = surf[:,2]
    
    
    
    stress = stress - stress.mean()
    topo = -stress /(rho*g*L) 
    
    # print(ref_topo - topo)
    print(np.sum(ref_topo-topo))
    
    x_sel = x
    topo_sel = z
    uz_sel = uz
    
   
    
    misfit_topo = np.sqrt(np.sum((topo-ref_topo)**2)/len(topo))
    # print(misfit_topo)
    ax.scatter(c, misfit_topo, label=label, color='darkblue')
    
    
ax.grid(alpha=0.3)
# ax.set_xlim(0,0)
ax.set_xlabel("upper viscosity in the scale of 1e20",fontsize=fontsize)
ax.set_ylabel("misfit topography",fontsize=fontsize)
ax.set_xscale('log')
ax.set_xlim(1e-1, 100)
ax.set_title(r"misfit = $\sqrt{\frac{1}{N}(topo - reftopo)^2}$",fontsize=fontsize)
