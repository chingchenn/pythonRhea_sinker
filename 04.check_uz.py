#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 20:31:16 2025

@author: chingchen
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

fontsize=12
bwidth=2

model1='1072'
model2='1076'
model3='1101'
model4='1102'
model5='1103'
model6='1104'
path = '/Users/chingchen/Desktop/Rhea/rhea_model/'
savepath = '/Users/chingchen/Desktop/Rhea/data/input/'


# ---------- Read meshes ----------
meshes = [
    pv.read(path + f"sinker{model1}_solution.face1.pvtu"),
    pv.read(path + f"sinker{model2}_solution.face1.pvtu"),
    pv.read(path + f"sinker{model3}_solution.face1.pvtu"),
    pv.read(path + f"sinker{model4}_solution.face1.pvtu"),
    pv.read(path + f"sinker{model5}_solution.face1.pvtu"),
    pv.read(path + f"sinker{model6}_solution.face1.pvtu")
]

# ---------- Extract data ----------
surf_list = [m.points for m in meshes]
uz_list   = [m.point_data['velocity'][:,2] for m in meshes]
C_values = np.array([1, 0.5,0.3,3.2,10,5.6])
labels = [f"viscosity={c} ref vis" for c in C_values]

fig,(ax) = plt.subplots(1,1,figsize=(10,8))
for surf, uz, c, label in zip(surf_list, uz_list, C_values, labels):
    
    reference_uz = pv.read(path + f"sinker{model1}_solution.face1.pvtu").point_data['velocity'][:,2]
    reference_uz = 0
    x = surf[:,0]
    y = surf[:,1]
    z = surf[:,2]
    
    x_sel = x
    topo_sel = z
    uz_sel = uz
    
    misfit_uz = np.sqrt(np.sum((uz-reference_uz)**2)/len(uz))
    
    misfit_topo = np.sqrt(np.sum((z-0)**2)/len(uz))
    print(misfit_topo)
    ax.scatter(c, misfit_uz, label=label, color='darkblue')
    
ax.grid(alpha=0.3)
ax.set_xlabel("reference viscosity",fontsize=fontsize)
ax.set_ylabel("misfit",fontsize=fontsize)
ax.set_title(r"misfit = $\sqrt{\frac{1}{N}(u_z - 0)^2}$",fontsize=fontsize)
