#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 12:42:13 2025

@author: chingchen
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# ---------- Model numbers ----------
model1='1072'
model2='1076'
model3='1101'
model4='1102'
model5='1103'
model6='1104'

# model1='1071'
# model2='1073'
# model3='1074'
# model4='1075'
# model5='1072'
# model6='1076'


C_values = np.array([0, 0.25, 0.5,0.75, 1, 1])
C_values = np.array([1, 0.5,0.3,3.2,10,5.6])

path = '/Users/chingchen/Desktop/Rhea/rhea_model/'

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

# ---------- y = 0.5 slice ----------
y_target = 0.5
tol = 0.02

fig,(ax) = plt.subplots(1,1,figsize=(10,8))

colors = ["#93CCB1","#D14309","#2554C7",
          "#F9DB24","#524B52","#550A35",
          "#7158FF","#008B8B","#FF8C00",
          "#455E45","#c98f49","#525252",
          "#CD5C5C","#00FF00","#FFFF00"] 
labels = [f"C={c}" for c in C_values]
labels = [f"viscosity={c} ref vis" for c in C_values]

for surf, uz, c, col, label in zip(surf_list, uz_list, C_values, colors, labels):

    x = surf[:,0]
    y = surf[:,1]
    z = surf[:,2]
    # 找到 y = 0.5 的點
    mask = np.abs(y - y_target) < tol

    x_sel = x[mask]
    topo_sel = z[mask]
    uz_sel = uz[mask]

    # 依 x 排序
    idx_sort = np.argsort(x_sel)
    x_sorted = x_sel[idx_sort]
    uz_sorted = uz_sel[idx_sort]
    z_sorted = topo_sel[idx_sort]
    # print(z_sorted)
    # plot
    ax.scatter(x_sorted, uz_sorted, color=col, label=label)
    # ax.plot(x_sorted, uz_sorted, '-', linewidth=2, color=col, label=label)
    # ax.plot(x_sorted, z_sorted, '-', linewidth=2, color=col, label=label)
    print(np.min(uz_sorted))
    
    
# ax.axhline(0, color='gray', linestyle='--')
# plt.xlabel("x", fontsize=14)
ax.set_xlim(0,1)
# plt.ylabel("Uz (at y=0.5, z=top)", fontsize=14)
# plt.title("Uz(x) Profile at y = 0.5 for Different C", fontsize=16)
ax.legend(fontsize=12)
# plt.grid(alpha=0.3)
# ax.tight_layout()
# plt.show()
fontsize=12
bwidth=2

ax.set_xlabel("x",fontsize=fontsize)
ax.set_ylabel("vertical velocity (at y=0.0, z=top)",fontsize=fontsize)
# ax.set_title("vertical velocity profile at y = 0.5 for Different C",fontsize=fontsize)
ax.set_title("vertical velocity profile at y = 0.5 for Different background viscosity",fontsize=fontsize)
# ax.set_ylabel("topography (at y=0.0, z=top)",fontsize=fontsize)
# ax.set_title("topography profile at y = 0.5 for Different C",fontsize=fontsize)
# ax.legend(title=label_title,fontsize = fontsize,title_fontsize=fontsize)
ax.grid(True)
# ax.set_xlim(0,1)
# ax.set_ylim(-1,1)

for aaa in [ax]:
    aaa.tick_params(which='minor', length=5, width=2, direction='in')
    aaa.tick_params(labelsize=fontsize,width=bwidth,length=10,right=True, top=True,direction='in',pad=15)
    for axis in ['top','bottom','left','right']:
        aaa.spines[axis].set_linewidth(bwidth)