#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 16:59:55 2025

@author: chingchen
"""


import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


fontsize=20;bwidth=2
model_number  = 'spb_0620'
# model_list = ['0401','0402','0403','0409']
# model_list = ['0412','0402','0413']
# model_list = ['0401','0404']
model_list = ['spb_0620',]

# model_list  = ['0903']

path = '/Users/chingchen/Desktop/Rhea/rhea_model/'
colors_list = ["#93CCB1","#D14309","#2554C7",
          "#F9DB24","#524B52","#550A35",
          "#7158FF","#008B8B","#FF8C00",
          "#455E45","#c98f49","#525252",
          "#CD5C5C","#00FF00","#FFFF00"] 

label_list = ['96','48','24']

label_title = 'elements'

fig,(ax) = plt.subplots(1,1,figsize=(8,8))
for uuu, model_number in enumerate(model_list):
    mesh = pv.read(path+str(model_number)+"_input.pvtu")
    #mesh_pri = pv.read(path+'sinker'+str(model_number)+"_solution_primary.pvtu")
    mesh_sec = pv.read(path+str(model_number)+"_solution_secondary.pvtu")
    mesh_face1 = pv.read(path+str(model_number)+"_solution.face1.pvtu")
    
    #temperature  = np.array(mesh.point_data.get('temperature'))
    #vel = np.array(mesh_pri.point_data.get('velocity'))
    #pressure = np.array(mesh_pri.point_data.get('pressure'))
    vis = np.array(mesh_sec.point_data.get('viscosity'))
    points = mesh.points
    
    eta0 = np.median(vis)
    eta_nd  = vis / eta0     
    
    points_face = mesh_face1.points
    stress_face = mesh_face1.point_data['stress_norm']
    
    z_top = points_face[:,2].max()
    tol = 1e-7  
    mask = np.abs(points_face[:,2] - z_top) < tol
    
    mask &= np.abs(points_face[:,1] - 0.5) < tol
    
    x = points_face[mask, 0]
    stress = stress_face[mask]
    
    idx = np.argsort(x)
    x = x[idx]
    h_line_face = stress[idx] - np.mean(stress[idx])
    print(len(points_face)**(1/2))
    
    ax.scatter(x, -h_line_face, color=colors_list[uuu], label=label_list[uuu])
    
    
    
    
ax.set_xlabel("x",fontsize=fontsize)
ax.set_ylabel("Dynamic topography h ",fontsize=fontsize)
ax.set_title("h(x) at y = 0.5",fontsize=fontsize)
ax.legend(title=label_title,fontsize = fontsize,title_fontsize=fontsize)
ax.grid(True)
for aaa in [ax]:
    aaa.tick_params(which='minor', length=5, width=2, direction='in')
    aaa.tick_params(labelsize=fontsize,width=bwidth,length=10,right=True, top=True,direction='in',pad=15)
    for axis in ['top','bottom','left','right']:
        aaa.spines[axis].set_linewidth(bwidth)