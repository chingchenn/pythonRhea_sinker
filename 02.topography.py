#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 07:41:06 2025

@author: chingchen
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


fontsize=20;bwidth=2

model_number  = '0404'
model_list = ['0401','0402','0403','0409']
model_list = ['0401','0404']

path = '/Users/chingchen/Desktop/Rhea/rhea_model/'
colors_list = ["#93CCB1","#D14309","#2554C7",
          "#F9DB24","#524B52","#550A35",
          "#7158FF","#008B8B","#FF8C00",
          "#455E45","#c98f49","#525252",
          "#CD5C5C","#00FF00","#FFFF00"] 

label_list = ['1:1', '1:10', '1:100','1:1000']
# label_list = ['1e0','1e3']
label_list = ['free slip','free surface']

label_title = 'UM:UL'
label_title = 'boundary condition'


# ----------  ----------

def knn_indices_bruteforce(points, k):
    N = points.shape[0]
    idx = np.empty((N, k), dtype=int)
    for i in range(N):
        d2 = np.sum((points - points[i])**2, axis=1)
        nn = np.argpartition(d2, k+1)[:k+1]   
        nn = nn[nn != i][:k]                  
        idx[i] = nn
    return idx
# ----------  ----------
def weighted_least_squares_grad(points, values, knn_idx):
    N, k = knn_idx.shape
    grad = np.zeros((N, 3), dtype=float)
    for i in range(N):
        nbr = knn_idx[i]                 # (k,)
        dX = points[nbr] - points[i]     # (k,3)
        dF = values[nbr] - values[i]     # (k,)

        r2 = np.sum(dX*dX, axis=1) 
        h = np.sqrt(np.mean(r2)) 
        w = np.exp(-r2/(h*h))            # Guess, (k,)

        A = dX                           # (k,3)
        # (A^T W A + lam I) g = A^T W b
        AtWA = (A * w[:, None]).T @ A    # (3,3)
        AtWb = (A * w[:, None]).T @ dF   # (3,)
        AtWA.flat[::4] += 1e-8
        g = np.linalg.solve(AtWA, AtWb)  # (3,)
        grad[i] = g
    return grad


# ---------- ----------
def strain_stress_topo_unstructured(points, vel, pressure, viscosity,
                                    k=24, rho=3300.0, g=9.81, top_percent=0.2):
    z = points[:, 2]
    u = vel[:, 0]; v = vel[:, 1]; w = vel[:, 2]
    N = points.shape[0]

    eta = (np.full(N, float(viscosity), float)
           if np.isscalar(viscosity) else np.asarray(viscosity, float))

    # 1) 建 kNN
    knn_idx = knn_indices_bruteforce(points, k=k)

    # 2) 各分量梯度
    gu = weighted_least_squares_grad(points, u, knn_idx)  # (N,3) -> [du/dx, du/dy, du/dz]
    gv = weighted_least_squares_grad(points, v, knn_idx)  # (N,3)
    gw = weighted_least_squares_grad(points, w, knn_idx)  # (N,3)

    # 3) 組 Jacobian G（行=分量, 列=座標）
    #    G[i] = [[du/dx, du/dy, du/dz],
    #            [dv/dx, dv/dy, dv/dz],
    #            [dw/dx, dw/dy, dw/dz]]
    G = np.stack([gu, gv, gw], axis=1)    # (N,3,3)

    # 4) D = 0.5*(G + G^T)
    D = 0.5 * (G + np.swapaxes(G, 1, 2))  # (N,3,3)


    # 6)  σ = -p I + 2 η D
    I = np.eye(3)[None, :, :]              # (1,3,3)
    sigma = -pressure[:, None, None]*I + 2.0*eta[:, None, None]*D  # (N,3,3)

    # 7) 取頂面樣本（用最高 z 的百分位），近似 n = ẑ → σ_nn ≈ σ_zz
    z_thr = np.percentile(z, 100.0 - top_percent)
    top_mask = z >= z_thr
    sigma_zz = sigma[:, 2, 2]
    
    # check

    
    
    # if type(rho)==float:
    #     h_top = sigma_zz[top_mask] / (rho * g)
    # else:
    #     h_top = sigma_zz[top_mask] / (rho[top_mask]  * g)
        
    h_top = -sigma_zz[top_mask]
    return D, sigma, sigma_zz, top_mask, h_top


fig,(ax) = plt.subplots(1,1,figsize=(8,8))

#fig2,(ax2) = plt.subplots(1,1,figsize=(8,8))
for uuu, model_number in enumerate(model_list):
    mesh = pv.read(path+'sinker'+str(model_number)+"_input.pvtu")
    mesh_pri = pv.read(path+'sinker'+str(model_number)+"_solution_primary.pvtu")
    mesh_sec = pv.read(path+'sinker'+str(model_number)+"_solution_secondary.pvtu")
    temperature  = np.array(mesh.point_data.get('temperature'))
    vel = np.array(mesh_pri.point_data.get('velocity'))
    pressure = np.array(mesh_pri.point_data.get('pressure'))
    vis = np.array(mesh_sec.point_data.get('viscosity'))
    points = mesh.points            # (N,3)  N=8000
    
    # alpha = 2e-5
    # rho0 = 3300.0
    # kappa = 1.0e-6   
    # dT = 1400.0
    # T_top = 273.0        
    # T0 = dT  # 
    # rho = rho0 * (1 - alpha * (temperature - T0))

    eta0 = np.median(vis)                        # 代表黏度 (Pa·s)
    eta_nd  = vis / eta0     
    
    D, sigma, sigma_zz, top_mask, h_top = strain_stress_topo_unstructured(
        points = points, vel = vel,  pressure = pressure,
        viscosity = eta_nd, k = 24, #rho = rho,
        g = 9.81, top_percent = 0.2)
    
    x_top = mesh.points[top_mask, 0]
    y_top = mesh.points[top_mask, 1]
    h = h_top
    
    mask_y = np.abs(y_top - 0.5) < 0.01
    x_line = x_top[mask_y]
    h_line = h[mask_y]
    
    # 
    order = np.argsort(x_line)
    x_line = x_line[order]
    h_line = h_line[order]
    
    
    ax.scatter(x_line, h_line, color=colors_list[uuu],label=label_list[uuu])
    ax.plot(x_line, h_line, color=colors_list[uuu])
    
    # ax2.semilogx(ratios, amps, 'o-', lw=2, ms=7)
    
    
ax.set_xlabel("x",fontsize=fontsize)
ax.set_ylabel("Dynamic topography h ",fontsize=fontsize)
ax.set_title("h(x) at y = 0.5",fontsize=fontsize)
ax.legend(title=label_title,fontsize = fontsize,title_fontsize=fontsize)
ax.grid(True)
ax.set_xlim(0,1)
# ax.set_ylim(-1,10)

for aaa in [ax]:
    aaa.tick_params(which='minor', length=5, width=2, direction='in')
    aaa.tick_params(labelsize=fontsize,width=bwidth,length=10,right=True, top=True,direction='in',pad=15)
    for axis in ['top','bottom','left','right']:
        aaa.spines[axis].set_linewidth(bwidth)



