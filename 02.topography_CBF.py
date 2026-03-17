#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 01:59:42 2025

@author: chingchen
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


fontsize=20;bwidth=2
CBF = 0

model_number  = '0761'
# model_list = ['0401','0402','0403','0409']
# model_list = ['0412','0402','0413']
# model_list = ['0401','0404']
model_list = ['0904',]


path = '/Users/chingchen/Desktop/Rhea/rhea_model/'
colors_list = ["#93CCB1","#D14309","#2554C7",
          "#F9DB24","#524B52","#550A35",
          "#7158FF","#008B8B","#FF8C00",
          "#455E45","#c98f49","#525252",
          "#CD5C5C","#00FF00","#FFFF00"] 

label_list = ['1:1', '1:10', '1:100','1:1000']
# label_list = ['8','16','32']
label_list = ['free slip','free surface']

label_title = 'UM:UL'
# label_title = 'elements'
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

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def compute_sigma_all_points(points, vel, pressure, viscosity, k=32):
    u, v, w = vel[:,0], vel[:,1], vel[:,2]
    knn = knn_indices_bruteforce(points, k)
    gu = weighted_least_squares_grad(points, u, knn)
    gv = weighted_least_squares_grad(points, v, knn)
    gw = weighted_least_squares_grad(points, w, knn)
    G  = np.stack([gu, gv, gw], axis=1)                 # (N,3,3)
    D  = 0.5*(G + np.swapaxes(G,1,2))                   # (N,3,3)
    eta = viscosity if np.ndim(viscosity)>0 else np.full(points.shape[0], float(viscosity))
    I = np.eye(3)[None,:,:]
    sigma = -pressure[:,None,None]*I + 2.0*eta[:,None,None]*D
    return sigma  # (N,3,3)

def compute_sigma_prime(points_nd, vel_nd, p_nd, eta_nd, k_grad=32, lam=1e-8):
    u, v, w = vel_nd[:,0], vel_nd[:,1], vel_nd[:,2]
    knn = knn_indices_bruteforce(points_nd, k_grad)
    gu = weighted_least_squares_grad(points_nd, u, knn)  # (N,3)
    gv = weighted_least_squares_grad(points_nd, v, knn)
    gw = weighted_least_squares_grad(points_nd, w, knn)
    G  = np.stack([gu, gv, gw], axis=1)                           # (N,3,3)
    D  = 0.5 * (G + np.swapaxes(G, 1, 2))                         # (N,3,3)
    if np.isscalar(eta_nd):
        eta_arr = np.full(points_nd.shape[0], float(eta_nd))
    else:
        eta_arr = np.asarray(eta_nd, float)
    I = np.eye(3)[None,:,:]
    sigma_nd = -p_nd[:,None,None]*I + 2.0*eta_arr[:,None,None]*D  # (N,3,3)
    return sigma_nd

def cbf_surface_sigma_nn_prime(ugrid,
                               points_nd, sigma_nd_at_points,
                               top_percent=1.0, nz_cos_th=0.8,
                               k_sample=24):
    # 
    surf = ugrid.extract_surface().triangulate()
    surf = surf.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True)
    surf = surf.compute_cell_sizes(area=True)

    faces = surf.faces.reshape(-1, 4)[:, 1:]  # (Nc,3)
    p_s   = surf.points                        # (Ns,3)
    n_c   = surf['Normals']                    # (Nc,3)
    A_c   = surf['Area']                       # (Nc,)

    # 
    mask_n = n_c[:,2] > nz_cos_th
    faces = faces[mask_n]; n_c = n_c[mask_n]; A_c = A_c[mask_n]

    # 
    z_nodes = p_s[:,2]
    z_thr = np.percentile(z_nodes, 100.0 - top_percent)
    keep_vertex = z_nodes >= z_thr

    # 
    face_keep = np.all(keep_vertex[faces], axis=1)
    faces = faces[face_keep]; n_c = n_c[face_keep]; A_c = A_c[face_keep]

    # 
    Ns = p_s.shape[0]
    sigma_surf_pts = np.zeros((Ns,3,3), dtype=float)
    for i, ps in enumerate(p_s):
        # 
        if not keep_vertex[i]:
            continue
        d2 = np.sum((points_nd - ps)**2, axis=1)
        nn = np.argpartition(d2, min(k_sample, len(d2)-1))[:k_sample]
        w  = np.exp(-d2[nn] / (np.mean(d2[nn])+1e-30))
        w /= w.sum()
        sigma_surf_pts[i] = np.tensordot(w, sigma_nd_at_points[nn], axes=(0,0))

    # 
    Nc = faces.shape[0]
    sigma_face = np.zeros((Nc,3,3), dtype=float)
    for c in range(Nc):
        a,b,cid = faces[c]
        sigma_face[c] = (sigma_surf_pts[a] + sigma_surf_pts[b] + sigma_surf_pts[cid]) / 3.0
    s_face = np.einsum('ci,cij,cj->c', n_c, sigma_face, n_c)  # (Nc,)

    #
    M = lil_matrix((Ns, Ns), dtype=float)
    b = np.zeros(Ns, dtype=float)
    mpat = np.array([[2,1,1],[1,2,1],[1,1,2]], float) / 12.0

    for k, (a,bv,cid) in enumerate(faces):
        A = A_c[k]
        mloc = A * mpat
        idxs = (a,bv,cid)
        # assemble M
        for ii in range(3):
            for jj in range(3):
                M[idxs[ii], idxs[jj]] += mloc[ii,jj]
        # assemble b
        rhs_loc = (A/3.0) * s_face[k] * np.ones(3)
        for ii in range(3):
            b[idxs[ii]] += rhs_loc[ii]

    
    keep_idx = np.where(keep_vertex)[0]
    M_sub = M[keep_idx[:,None], keep_idx].tocsr()
    b_sub = b[keep_idx]
    s_sub = spsolve(M_sub, b_sub)

    # 
    s_nodal = np.full(Ns, np.nan, float)
    s_nodal[keep_idx] = -s_sub
    surf_kept = surf  # 
    return surf_kept, s_nodal, keep_vertex


# ----  ----
def cbf_topography_nondim(mesh, vel_nd, p_nd, vis_nd,
                          k_grad=24, lam=1e-8,
                          top_percent=1.0, nz_cos_th=0.8, k_sample=24):
   
    points_nd = mesh.points.astype(float)
    sigma_nd = compute_sigma_prime(points_nd, vel_nd, p_nd, vis_nd,
                                   k_grad=k_grad, lam=lam)
    surf, s_nn, keep_vertex = cbf_surface_sigma_nn_prime(
        mesh, points_nd, sigma_nd,
        top_percent=top_percent, nz_cos_th=nz_cos_th, k_sample=k_sample
    )
    
    return surf, s_nn, keep_vertex

# ---------- ----------
def strain_stress_topo_unstructured(points, vel, pressure, viscosity,
                                    k=24, rho=3300.0, g=9.81, top_percent=0.2):
    z = points[:, 2]
    u = vel[:, 0]; v = vel[:, 1]; w = vel[:, 2]
    N = points.shape[0]

    eta = (np.full(N, float(viscosity), float)
           if np.isscalar(viscosity) else np.asarray(viscosity, float))


    knn_idx = knn_indices_bruteforce(points, k=k)

    gu = weighted_least_squares_grad(points, u, knn_idx)  # (N,3) 
    gv = weighted_least_squares_grad(points, v, knn_idx)  # (N,3)
    gw = weighted_least_squares_grad(points, w, knn_idx)  # (N,3)


    G = np.stack([gu, gv, gw], axis=1)    # (N,3,3)

    D = 0.5 * (G + np.swapaxes(G, 1, 2))  # (N,3,3)


    I = np.eye(3)[None, :, :]              # (1,3,3)
    sigma = -pressure[:, None, None]*I + 2.0*eta[:, None, None]*D  # (N,3,3)

    # 7) 
    z_thr = np.percentile(z, 100.0 - top_percent)
    top_mask = z >= z_thr
    sigma_zz = sigma[:, 2, 2]
    
    # if type(rho)==float:
    #     h_top = sigma_zz[top_mask] / (rho * g)
    # else:
    #     h_top = sigma_zz[top_mask] / (rho[top_mask]  * g)
        
    h_top = -sigma_zz[top_mask]
    return D, sigma, sigma_zz, top_mask, h_top



fig,(ax) = plt.subplots(1,1,figsize=(8,8))

# fig2,(ax2) = plt.subplots(1,1,figsize=(8,8))
for uuu, model_number in enumerate(model_list):
    mesh = pv.read(path+'sinker'+str(model_number)+"_input.pvtu")
    mesh_pri = pv.read(path+'sinker'+str(model_number)+"_solution_primary.pvtu")
    mesh_sec = pv.read(path+'sinker'+str(model_number)+"_solution_secondary.pvtu")
    mesh_face1 = pv.read(path+'sinker'+str(model_number)+"_solution.face1.pvtu")
    
    
    temperature  = np.array(mesh.point_data.get('temperature'))
    vel = np.array(mesh_pri.point_data.get('velocity'))
    pressure = np.array(mesh_pri.point_data.get('pressure'))
    vis = np.array(mesh_sec.point_data.get('viscosity'))
    points = mesh.points            # 
    

    eta0 = np.median(vis)
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
    
    
    z_top = mesh.points[top_mask, 2]
    z_line = z_top[mask_y]
    
    
    idx = np.argsort(x_line)
    x_sorted = x_line[idx]
    h_sorted = h_line[idx]
    z_sorted = z_line[idx]
    
    tol = 1e-6
    cuts = np.where(np.diff(x_sorted) > tol)[0] + 1
    groups = np.split(np.arange(len(x_sorted)), cuts)
    

    x_keep = []
    h_keep = []
    for g in groups:
        j = g[np.argmax(z_sorted[g])]   
        x_keep.append(x_sorted[j])
        h_keep.append(h_sorted[j])
    
    x_line = np.array(x_keep)
    h_line = np.array(h_keep)
    
    ok = np.isfinite(h_line)
    x_line, h_line = x_line[ok], h_line[ok]
    h_line = h_line - np.mean(h_line)
    
    
    
    ax.scatter(x_line, h_line, color=colors_list[uuu+1],label=label_list[uuu])
    ax.plot(x_line, h_line, color=colors_list[uuu+1],lw=4)

    
    
    # get the x-range and z-level
    xmin, xmax = mesh_face1.points[:,0].min(), mesh_face1.points[:,0].max()
    zlevel = float(np.median(mesh_face1.points[:,2]))   
    
    # build a straight line along x at y=0.5
    npts = 20
    line = pv.Line(pointa=(xmin, 0.5, zlevel),
                   pointb=(xmax, 0.5, zlevel),
                   resolution=npts-1)
    
    sampled = mesh_face1.sample_over_line((xmin,0.5,zlevel), (xmax,0.5,zlevel), resolution=npts-1)
    
    x = sampled.points[:,0]
    stress = sampled.point_data['stress_norm']
    
    # sort and plot
    idx = np.argsort(x)
    x, stress = x[idx], stress[idx]    
    h_line_face = stress-np.mean(stress)
    ax.plot(x, -h_line_face,color=colors_list[uuu],label=label_list[uuu])
    

  
    if CBF: 
        vel_nd = vel
        p_nd = pressure
        vis_nd = eta_nd
        
        surf, hstar, keep_vertex = cbf_topography_nondim(
            mesh, vel_nd, p_nd, vis_nd,
            k_grad=32, lam=1e-8,
            top_percent=0.2,      
            nz_cos_th=0.2,        
            k_sample=32
        )
    
        # 
        y = surf.points[:,1]
        x = surf.points[:,0]
        surf['hstar'] = hstar 
        pline = surf.slice(normal=(0,1,0), origin=(0,0.5,0)) 
        x_line = pline.points[:, 0]
        h_line = pline.point_data['hstar']
    
        order = np.argsort(x_line)
        xline = x_line[order]; hline = h_line[order]
        ok = np.isfinite(hline)
        xline, hline = xline[ok], hline[ok]
        hline = hline-np.mean(hline)
        
        ax.scatter(xline, hline ,color=colors_list[11], s=20)
        ax.plot(xline, hline ,color=colors_list[11],lw=2,linestyle='dashed')

    
ax.set_xlabel("x",fontsize=fontsize)
ax.set_ylabel("Dynamic topography h ",fontsize=fontsize)
ax.set_title("h(x) at y = 0.5",fontsize=fontsize)
# ax.legend(title=label_title,fontsize = fontsize,title_fontsize=fontsize)
ax.grid(True)
# ax.set_xlim(0,1)
# ax.set_ylim(-1,1)

for aaa in [ax]:
    aaa.tick_params(which='minor', length=5, width=2, direction='in')
    aaa.tick_params(labelsize=fontsize,width=bwidth,length=10,right=True, top=True,direction='in',pad=15)
    for axis in ['top','bottom','left','right']:
        aaa.spines[axis].set_linewidth(bwidth)