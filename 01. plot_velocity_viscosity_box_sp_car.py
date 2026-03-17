#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 02:03:49 2026

@author: chingchen
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------- 使用者參數 ----------------
model_number = 'spb_0610'
FIELD = 'viscosity'  # 改為 viscosity
phi0_deg = 0.0
nx, nz = 3501, 3501
step = 60
scale=5000
fontsize = 12
vel_plot = 1

# 物理常數轉換因子 (從非維度轉為 cm/yr)
kappa = 1.0e-6
R_earth = 6371.0e3
sec_per_yr = 31557600.0
# 如果您的 VTK 已經是 m/yr，請設為 100.0；如果是原始非維度，請用下面的公式
v_scaling = (kappa / R_earth) * 100.0 * sec_per_yr 

# 參考箭頭
vref_val = 0.5 # cm/yr

# --------------------------------------------

mesh = pv.read(f'/Users/chingchen/Desktop/Rhea/rhea_model/{model_number}_input.pvtu')

phi0 = np.deg2rad(phi0_deg)
xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

x_lin = np.linspace(xmin, xmax, nx)
z_lin = np.linspace(zmin, zmax, nz)
XX, ZZ = np.meshgrid(x_lin, z_lin)

YY = XX * np.tan(phi0)
mask_in = (YY >= ymin) & (YY <= ymax)

pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
cloud = pv.PolyData(pts)

# ---- 採樣黏滯係數 ----
sampVisc = cloud.sample(mesh)
# 確保使用 LogNorm 處理黏度
Visc = np.asarray(sampVisc.point_data[FIELD]).reshape(nz, nx)
Visc[~mask_in] = np.nan

# ---- 採樣速度並轉換單位 ----
if vel_plot:
    mesh_vel = pv.read(f'/Users/chingchen/Desktop/Rhea/rhea_model/{model_number}_solution_primary.pvtu')
    vel_cloud = cloud.sample(mesh_vel)
    V = np.asarray(vel_cloud.point_data['velocity']).reshape(nz, nx, 3)
    
    # 轉換為 cm/yr
    Ux_u = V[:, :, 0] * v_scaling
    Uz_u = V[:, :, 2] * v_scaling
    


# ----------------- 圖 1: 笛卡爾座標 (扇形圖) -----------------
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))

# 使用 pcolormesh 以獲得更準確的幾何形狀，並套用 LogNorm
im1 = ax1.pcolormesh(
    XX, ZZ, Visc, 
    shading='auto', 
    cmap='Blues_r', 
    norm=LogNorm(vmin=1e18, vmax=1e25)
)

ax1.set_aspect('equal')
ax1.set_title(f'{FIELD} & Velocity (cm/yr) - Cartesian', fontsize=fontsize+4)

# 箭頭與參考標籤
if vel_plot:
    xq, zq = XX[::step, ::step], ZZ[::step, ::step]
    uq, wq = Ux_u[::step, ::step], Uz_u[::step, ::step]
    m = np.isfinite(Visc[::step, ::step]) & np.isfinite(uq) & np.isfinite(wq)
    
    q1 = ax1.quiver(xq[m], zq[m], uq[m], wq[m], color='k', scale=scale)
    ax1.quiverkey(q1, X=0.85, Y=1.05, U=vref_val, label=f'{vref_val} cm/yr', labelpos='E', coordinates='axes')

# 讓 Colorbar 高度與主圖對齊
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="3%", pad=0.1)
fig1.colorbar(im1, cax=cax, label='Viscosity [Pa s]')
ax1.set_xlim(0.2,0.5)
ax1.set_ylim(0.8,1)
# ----------------- 圖 2: 極座標 (r-theta) -----------------
r = np.sqrt(XX**2 + ZZ**2)
theta = np.arctan2(XX, ZZ)

if vel_plot:
# 旋轉速度向量到極座標系 (Vr, Vtheta)

    Uth_u = Ux_u * np.cos(theta) - Uz_u * np.sin(theta)
    Ur_u  = Ux_u * np.sin(theta) + Uz_u * np.cos(theta)

fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))

pcm = ax2.pcolormesh(theta, r, Visc, shading='auto', cmap='Blues', norm=LogNorm(vmin=1e18, vmax=1e25))

ax2.set_theta_zero_location("N")
ax2.set_theta_direction(-1)
ax2.set_thetamin(-45)
ax2.set_thetamax(45)
ax2.set_rlim(0.0, 1.0) # 根據您的 Z 範圍設定
ax2.set_title(f'{FIELD} & Velocity - Polar', fontsize=fontsize+4)


if vel_plot:
# 極座標箭頭
    th_q, r_q = theta[::step, ::step], r[::step, ::step]
    uth_q, ur_q = Uth_u[::step, ::step], Ur_u[::step, ::step]
    m_p = np.isfinite(Visc[::step, ::step]) & np.isfinite(uth_q)
    
    q2 = ax2.quiver(th_q[m_p], r_q[m_p], uth_q[m_p], ur_q[m_p], scale=scale)
    ax2.quiverkey(q2, X=0.85, Y=0.95, U=vref_val, label=f'{vref_val} cm/yr', labelpos='E', coordinates='axes')

# 極座標 Colorbar 手動縮放高度
fig2.colorbar(pcm, ax=ax2, pad=0.1, shrink=0.6, label='Viscosity [Pa s]')

print(f'{model_number}',np.max(Visc))
print(np.min(Visc)/1e18, ' of', 1e18)

plt.show()