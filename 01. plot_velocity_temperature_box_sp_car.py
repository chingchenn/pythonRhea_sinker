#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 01:30:09 2026

@author: chingchen
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------- 使用者參數 ----------------
model_number = 'spb_0610'
FIELD = 'temperature'  # 您可以視需要改為 'viscosity'
phi0_deg = 0.0
nx, nz = 301, 301
step = 10
fontsize = 12
scale=10
# 物理常數 (來源：rhea_temperature.c 與 rhea_domain.c)
kappa = 1.0e-6      # thermal-diffusivity [m^2/s]
R_earth = 6371.0e3  # earth-radius [m]
sec_per_yr = 31557600.0 # seconds in a year

# 速度轉換因子：從非維度 (Non-dimensional) 轉為 cm/yr
# 若您的 VTK 已經是維度化數值，請將此處改為 100.0 (m/yr -> cm/yr)
v_scaling = (kappa / R_earth) * 100.0 * sec_per_yr 

# 參考箭頭 (cm/yr)
vref_val = 0.5

# ---------------- 讀取資料 ----------------
mesh = pv.read(f'/Users/chingchen/Desktop/Rhea/rhea_model/{model_number}_input.pvtu')
# mesh_vel = pv.read(f'/Users/chingchen/Desktop/Rhea/rhea_model/{model_number}_solution_primary.pvtu')


phi0 = np.deg2rad(phi0_deg)
xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

x_lin = np.linspace(xmin, xmax, nx)
z_lin = np.linspace(zmin, zmax, nz)
XX, ZZ = np.meshgrid(x_lin, z_lin)

YY = XX * np.tan(phi0)
mask_in = (YY >= ymin) & (YY <= ymax)

pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
cloud = pv.PolyData(pts)

# ---- 採樣數據場 ----
sampF = cloud.sample(mesh)
F_data = np.asarray(sampF.point_data[FIELD]).reshape(nz, nx)
F_data[~mask_in] = np.nan

# ---- 採樣速度並修正單位 (cm/yr) ----
# vel_cloud = cloud.sample(mesh_vel)
# V = np.asarray(vel_cloud.point_data['velocity']).reshape(nz, nx, 3)

# 修正：使用物理縮放因子
# Ux_u = V[:, :, 0] * v_scaling
# Uz_u = V[:, :, 2] * v_scaling 

# ----------------- 圖 1: 笛卡爾座標 (X-Z 扇形圖) -----------------
fig1, ax1 = plt.subplots(figsize=(10, 6))

# 使用 pcolormesh 以符合球形箱體 (Box Spherical) 的幾何映射
im1 = ax1.pcolormesh(XX, ZZ, F_data, shading='auto', cmap='jet')

# 繪製 Quiver
xq, zq = XX[::step, ::step], ZZ[::step, ::step]
# uq, wq = Ux_u[::step, ::step], Uz_u[::step, ::step]
# m = np.isfinite(F_data[::step, ::step]) & np.isfinite(uq)

# q1 = ax1.quiver(xq[m], zq[m], uq[m], wq[m], color='k', scale=100)
# ax1.quiverkey(q1, X=0.85, Y=1.05, U=vref_val, label=f'{vref_val} cm/yr', labelpos='E', coordinates='axes')

ax1.set_aspect('equal')
ax1.set_title(f'{FIELD} & Velocity (cm/yr) - Cartesian', fontsize=fontsize+4)

# 讓 Colorbar 高度與主圖對齊
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="3%", pad=0.1)
fig1.colorbar(im1, cax=cax, label=FIELD)

# ----------------- 圖 2: 極座標 (r-theta 拉直圖) -----------------
grid_r = np.sqrt(XX**2 + ZZ**2) # 計算半徑
grid_theta = np.arctan2(XX, ZZ) # 計算角度

# 修正：極座標向量旋轉 (將 vx, vz 投影至 vr, vtheta)

# V_theta = Ux_u * np.cos(grid_theta) - Uz_u * np.sin(grid_theta)
# V_r = Ux_u * np.sin(grid_theta) + Uz_u * np.cos(grid_theta)

fig2, ax2 = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
pcm2 = ax2.pcolormesh(grid_theta, grid_r, F_data, shading='auto', cmap='jet')

ax2.set_theta_zero_location("N") # 0度朝北
ax2.set_theta_direction(-1)      # 順時針增加
ax2.set_thetamin(-45)
ax2.set_thetamax(45)
ax2.set_rlim(0.0, 1.05)         # 配合 Rhea 的 radius 範圍

# 繪製極坐標箭頭
th_q, r_q = grid_theta[::step, ::step], grid_r[::step, ::step]
# vth_q, vr_q = V_theta[::step, ::step], V_r[::step, ::step]
# m_p = np.isfinite(F_data[::step, ::step]) & np.isfinite(vth_q)

# 在 Matplotlib 極坐標 Quiver 中，方向是 (d_theta, d_r)
# q2 = ax2.quiver(th_q[m_p], r_q[m_p], vth_q[m_p], vr_q[m_p], color='k', scale=scale)
# ax2.quiverkey(q2, X=0.85, Y=0.95, U=vref_val, label=f'{vref_val} cm/yr', labelpos='E', coordinates='axes')

ax2.set_title(f'{FIELD} & Velocity - Polar', fontsize=fontsize+4)
fig2.colorbar(pcm2, ax=ax2, pad=0.1, shrink=0.6, label=FIELD)

plt.show()