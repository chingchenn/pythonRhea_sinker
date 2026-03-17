# pythonRhea_sinker
Rhea sinker example

Exploratory scripts numbered in chronological order, documenting the research process.
For formal analysis, use the modular code in the parent directory.

---

## Series 01 — Field Visualization

| File | Description |
|------|-------------|
| `01_plot_velocity.py` | sinker model, temperature + velocity, includes x-y plane slice |
| `01_plot_velocity_temperature.py` | slab/spb model, single y-slice, temperature + velocity |
| `01_plot_velocity_temperature_slices.py` | sinker model, 4 y-slice temperature overview |
| `01_plot_velocity_temperature_slices_zoomin.py` | same as above with xlim/ylim zoom |
| `01_plot_velocity_temperature_box_sp_car.py` | spb_ spherical model, Cartesian + Polar dual view, temperature |
| `01_plot_velocity_viscosity.py` | slab/spb model, single y-slice, viscosity + velocity |
| `01_plot_velocity_viscosity_slices.py` | sinker model, 4 y-slice viscosity overview |
| `01_plot_velocity_viscosity_slices_zoomin.py` | same as above with xlim/ylim zoom |
| `01_plot_velocity_viscosity_box_sp_car.py` | spb_ spherical model, Cartesian + Polar dual view, viscosity |

---

## Series 02 — Topography Calculation and Validation

| File | Description |
|------|-------------|
| `02_check_topo.py` | sinker surface topography visualization and basic check |
| `02_topography.py` | sinker multi-model, topography via sigma_zz with kNN gradient method |
| `02_topography_CBF.py` | sinker multi-model, **CBF (Consistent Boundary Flux)** algorithm, compared against stress_norm |
| `02_topography_compare_models.py` | spb_ multi-model, topography comparison directly from stress_norm |
| `02_topography_example.py` | sinker, topography via stress_diag - pressure |
| `02_topography_find_creat.py` | sinker, stress_norm algorithm, confirmed and saved to file |

---

## Series 03 — Gravity

| File | Description |
|------|-------------|
| `03_gravity_profile.py` | sinker model, free-air gravity anomaly via **3D point-mass volume integration** |

---

## Series 04 — Velocity Validation

| File | Description |
|------|-------------|
| `04_check_uz.py` | sinker multi-model, surface vertical velocity misfit vs viscosity contrast |
| `04_check_uz_profile.py` | sinker multi-model, vertical velocity profile at y=0.5 |

---

## Series 05 — Mesh Basics

| File | Description |
|------|-------------|
| `05_elements_nodes.py` | Compute element and node counts for different refinement levels and polynomial orders |

---

## Series 06 — Benchmark Error

| File | Description |
|------|-------------|
| `06_check_topo_misfit.py` | sinker multi-model, topography misfit vs viscosity contrast, solver convergence validation |

---

## Series 07 — Temperature Field Concept

| File | Description |
|------|-------------|
| `07_temperature_2dfield.py` | 2D subduction zone temperature field proof-of-concept, half-space cooling overlaid on slab geometry |
