# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 16:03:33 2025

@author: kaanc
"""

import math
import matplotlib.pyplot as plt

# Constants
rho = 1000      # Water density (kg/m^3)
mu = 0.001      # Dynamic viscosity (Pa·s)
g = 9.81        # Gravity (m/s^2)

# Initial conditions
v_initial = 4.0           # Initial velocity (m/s)
flow_depth = 0.8          # Flow depth (m)
obstacle_diameter = 0.4   # Characteristic length (m)

# Turbulence loss coefficient per layer (K values) → 4 layers
K_layers = [0.5, 0.5, 0.5, 0.5]
# Türbülans kaynaklı ek kayıp katsayısı per layer (örnek değerler)
beta_layers = [0.2, 0.3, 0.4, 0.5]  # Example beta values for each layer


K_control = [0.0, 0.0, 0.0, 0.0]  # No-obstacle scenario
beta_control = [0.0, 0.0, 0.0, 0.0]  # No turbulence loss control
layer_count = len(K_layers)

def simulate_layers_with_turbulence(K_layers, beta_layers):
    velocities = [v_initial]
    reynolds_numbers = []
    froude_numbers = []
    total_energy_loss = 0
    v = v_initial

    for i, K in enumerate(K_layers):
        # Reynolds and Froude before losses
        Re = (rho * v * obstacle_diameter) / mu
        Fr = v / math.sqrt(g * flow_depth)
        reynolds_numbers.append(Re)
        froude_numbers.append(Fr)

        # Engel kaybı (minor loss)
        v_after_obstacle_squared = v**2 * (1 - K)
        v_after_obstacle = math.sqrt(v_after_obstacle_squared) if v_after_obstacle_squared > 0 else 0

        # Türbülans kaybı (ek kayıp)
        beta = beta_layers[i]
        v_after_turbulence_squared = v_after_obstacle**2 * (1 - beta)
        v_after_turbulence = math.sqrt(v_after_turbulence_squared) if v_after_turbulence_squared > 0 else 0

        # Enerji kaybı bu katman için (önce + sonra hız farkı)
        delta_E = 0.5 * rho * (v**2 - v_after_turbulence**2)
        total_energy_loss += delta_E

        velocities.append(v_after_turbulence)
        v = v_after_turbulence

    # Add last layer's Re and Fr (final velocity)
    reynolds_numbers.append((rho * v * obstacle_diameter) / mu)
    froude_numbers.append(v / math.sqrt(g * flow_depth))

    return velocities, reynolds_numbers, froude_numbers, total_energy_loss

# Run both test and control
velocities_test, Re_test, Fr_test, E_loss_test = simulate_layers_with_turbulence(K_layers, beta_layers)
velocities_control, Re_control, Fr_control, E_loss_control = simulate_layers_with_turbulence(K_control, beta_control)

# Plotting velocity drop
layer_labels = [f"Layer {i}" for i in range(layer_count + 1)]
x = list(range(len(layer_labels)))

plt.figure(figsize=(12, 6))
plt.plot(x, velocities_test, marker='o', label='Test Velocity (with obstacles + turbulence)', color='darkblue')
plt.plot(x, velocities_control, marker='o', linestyle='--', label='Control Velocity (no obstacles)', color='gray')
plt.xticks(x, layer_labels)
plt.title("Velocity Reduction Across 4 Obstacle Layers (with Turbulence Loss)")
plt.xlabel("Layer")
plt.ylabel("Velocity (m/s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Labeling Re and Fr values
for i, label in enumerate(layer_labels):
    print(f"{label}:")
    print(f"  Reynolds (Test): {Re_test[i]:,.0f}")
    print(f"  Froude   (Test): {Fr_test[i]:.2f}")
    print(f"  Reynolds (Control): {Re_control[i]:,.0f}")
    print(f"  Froude   (Control): {Fr_control[i]:.2f}")
    print()

# Summary
print(f"Total Energy Loss (Test, with obstacles + turbulence): {E_loss_test:.2f} J/kg")
print(f"Total Energy Loss (Control, no obstacles): {E_loss_control:.2f} J/kg")
print(f"Final Velocity (Test): {velocities_test[-1]:.2f} m/s")
print(f"Final Velocity (Control): {velocities_control[-1]:.2f} m/s")