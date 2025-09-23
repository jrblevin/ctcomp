#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from model import Model
from benchmarks.plot_style import setup_plot_style, get_line_styles

# Set up the plot style
setup_plot_style()

# Model parameters
param = {
    'theta_ec': -0.5,
    'theta_rn': -0.2,
    'theta_d': 0.3,
    'lambda': 1.0,
    'gamma': 0.5,
    'rho': 0.05,
}

# Configuration
config = {
    'vf_algorithm': 'polyalgorithm',
    'vf_max_iter': 5000,
    'vf_tol': 1e-13,
    'vf_rtol': 0.01,
}

# Create model
model = Model(n_players=3, n_demand=3, param=param, config=config)

# Run polyalgorithm with tracking
v_poly = model.polyalgorithm(
    vf_max_iter=config['vf_max_iter'],
    vf_tol=config['vf_tol'],
    vf_rtol=config['vf_rtol'],
    vf_max_newton_iter=10,
    vf_newton_solver='direct',
    track_convergence=True
)

# Get tracking data
poly_residuals = model._convergence_history['residuals']
switch_iter = model._convergence_history['switch_iteration']

# Run pure value iteration - manual implementation with tracking
vi_residuals = []
v_vi = np.zeros((model.n_players, model.K))

for i in range(5000):
    v_old = v_vi.copy()
    v_vi = model.bellman_operator(v_vi)
    diff = np.max(np.abs(v_vi - v_old))
    vi_residuals.append(diff)
    if diff < config['vf_tol']:
        break

# Calculate theoretical contraction factor for rate analysis
unifrate = param['lambda'] * model.n_players + 2*param['gamma']
beta = unifrate / (model.rho + unifrate)

# Calculate convergence rates for polyalgorithm
poly_rates = []
for i in range(1, len(poly_residuals)):
    rate = poly_residuals[i] / poly_residuals[i-1] if poly_residuals[i-1] > 1e-16 else np.inf
    poly_rates.append(rate)

vi_iterations = np.arange(1, len(vi_residuals) + 1)
poly_iterations = np.arange(1, len(poly_residuals) + 1)
colors, markers, linestyles = get_line_styles(3)

# Figure 1: Full convergence comparison
plt.figure(figsize=(8, 6))
plt.semilogy(poly_iterations, poly_residuals, color=colors[0], linestyle=linestyles[0],
             linewidth=2, label='Polyalgorithm')
plt.semilogy(vi_iterations, vi_residuals, color=colors[1], linestyle=linestyles[1],
             linewidth=2, label='Value Iteration')
plt.axhline(y=config['vf_tol'], color=colors[2], linestyle=':', linewidth=2, alpha=0.4,
            label='Convergence threshold')
# if switch_iter:
#     plt.axvline(x=switch_iter, color=colors[1], linestyle='--', alpha=0.7,
#                 label='Switch to N-K')
plt.xlabel('Iteration')
# Residual: $\| V^{(n)} - T(V^{(n)}) \|_{\infty}$
plt.ylabel(r'Residual (log scale)')
plt.legend()
plt.grid(True, alpha=0.3)
max_iter = max(len(vi_residuals), len(poly_residuals))
plt.xlim(1, max_iter)
plt.ylim(1e-16, 10)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig('vf_algorithms.pdf', bbox_inches='tight')
plt.close()

# Figure 2: Zoom on first 50 iterations
plt.figure(figsize=(8, 6))
zoom_limit = min(20, max(len(vi_residuals), len(poly_residuals)))
vi_zoom = vi_iterations[:min(zoom_limit, len(vi_iterations))]
poly_zoom = poly_iterations[:min(zoom_limit, len(poly_iterations))]
plt.semilogy(poly_zoom, poly_residuals[:len(poly_zoom)], color=colors[0], linestyle=linestyles[0],
             linewidth=2, label='Polyalgorithm')
plt.semilogy(vi_zoom, vi_residuals[:len(vi_zoom)], color=colors[1], linestyle=linestyles[1],
             linewidth=2, label='Value Iteration')
plt.axhline(y=config['vf_tol'], color=colors[2], linestyle=':', linewidth=2, alpha=0.4,
            label='Convergence threshold')
# if switch_iter and switch_iter <= zoom_limit:
#     plt.axvline(x=switch_iter, color=colors[1], linestyle='--', alpha=0.7,
#                 label='Switch to N-K')
plt.xlabel('Iteration')
# Residual: $\| V^{(n)} - T(V^{(n)}) \|_{\infty}$
plt.ylabel(r'Residual (log scale)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(1, zoom_limit)
plt.ylim(1e-16, 10)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig('vf_algorithms_zoom.pdf', bbox_inches='tight')
plt.close()

# Figure 3: Convergence rate analysis
plt.figure(figsize=(8, 6))
rate_iterations = np.arange(2, len(poly_residuals) + 1)
rates_display = np.array(poly_rates)
rates_display[rates_display > 2] = np.nan
plt.plot(rate_iterations, rates_display, color=colors[0], linestyle=linestyles[0],
         linewidth=2, label='Convergence rate')
plt.axhline(y=beta, color=colors[2], linestyle=linestyles[1], linewidth=2, alpha=0.4,
            label='Target rate (β)')
plt.fill_between(poly_iterations, beta - config['vf_rtol'], beta,
                 alpha=0.4, color=colors[2], label='Switch zone')
# if switch_iter:
#     plt.axvline(x=switch_iter, color=colors[1], linestyle='--', alpha=0.7,
#                 label='Switch to N-K')
plt.xlabel('Iteration')
# Convergence rate: $\|V^{(n)} - V^{(n-1)}\|_{\infty} / \|V^{(n-1)} - V^{(n-2)}\|_{\infty}$
plt.ylabel(r'Convergence Rate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, min(1.2, np.nanmax(rates_display) * 1.1))
plt.xlim(poly_iterations[0], poly_iterations[-1])
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig('vf_rate_analysis.pdf', bbox_inches='tight')
plt.close()

# Show detailed comparison
vi_final_iter = len(vi_residuals)
poly_final_iter = len(poly_residuals)
speedup = vi_final_iter / poly_final_iter if poly_final_iter > 0 else 1

print('Value Iteration:')
print(f"- Number of iterations: {vi_final_iter}")
print(f"- Final residual: {vi_residuals[-1]:.2e}")

print('\nPolyalgorithm:')
print(f"- Switch at iteration: {switch_iter}")
print(f"- Number of iterations: {poly_final_iter}")
print(f"- Number of Newton iterations: {poly_final_iter - switch_iter}")
print(f"- Final residual: {poly_residuals[-1]:.2e}")
print(f"- Speedup: {speedup:.1f}x")

# Check solution accuracy
v_diff = np.max(np.abs(v_vi - v_poly))
print('\nSolution accuracy:')
print(f"- Max |v_vi - v_poly|: {v_diff:.2e}")

# Print detailed rate analysis
switch_rate = poly_rates[switch_iter-2] if switch_iter > 1 else poly_rates[0]
switch_diff = abs(switch_rate - beta)
print('\nRate Analysis:')
print(f"- Theoretical beta: {beta:.6f}")
print(f"- Switch threshold: {config['vf_rtol']}")
print(f"- Switch iteration: {switch_iter}")
print(f"- Rate at switch: {switch_rate:.6f}")
print(f"- |rate - beta| at switch: {switch_diff:.6f}")

# Print detailed iteration table
print()
print(f"{'Iter':<4} {'Residual':<12} {'Rate':<10} {'Phase':<15}")
print('----------------------------------------------')

for i in range(len(poly_residuals)):
    iter_num = i + 1
    residual = poly_residuals[i]

    # Calculate rate
    if i > 0:
        rate = residual / poly_residuals[i-1]
        rate_str = f"{rate:.6f}" if rate < 100 else f"{rate:.2e}"
    else:
        rate_str = "N/A"

    # Determine phase
    if switch_iter and iter_num <= switch_iter:
        phase = "Value Iteration"
    else:
        phase = "Newton"

    # Mark switch point
    marker = " *" if iter_num == switch_iter else ""

    print(f"{iter_num:<4} {residual:<12.2e} {rate_str:<10} {phase:<15}{marker}")

print("\nThree plots saved:")
print("  vf_algorithms.pdf - Full convergence comparison")
print("  vf_algorithms_zoom.pdf - Initial iterations zoom")
print("  vf_rate_analysis.pdf - Convergence rate analysis")
