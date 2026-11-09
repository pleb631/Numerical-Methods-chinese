# 本文件为 plot_partial_differential_equations.py 的中文注释版本（代码逻辑与英文版完全一致）
import numpy as np
import matplotlib.pyplot as plt
from plot_style import finish

x = np.linspace(0, 1, 500)

# 1. 热方程：使用正弦模式的类高斯傅里叶兼容轮廓
def heat_profile(t):
    return (
        np.exp(-np.pi**2*t)*np.sin(np.pi*x)
        + 0.45*np.exp(-9*np.pi**2*t)*np.sin(3*np.pi*x)
        + 0.22*np.exp(-25*np.pi**2*t)*np.sin(5*np.pi*x)
    )

fig, ax = plt.subplots(figsize=(7, 4.3))
for t in [0.0, 0.01, 0.04, 0.12]:
    ax.plot(x, heat_profile(t), label=f"$t={t:g}$")
ax.set_xlabel("$x$")
ax.set_ylabel("$u(x,t)$")
ax.set_title("Heat equation: high-frequency structure decays fastest")
ax.grid(alpha=0.25)
ax.legend()
finish(fig, "pde_heat_diffusion.svg")

# 2. 波动方程：固定端模态叠加
def wave_profile(t):
    return (
        np.cos(np.pi*t)*np.sin(np.pi*x)
        + 0.35*np.cos(3*np.pi*t)*np.sin(3*np.pi*x)
    )

fig, ax = plt.subplots(figsize=(7, 4.3))
for t in [0.0, 0.15, 0.30, 0.45]:
    ax.plot(x, wave_profile(t), label=f"$t={t:g}$")
ax.set_xlabel("$x$")
ax.set_ylabel("$u(x,t)$")
ax.set_title("Wave equation: modes oscillate instead of diffusing away")
ax.grid(alpha=0.25)
ax.legend()
finish(fig, "pde_wave_propagation.svg")
