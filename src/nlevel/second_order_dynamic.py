from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from dynamic import potential_by_level
from old.dynamic_exactly import structure_size
from plot_nlevels import print_structure

import math
from scipy.optimize import brentq

def solve_thetaN_brent(fL, k_N):
    f = lambda th: 8.0*k_N*th - 0.5*fL*((math.cos(th/2)-math.sin(th/2))/(math.cos(th/2)+math.sin(th/2)))
    return brentq(f, 1e-16, math.pi/2)

def compute_r_s_vectors(N, lambda_restriction, k_i, B_fn=None):
    """
    Calcula r_j y s_j (j=1..N) usando la recurrencia:
        r_{j+1} = alpha_j r_j + b_j s_j
        s_{j+1} = s_j + r_{j+1}

    Convención Python:
        r[0]=r1, ..., r[N-1]=rN
        s[0]=s1, ..., s[N-1]=sN

    Args:
        N (int): número de niveles
        lambda_restriction (float): lambda = f*L (constante en la recurrencia)
        k_i (array-like): k_1..k_N  -> en Python k_i[0]=k1, ..., k_i[N-1]=kN
        B_fn (callable): función B_fn(j_math) que devuelve B_j con j_math=1..N-1
                         Si es None, uso tu forma actual: B_j = 2 * 4**(N - (j_math))
                         (OJO: revisa si tu paper usa N-j_math o N-j_math+1)

    Returns:
        r, s (np.ndarray, np.ndarray)
    """
    k_i = np.asarray(k_i, dtype=float)
    if k_i.shape[0] != N:
        raise ValueError(f"k_i debe tener largo N={N}, pero tiene {k_i.shape[0]}")

    if B_fn is None:
        # Tu forma actual pero usando j_math explícito (1..N-1)
        B_fn = lambda j_math: 2.0 * (4.0 ** (N - j_math))

    r = np.zeros(N, dtype=float)
    s = np.zeros(N, dtype=float)

    # Condiciones iniciales (según lo que vienes usando)
    r[0] = 1.0   # r1
    s[0] = 1.0   # s1 (si en tu derivación s1=r1, entonces 1.0 está bien)

    C = lambda_restriction / 2.0

    # j_idx = 0..N-2  <->  j_math = j_idx+1 (o sea 1..N-1)
    for j_idx in range(N - 1):
        j_math = j_idx + 1

        B_j = B_fn(j_math)

        # En tu fórmula estás usando k_j y k_{j+1}
        k_j   = k_i[j_idx]       # k_{j}
        k_j1  = k_i[j_idx + 1]   # k_{j+1}

        denom = (B_j * k_j1 + C)
        b_j = (B_j * k_j1) / denom
        alpha_j = (4.0 * B_j * k_j + C) / denom

        r[j_idx + 1] = alpha_j * r[j_idx] + b_j * s[j_idx]
        s[j_idx + 1] = s[j_idx] + r[j_idx + 1]

    return r, s

def compute_S_Q_Ak(N, k_i, r, prefactor_fn=None):
    """
    Calcula:
      S = sum_i r_i
      Q = sum_i r_i^2
      Ak = sum_i prefactor(i) * k_i * (r_i^2 + s_{i-1}^2)

    Convención Python:
      r[0]=r1, ..., r[N-1]=rN
      k_i[0]=k1, ..., k_i[N-1]=kN
      s_{i-1} = r1 + ... + r_{i-1}  -> en código: s_prev antes de sumar r[i]

    Args:
      prefactor_fn(i_idx): devuelve el prefactor del término i (i_idx=0..N-1).
        Si None, usa 4**(N - i_idx + 1)  (AJUSTA si tu paper usa otra potencia)

    Returns:
      S, Q, Ak (floats)
    """
    r = np.asarray(r, dtype=float)
    k_i = np.asarray(k_i, dtype=float)

    if len(r) != N or len(k_i) != N:
        raise ValueError("r y k_i deben tener largo N")

    # S y Q
    S = float(np.sum(r))
    Q = float(np.sum(r**2))

    if prefactor_fn is None:
        # OJO: este prefactor depende de tu definición en el paper.
        # Este es el más típico si tienes algo como 4^{N-i+1}.
        prefactor_fn = lambda i_idx: 4.0 ** (N - (i_idx + 1) + 1)  # = 4**(N-i_idx)

    Ak = 0.0
    s_prev = 0.0  # s_{0} = 0 si interpretas s_{i-1} como suma previa

    for i_idx in range(N):
        pref = prefactor_fn(i_idx)
        Ak += pref * k_i[i_idx] * (r[i_idx]**2 + s_prev**2)
        s_prev += r[i_idx]

    return S, Q, Ak

def theta_second_order_calculation(N, lambda_restriction, k_i):
    r, s = compute_r_s_vectors(N, lambda_restriction, k_i)
    S, Q, Ak = compute_S_Q_Ak(N, k_i, r)
    #theta_1 = (1/r[-1])  * ((lambda_restriction + 16 * k_i[-1]) -
    #            np.sqrt( (lambda_restriction + 16 * k_i[-1])**2 - 2 * (lambda_restriction)**2 ) ) / (lambda_restriction)
    theta_N = solve_thetaN_brent(lambda_restriction, k_i[-1])
    theta_1 = (1/r[-1]) * theta_N
    thetas = r * theta_1
    return thetas, S, Q, Ak, theta_1, theta_N

def example_animation():
    n = 4  # Número de ángulos
    k_i = [1, 1, 1, 1]  # Constantes elásticas para cada nivel
    # k_i = []
    # for i in range(n):
    #     k_i.append(4**(n-i-1) * (n-i)**2)
    k_i = k_i/np.max(k_i)
    print("k_i:", k_i)
    
    # Rango de valores para lambda_restriction
    lambda_values = np.logspace(-3, 3.0, 60) 

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 5),
        gridspec_kw={'width_ratios': [5.2, 4.8]}
    )

    def update(frame):
        lambda_restriction = lambda_values[frame]
        thetas, S, Q, Ak, theta_1, theta_N= theta_second_order_calculation(n, lambda_restriction, k_i)
        energy_by_levels = potential_by_level(np.asarray(thetas, dtype=float), np.asarray(k_i, dtype=float))
        energy_by_levels = energy_by_levels / np.sum(energy_by_levels)
        ax1.clear()
        structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min = print_structure(thetas, ax1, square_size=1.0/2**n)
        ax1.set_title(f"Lambda = {lambda_restriction:.2e}\nLx = {structure_size_x_max:.2f} \nArea = {structure_size_x_max * structure_size_y_max:.2f}")
        ax2.clear()
        x_vals = np.arange(1, len(thetas)+1)
        y_vals = energy_by_levels 
        ax2.plot(x_vals, y_vals, 'o')
        ax2.set_title(r"$E_i / E$", fontsize=14)
        ax2.set_xticks(x_vals)
        ax2.set_xlabel("i")
        ax2.set_ylim([0, 1])
        for i, y in zip(x_vals, y_vals):
            ax2.text(i, y, f"{y:.2f}", fontsize=9, ha='left', va='bottom')

    anim = FuncAnimation(fig, update, frames=len(lambda_values), interval=30)
    plt.show()

def example_animation_energy():
    n = 16  # Número de ángulos
    k_i = []
    for i in range(n):
        k_i.append(1)
        #k_i.append(2**(-i))
    k_i = k_i/np.max(k_i) 

    print("k_i:", k_i)
    
    # Rango de valores para lambda_restriction
    lambda_values = np.logspace(-4, 12.0, 60) 
    fig, ax = plt.subplots(figsize=(10, 4.8))

    def update(frame):
        lambda_restriction = lambda_values[frame]
        thetas, S, Q, Ak, theta_1, theta_N= theta_second_order_calculation(n, lambda_restriction, k_i)
        structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min = structure_size(thetas)
        force = lambda_restriction / structure_size_x_max
        energy_by_levels = potential_by_level(np.asarray(thetas, dtype=float), np.asarray(k_i, dtype=float))
        energy_by_levels = energy_by_levels 
        ax.clear()
        x_vals = np.arange(1, len(thetas) + 1)
        y_vals = energy_by_levels/ np.sum(energy_by_levels)
        ax.plot(x_vals, y_vals, 'o', label=f'f={force:.2e}')
        ax.set_title(r"$E_i / E$", fontsize=14)
        ax.set_xticks(x_vals)
        ax.set_xlabel("i")
        ax.set_ylim([0, 1])
        ax.legend(loc='upper left')
        for i, y in zip(x_vals, y_vals):
            ax.text(i, y, f"{y:.2f}", fontsize=9, ha='left', va='bottom')

    anim = FuncAnimation(fig, update, frames=len(lambda_values), interval=100)
    #writer = PillowWriter(fps=10)
    #anim.save("energy_spectrum_2D.gif", writer=writer, dpi=250)
    plt.show()

if __name__ == "__main__":
    #example_animation_energy()
    example_animation()