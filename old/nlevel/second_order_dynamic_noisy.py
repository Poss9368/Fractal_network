from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from old.nlevel.plot_nlevels import print_structure
import math
from scipy.optimize import brentq

def potential_by_level(thetas: np.ndarray, k_i: np.ndarray | None = None, delta_i: np.ndarray | None = None) -> np.ndarray:
    """
    Energy per level:
        E_i ~ 4^(n-i) k_i [ (theta_i - alpha_i)^2 + (sum_{m<i} theta_m)^2 ]
    If alpha_i is None, assumes alpha_i = 0.
    """
    thetas = np.asarray(thetas, dtype=float)
    n = len(thetas)

    k_i = np.asarray(k_i, dtype=float) if k_i is not None else np.ones(n, dtype=float)
    delta_i = np.asarray(delta_i, dtype=float) if delta_i is not None else np.zeros(n, dtype=float)

    sum_thetas = 0.0
    sum_delta = 0.0
    energy_by_levels = np.zeros(n, dtype=float)
    for i in range(n):
        multiplicidad = 4.0**(n - i) 
        energy_by_levels[i] = multiplicidad * k_i[i] * ((thetas[i] - delta_i[i])**2 + (sum_thetas-sum_delta)**2)
        sum_delta += delta_i[i]
        sum_thetas += thetas[i]
    return energy_by_levels

def solve_thetaN_brent(fL: float, k_N: float = 1.0, delta_N: float | None = None) -> float:
    delta_N = delta_N if delta_N is not None else 0.0
    f = lambda th: 8.0*k_N*(th-delta_N) - 0.5*fL*((math.cos(th/2)-math.sin(th/2))/(math.cos(th/2)+math.sin(th/2)))
    return brentq(f, -math.pi/2 +1e-8 , math.pi/2 - 1e-8)  

def compute_r_s_vectors(N: int, lambda_restriction: float, k_i: np.ndarray | None = None, theta_1: float | None = None ,delta_i: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
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
    k_i = np.asarray(k_i, dtype=float) if k_i is not None else np.ones(N, dtype=float)
    delta_i = np.asarray(delta_i, dtype=float) if delta_i is not None else np.zeros(N, dtype=float)
    theta_1 = theta_1 if theta_1 is not None else 1.0
    r = np.zeros(N, dtype=float)
    s = np.zeros(N, dtype=float)

    # Condiciones iniciales
    r[0] = 1.0   # r1
    s[0] = 1.0   # s1 
    q = 0

    C = lambda_restriction / 2.0

    # j_idx = 0..N-2
    for j_idx in range(N - 1):
        B_j = 2.0 * (4.0 ** (N - (j_idx + 1)))

        k_j   = k_i[j_idx]       # k_{j}
        k_j1  = k_i[j_idx + 1]   # k_{j+1}
        delta_j   = delta_i[j_idx] 
        delta_j1  = delta_i[j_idx + 1]

        denom = (B_j * k_j1 + C)
        beta_j = (B_j * k_j1) / denom
        alpha_j = (4.0 * B_j * k_j + C) / denom
        if theta_1 == 0.0:
            gamma_j = 0.0
        else:
            q += delta_j 
            gamma_j = (B_j / theta_1) * (k_j1 * (delta_j1-q) - 4*k_j * delta_j) / denom

        r[j_idx + 1] = alpha_j * r[j_idx] + beta_j * s[j_idx] + gamma_j
        s[j_idx + 1] = s[j_idx] + r[j_idx + 1]

    return r, s

def solve_theta1_from_delta(N, lambda_restriction, k_i, delta_i, theta1_guess):
    k_i = np.asarray(k_i, float)
    delta_i = np.asarray(delta_i, float)

    # 1) theta_N_real con delta_N
    theta_N_real = solve_thetaN_brent(lambda_restriction, k_i[-1], delta_i[-1])

    # 2) define F(theta1)
    def F(theta1):
        r, s = compute_r_s_vectors(N, lambda_restriction, k_i, theta_1=theta1, delta_i=delta_i)
        return r[-1] * theta1 - theta_N_real

    # 3) bracket alrededor del guess (puedes ajustar)
    lo = -np.pi/2 + 1e-6
    hi = np.pi/2 - 1e-6

    # Asegurar cambio de signo
    f_lo, f_hi = F(lo), F(hi)
    it = 0
    while f_lo * f_hi > 0 and it < 20:
        hi *= 2.0
        f_hi = F(hi)
        it += 1

    if f_lo * f_hi > 0:
        raise RuntimeError("No pude encerrar la raíz de theta1. Revisa delta_i o el bracket.")

    theta1 = brentq(F, lo, hi, xtol=1e-12, rtol=1e-10, maxiter=200)
    r, s = compute_r_s_vectors(N, lambda_restriction, k_i, theta_1=theta1, delta_i=delta_i)

    return theta1, theta_N_real, r, s

def theta_second_order_calculation(N, lambda_restriction, k_i=None, delta_i=None):
    k_i = np.asarray(k_i, float) if k_i is not None else np.ones(N, float)

    # baseline delta=0
    theta_N0 = solve_thetaN_brent(lambda_restriction, k_i[-1], delta_N=None)
    r0, s0 = compute_r_s_vectors(N, lambda_restriction, k_i, theta_1=None, delta_i=None)
    theta1 = theta_N0 / r0[-1]

    if delta_i is None:
        return r0 * theta1

    # delta != 0: resuelve theta1 por root-finding
    theta1, thetaN_real, r, s = solve_theta1_from_delta(N, lambda_restriction, k_i, delta_i, theta1)
    return r * theta1

def example_animation():
    def random_zero_sum_alternating(N, a):
        if N % 2 != 0:
            raise ValueError("n debe ser par para que la suma pueda ser 0.")
        half = N // 2
        vals = np.array([a] * half + [-a] * half, dtype=float)
        np.random.shuffle(vals)  # mezclar al azar
        return vals

    n = 4  # Número de ángulos
    k_i = [1, 1, 1, 1]  # Constantes k_i
    delta_i = random_zero_sum_alternating(n, a=0.1)  
    
    # Rango de valores para lambda_restriction
    lambda_values = np.logspace(-3, 3.0, 60) 
    lambda_values = np.insert(lambda_values, 0, 0.0)


    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 5),
        gridspec_kw={'width_ratios': [5.2, 4.8]}
    )

    def update(frame):
        lambda_restriction = lambda_values[frame]
        thetas = theta_second_order_calculation(n, lambda_restriction, k_i, delta_i)
        energy_by_levels = potential_by_level(np.asarray(thetas, dtype=float), np.asarray(k_i, dtype=float), np.asarray(delta_i, dtype=float))
        print('Total energy:', np.sum(energy_by_levels))
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

    anim = FuncAnimation(fig, update, frames=len(lambda_values), interval=0)
    plt.show()

if __name__ == "__main__":
    example_animation()