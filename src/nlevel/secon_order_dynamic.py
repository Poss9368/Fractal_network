from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from dynamic import potential_by_level
from plot_nlevels import print_structure


def compute_r_vector(N, lambda_restriction):
    """
    Calcula el vector r_j para j=1,...,N usando las recurrencias dadas.
    Args:
        N (int): Número de niveles.
        L (float): Parámetro L.
        f (float): Parámetro f.
    Returns:
        np.ndarray: Vector r de tamaño N.
    """
    r = np.zeros(N)
    s = np.zeros(N)
    r[0] = 1.0
    s[0] = 1.0
    for j in range(N-1):
        Bj = 2 * 4**(N - (j+1))
        C = lambda_restriction / 2.0
        bj = Bj / (Bj + C)
        alphaj = 1 + 3 * bj
        r[j+1] = alphaj * r[j] + bj * s[j]
        s[j+1] = s[j] + r[j+1]
    return r

def theta_second_order_calculation(N, lambda_restriction):
    r = compute_r_vector(N, lambda_restriction)
    theta_1 = (lambda_restriction / (lambda_restriction + 16))*(1/r[-1])   
    thetas = r * theta_1
    return thetas

def example_animation():
    n = 5  # Número de ángulos

    def compute_aj(j, a1=1, a2=5):
            if j == 1:
                return a1
            elif j == 2:
                return a2
            else:
                a_prev = a1
                a_curr = a2
                for _ in range(3, j + 1):
                    a_next = 6 * a_curr - 4 * a_prev
                    a_prev, a_curr = a_curr, a_next
                return a_curr
    
    aj_max = compute_aj(n)

    # Rango de valores para lambda_restriction
    lambda_values = np.logspace(-1, 2.5, 60) 

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 5),
        gridspec_kw={'width_ratios': [5.2, 4.8]}
    )

    def update(frame):
        lambda_restriction = lambda_values[frame]
        thetas = theta_second_order_calculation(n, lambda_restriction)
        energy_by_levels = potential_by_level(thetas)
        ax1.clear()
        structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min = print_structure(thetas, ax1, square_size=1.0/2**n)
        ax1.set_title(f"Lambda = {lambda_restriction:.2e}\nLx = {structure_size_x_max:.2f} \nArea = {structure_size_x_max * structure_size_y_max:.2f}")

        ax2.clear()
        x_vals = np.arange(1, len(thetas)+1)
        y_vals = thetas/thetas[0]
        ax2.plot(x_vals, y_vals, 'o')
        r = compute_r_vector(n, lambda_restriction)
        ax2.plot(x_vals, r, 'x')
        ax2.set_title(r"$\theta_i / \theta_1$", fontsize=14)
        ax2.set_xticks(x_vals)
        ax2.set_xlabel("i")
        ax2.set_ylim([0, aj_max * 1.1])
        for i, y in zip(x_vals, y_vals):
            ax2.text(i, y, f"{y:.2f}", fontsize=9, ha='left', va='bottom')

        # ax2.clear()
        # x_vals = np.arange(1, len(thetas)+1)
        # y_vals = energy_by_levels 
        # ax2.plot(x_vals, y_vals, 'o')
        # ax2.set_title(r"$E_i / E$", fontsize=14)
        # ax2.set_xticks(x_vals)
        # ax2.set_xlabel("i")
        # ax2.set_ylim([0, 80])
        # for i, y in zip(x_vals, y_vals):
        #     ax2.text(i, y, f"{y:.2f}", fontsize=9, ha='left', va='bottom')

    anim = FuncAnimation(fig, update, frames=len(lambda_values), interval=30)
    plt.show()

if __name__ == "__main__":
    example_animation()  