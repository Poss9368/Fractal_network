from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numba import njit
from matplotlib.animation import FFMpegWriter

from plot_nlevels import draw_total_levels, pint_box, print_square, print_structure

square_color = '#4DD0E1'  # color del cuadrado

@njit
def structure_size(thetas: np.ndarray) -> tuple:
    """
    Calculate the size of the structure in the x and y directions given the angles.

    Args:
        thetas (array): Angles of the system.
    Returns:
        tuple: The size of the structure in the x and y directions.
    """
    n = len(thetas)
    # Precompute cos and sin of half angles
    cos_tethas_2 = np.cos(thetas * 0.5)
    sin_tethas_2 = np.sin(thetas * 0.5)

    structure_size_y_max = 1.0
    structure_size_y_min = 1.0
    structure_size_x_max = 1.0
    structure_size_x_min = 1.0
    for i in range(n):
        structure_size_y_max = sin_tethas_2[i]*structure_size_x_max + structure_size_y_min*cos_tethas_2[i] 
        structure_size_x_min = structure_size_y_min*sin_tethas_2[i]

        structure_size_x_max = sin_tethas_2[i]*structure_size_y_min + structure_size_x_max*cos_tethas_2[i] 
        structure_size_y_min = structure_size_y_min*cos_tethas_2[i] 

    return structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min

@njit
def potential_by_level(thetas: np.ndarray, k_i: np.ndarray) -> np.ndarray:
    """
    Calculate the energy of any level of the system given the angles.

    Args:
        thetas (array): Angles of the system.
        level (int): Level of the system.
    Returns:
        float: The energy of the system.
    """
    n = len(thetas)
    sum_thetas = 0.0
    energy_by_levels = np.zeros(n)
    for i in range(n):
        mutiplicidad = 4**(n-i) * k_i[i]
        energy_by_levels[i] = mutiplicidad * (thetas[i]**2 + sum_thetas**2)
        sum_thetas += thetas[i]
    return energy_by_levels

@njit
def potential(thetas: np.ndarray) -> float:
    """
    Calculate the energy of a system given the angles

    Args:
        thetas (array): Angles of the system.
    Returns:
        float: The energy of the system.
    """
    n = len(thetas)
    sum_thetas = 0.0
    energy = 0.0 
    for i in range(n):
        mutiplicidad = 4**(n-i) 
        energy += mutiplicidad * (thetas[i]**2 + sum_thetas**2) 
        sum_thetas += thetas[i]

    return energy

@njit
def potential_gradient(thetas: np.ndarray) -> np.ndarray:
    """
    Calculate the gradient of the energy with respect to the angles using analitic differentiation.

    Args:
        thetas (array): Angles of the system. 
    Returns:
        array: The gradient of the energy with respect to the angles.
    """
    n = len(thetas)
    grad = np.zeros_like(thetas)
    
    # Precomputar sumas parciales
    partial_sums = np.cumsum(thetas)  
    for k in range(n):
        term1 = 2 * (4 ** (n - k)) * thetas[k]
        term2 = 0.0
        for i in range(k + 1, n):
            S_i = partial_sums[i - 1]
            term2 += 2 * (4 ** (n - i)) * S_i
        grad[k] = term1 + term2
    return grad   

@njit
def constraint_term(thetas: np.ndarray) -> float:
    """
    Calculate the size of the structure in the y direction given the angles.

    Args:
        thetas (array): Angles of the system.
    Returns:
        float: The size of the structure in the y direction.
    """
    n = len(thetas)
    # Precompute cos and sin of half angles
    cos_tethas_2 = np.cos(thetas * 0.5)
    sin_tethas_2 = np.sin(thetas * 0.5)
    structure_size_x = 1.0
    for i in range(n):
        structure_size_x = structure_size_x*(cos_tethas_2[i] + sin_tethas_2[i])
    return 1.0 - structure_size_x

@njit
def constraint_term_gradient(thetas: np.ndarray) -> np.ndarray:
    """
    Calculate the gradient of the size of the structure in the y direction with respect to the angles.

    Args: 
        thetas (array): Angles of the system.
    Returns:
        array: The gradient of the size of the structure in the y direction with respect to the angles.
    """
    n = len(thetas)
    # Precompute cos and sin of half angles
    cos_tethas_2 = np.cos(thetas * 0.5)
    sin_tethas_2 = np.sin(thetas * 0.5)
    
    structure_size_x = 1.0
    for i in range(n):
        structure_size_x = structure_size_x*(cos_tethas_2[i] + sin_tethas_2[i])
    
    grad = np.zeros(n)
    for j in range(n):
        grad[j] = 0.5 * (cos_tethas_2[j] - sin_tethas_2[j]) * structure_size_x / (cos_tethas_2[j] + sin_tethas_2[j]) 
    
    return - grad 

@njit
def modified_hamiltonian_gradient(thetas: np.ndarray, lambda_restriction) -> np.ndarray:
    """
    Calculate the total gradient of the energy and the size of the structure in the y direction.

    Args:
        thetas (array): Angles of the system.
    Returns:
        array: The total gradient of the energy and the size of the structure in the y direction.
    """
    n = len(thetas)
    energy_grad: np.ndarray = potential_gradient(thetas)
    constraint_gradient: np.ndarray = constraint_term_gradient(thetas)
    return energy_grad + lambda_restriction * constraint_gradient

def mean_error(x: np.ndarray) -> float:
    return np.sqrt(np.mean(x**2))

def conjudate_gradient(thetas: np.ndarray, lambda_restriction: float) -> np.ndarray:
    presicion: float    = 1e-7    # Presición para la minimización
    step_size: float    = 2e-3   # Tamaño del paso de integración para minimización
    n: np.ndarray       = len(thetas) # Numero de angulos del sistema"
    thetas_grad: np.ndarray  = modified_hamiltonian_gradient(thetas, lambda_restriction) # initial gradient
    error: float             = mean_error(thetas_grad) # initial error
    
    thetas_velocity_CG: np.ndarray  = np.array(np.zeros(n))  
    alpha_CG: float                 = 0 
    iter: int                       = 0
    max_iter: int                   = 1e10
    
    while error > presicion and iter < max_iter:
        if iter%50000 == 0:
            print(f"n = {n}, iter = {iter}, error = {error:.2e}")
        thetas_velocity_CG = thetas_grad + alpha_CG * thetas_velocity_CG
        thetas -= step_size * thetas_velocity_CG
        thetas_grad = modified_hamiltonian_gradient(thetas, lambda_restriction)
        new_error = mean_error(thetas_grad)

        if new_error < error:
            alpha_CG = new_error/error
        else:
            alpha_CG = 0.0
        error = new_error
        iter += 1
        
    if iter == max_iter:
        print("Warning: Maximum number of iterations reached without convergence.")
    #print(f"Optimized angles: {thetas*180/np.pi}, sum of angles: {np.sum(thetas)*180/np.pi}")
    
    return thetas

def example_animation():
    # Ángulos iniciales del sistema
    n = 4  # Número de ángulos
    thetas_init = np.zeros(n)  # Inicializar con ceros
    thetas_init[-1] = np.pi / 512  # Último ángulo en radianes

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

    def compute_r_vector(N, L, f):
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
            C = (f/2) * L
            bj = Bj / (Bj + C)
            alphaj = 1 + 3 * bj
            r[j+1] = alphaj * r[j] + bj * s[j]
            s[j+1] = s[j] + r[j+1]
        return r
    
    # Rango de valores para lambda_restriction
    lambda_values = np.logspace(-1, 2.5, 60) 

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 5),
        gridspec_kw={'width_ratios': [5.2, 4.8]}
    )

    def update(frame):
        lambda_restriction = lambda_values[frame]
        thetas = conjudate_gradient(thetas_init, lambda_restriction)
        energy_by_levels = potential_by_level(thetas)
        ax1.clear()
        structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min = print_structure(thetas, ax1, square_size=1.0/2**n)
        ax1.set_title(f"Lambda = {lambda_restriction:.2e}\nLx = {structure_size_x_max:.2f} \nArea = {structure_size_x_max * structure_size_y_max:.2f}")

        ax2.clear()
        x_vals = np.arange(1, len(thetas)+1)
        y_vals = thetas/thetas[0]
        ax2.plot(x_vals, y_vals, 'o')
        r = compute_r_vector(n, structure_size_x_max, lambda_restriction)
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

    







