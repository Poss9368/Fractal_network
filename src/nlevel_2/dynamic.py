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
    N = len(thetas)
    sum_thetas = 0.0
    energy_by_levels = np.zeros(N)
    for i in range(N):
        mutiplicidad = 2*4**(N-(i+1)) * k_i[i]
        energy_by_levels[i] = mutiplicidad * (thetas[i] - sum_thetas)**2
        sum_thetas += thetas[i]
    return energy_by_levels

@njit
def potential(thetas: np.ndarray, k_i: np.ndarray) -> float:
    """
    Calculate the energy of a system given the angles

    Args:
        thetas (array): Angles of the system.
    Returns:
        float: The energy of the system.
    """
    N = len(thetas)
    ant_thetas = 0.0
    energy = 0.0 
    for i in range(N):
        mutiplicidad = 2*4**(N-(i+1)) * k_i[i]
        energy += mutiplicidad * (thetas[i] - ant_thetas)**2
        ant_thetas = thetas[i]

    return energy

@njit
def potential_gradient(thetas, k_i):
    n = len(thetas)

    residuals = np.zeros(n)
    terms = np.zeros(n)
    gradient = np.zeros(n)

    ant_thetas = 0.0
    for i in range(n):
        coefficient = 2.0 * (4 ** (n - i - 1)) * k_i[i]
        residuals[i] = thetas[i] - ant_thetas
        terms[i] = coefficient * residuals[i]
        ant_thetas = thetas[i]

    # gradient[i] = 2*(t_i - t_{i+1}), con t_n = 0
    for i in range(n):
        next_term = terms[i + 1] if i + 1 < n else 0.0
        gradient[i] = 2.0 * (terms[i] - next_term)

    return gradient

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
    tan_tethas_2 = np.tan(thetas * 0.5)
    term1 = 1.0
    term2 = 0.0
    for i in range(n):
        term1 = term1*cos_tethas_2[i]
        term2 = term2 + tan_tethas_2[i]
    return term1 * (1 + term2)

@njit
def constraint_term_gradient(thetas: np.ndarray) -> np.ndarray:
    """
    Calcula el gradiente de constraint_term respecto a cada theta_i.

    Args:
        thetas (array): Ángulos del sistema.
    Returns:
        array: d(constraint_term)/d(theta_i) para cada i.
    """
    cos_half = np.cos(thetas * 0.5)
    tan_half = np.tan(thetas * 0.5)

    C = np.prod(cos_half)     # term1
    T = np.sum(tan_half)      # term2

    grad = 0.5 * C * (1.0 / cos_half**2 - tan_half * (1.0 + T))
    return grad


@njit
def modified_hamiltonian(thetas: np.ndarray, k_i: np.ndarray ,lambda_restriction) -> float:
    """
    Calculate the total gradient of the energy and the size of the structure in the y direction.

    Args:
        thetas (array): Angles of the system.
    Returns:
        array: The total gradient of the energy and the size of the structure in the y direction.
    """

    energy = potential(thetas, k_i)
    constraint = constraint_term(thetas)
    return energy - lambda_restriction * constraint

@njit
def modified_hamiltonian_gradient(thetas: np.ndarray, k_i: np.ndarray ,lambda_restriction) -> np.ndarray:
    """
    Calculate the total gradient of the energy and the size of the structure in the y direction.

    Args:
        thetas (array): Angles of the system.
    Returns:
        array: The total gradient of the energy and the size of the structure in the y direction.
    """
    energy_grad: np.ndarray = potential_gradient(thetas, k_i)
    constraint_gradient: np.ndarray = constraint_term_gradient(thetas)
    return energy_grad - lambda_restriction * constraint_gradient

def mean_error(x: np.ndarray) -> float:
    return np.sqrt(np.mean(x**2))

def conjudate_gradient(thetas: np.ndarray, k_i: np.ndarray, lambda_restriction: float) -> np.ndarray:
    presicion: float    = 1e-7    # Presición para la minimización
    step_size: float    = 2e-5   # Tamaño del paso de integración para minimización
    n: np.ndarray       = len(thetas) # Numero de angulos del sistema"
    thetas_grad: np.ndarray  = modified_hamiltonian_gradient(thetas, k_i, lambda_restriction) # initial gradient
    error: float             = mean_error(thetas_grad) # initial error
    
    thetas_velocity_CG: np.ndarray  = np.array(np.zeros(n))  
    alpha_CG: float                 = 0 
    iter: int                       = 0
    max_iter: int                   = 1e10
    
    while error > presicion and iter < max_iter:
        if iter%10000 == 0:
            print(f"n = {n}, iter = {iter}, error = {error:.2e}")
        thetas_velocity_CG = thetas_grad + alpha_CG * thetas_velocity_CG
        thetas -= step_size * thetas_velocity_CG
        thetas_grad = modified_hamiltonian_gradient(thetas, k_i, lambda_restriction)
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
    thetas_init[-1] = np.pi / 1024  # Último ángulo en radianes
    k_i = np.ones(n)  # Constantes de rigidez/acoplamiento para cada nivel

    k_i = []
    for i in range(n):
        k_i.append((2*4**(n-(i+1)))**-1)

    # Rango de valores para lambda_restriction
    lambda_values = np.logspace(-5, 1.5, 60) 

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 5),
        gridspec_kw={'width_ratios': [5.2, 4.8]}
    )

    def update(frame):
        lambda_restriction = lambda_values[frame]
        thetas = conjudate_gradient(thetas_init, k_i, lambda_restriction)
        energy_by_levels = potential_by_level(thetas, k_i)
        
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
        ax2.set_ylim([0, 5])
        for i, y in zip(x_vals, y_vals):
             ax2.text(i, y, f"{y:.2f}", fontsize=9, ha='left', va='bottom')

    anim = FuncAnimation(fig, update, frames=len(lambda_values), interval=30)
    plt.show()

if __name__ == "__main__":
    example_animation()
    #thetas example
    # theta = np.array([np.pi/8, np.pi/4, np.pi/6, np.pi/3])
    # structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min = structure_size(theta)
    # constraint_term = constraint_term(theta)
    # print(f"Structure size x max: {structure_size_x_max}, x min: {structure_size_x_min}, y max: {structure_size_y_max}, y min: {structure_size_y_min}")
    # print(f"Constraint term: {constraint_term}")

    







