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
        structure_size_y_max = 2*sin_tethas_2[i]*structure_size_x_max + 2*structure_size_y_min*cos_tethas_2[i] 
        structure_size_x_min = 2*structure_size_y_min*sin_tethas_2[i]

        structure_size_x_max = 2*sin_tethas_2[i]*structure_size_y_min + 2*structure_size_x_max*cos_tethas_2[i] 
        structure_size_y_min = 2*structure_size_y_min*cos_tethas_2[i] 

    return structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min

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
    return grad   # Normalizar por 2^n para que la energía sea adimensional

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
    structure_size_y = 1.0
    for i in range(n):
        structure_size_x = 2*(structure_size_x*cos_tethas_2[i] + structure_size_y*sin_tethas_2[i])
        structure_size_y = 2*structure_size_y*cos_tethas_2[i]
    return 1.0 - structure_size_x/(2**n) 

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

    def aux_function(j: float, cos_tethas_2: np.ndarray, sin_tethas_2: np.ndarray) -> float:
        """
        Auxiliary function to calculate the gradient for a specific angle.
        """
        grad_x = 1.0
        grad_y = 1.0
        for i in range(n):
            if i == j:
                grad_x = -4*grad_x*sin_tethas_2[i] + 4*grad_y*cos_tethas_2[i] 
                grad_y = -4*grad_y*sin_tethas_2[i]
            else:
                grad_x = 2*grad_x*cos_tethas_2[i] + 2*grad_y*sin_tethas_2[i]
                grad_y = 2*grad_y*cos_tethas_2[i]
        return grad_x

    grad = np.zeros(n)
    for j in range(n):
        grad[j] = aux_function(j, cos_tethas_2, sin_tethas_2)

    return - grad / 2**n 

@njit
def modified_hamiltonian_gradient(thetas: np.ndarray, lambda_restriction) -> np.ndarray:
    """
    Calculate the total gradient of the energy and the size of the structure in the y direction.

    Args:
        thetas (array): Angles of the system.
    Returns:
        array: The total gradient of the energy and the size of the structure in the y direction.
    """
    energy_grad: np.ndarray = potential_gradient(thetas)
    constraint_gradient: np.ndarray = constraint_term_gradient(thetas)
    return energy_grad + lambda_restriction * constraint_gradient

def mean_error(x: np.ndarray) -> float:
    return np.sqrt(np.mean(x**2))

def conjudate_gradient(thetas: np.ndarray, lambda_restriction: float) -> np.ndarray:
    presicion: float    = 1e-8    # Presición para la minimización
    step_size: float    = 1e-5    # Tamaño del paso de integración para minimización
    n: np.ndarray       = len(thetas) # Numero de angulos del sistema"
    thetas_grad: np.ndarray  = modified_hamiltonian_gradient(thetas, lambda_restriction) # initial gradient
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
    n = 3  # Número de ángulos
    thetas_init = np.zeros(n)  # Inicializar con ceros
    thetas_init[-1] = np.pi / 512  # Último ángulo en radianes

    # Rango de valores para lambda_restriction
    lambda_values = np.logspace(-2, 1.7, 200) 

    fig, ax = plt.subplots()

    def update(frame):
        lambda_restriction = lambda_values[frame]
        thetas = conjudate_gradient(thetas_init, lambda_restriction)
        ax.clear()
        structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min = print_structure(thetas, ax)
        ax.set_title(f"Lambda = {lambda_restriction:.2e}\nLx = {structure_size_x_max:-2f} \nArea = {structure_size_x_max * structure_size_y_max:.2f}")
    anim = FuncAnimation(fig, update, frames=len(lambda_values), interval=30)

    ## guardar la animación en formato gif"
    ##anim.save('dynamic_exactly.gif', writer='pillow', fps=50)

    plt.show()


if __name__ == "__main__":
    example_animation()  

    







