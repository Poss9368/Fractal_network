import jax.numpy as jnp
from jax import grad
from matplotlib import pyplot as plt
import numpy as np

from old.nlevel.plot_nlevels import draw_total_levels, pint_box, print_structure

def potential(thetas: jnp.ndarray) -> float:
    """
    Calculate the energy of a system given the angles

    Args:
        thetas (array): Angles of the system.
    Returns:
        float: The energy of the system.
    """
    number_of_levels = len(thetas)
    sum_thetas = 0.0
    energy = 0.0 
    for i in range(len(thetas)):
        theta = thetas[i]
        mutiplicidad = 4**(number_of_levels-1-i) 
        energy += mutiplicidad * 2 * (theta - sum_thetas)**2 
        energy += mutiplicidad * 2 * (theta + sum_thetas)**2
        sum_thetas += theta
    return energy

def potential_gradient(thetas: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the gradient of the energy with respect to the angles using automatic differentiation.

    Args:
        thetas (array): Angles of the system.
    Returns:
        array: The gradient of the energy with respect to the angles.
    """
    grad_fn = grad(potential)
    return grad_fn(thetas)

def constraint_term(thetas: jnp.ndarray) -> float:
    """
    Calculate the size of the structure in the y direction given the angles.

    Args:
        thetas (array): Angles of the system.
    Returns:
        float: The size of the structure in the y direction.
    """
    structure_size_x = 1.0
    structure_size_y = 1.0
    for theta in thetas:
        structure_size_x = 2*structure_size_x/jnp.cos(theta * 0.5) + 2*jnp.sin(theta * 0.5)*(structure_size_y - structure_size_x*jnp.tan(theta * 0.5)) 
        structure_size_y = 2*structure_size_y*jnp.cos(theta * 0.5)
    L_x = 0
    L_y = 0
    return (L_y - structure_size_y + L_x - structure_size_x) 

def constraint_term_gradient(thetas: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the gradient of the size of the structure in the y direction with respect to the angles.

    Args:
        thetas (array): Angles of the system.
    Returns:
        array: The gradient of the size of the structure in the y direction with respect to the angles.
    """
    grad_fn = grad(constraint_term)
    return grad_fn(thetas)

def modified_hamiltonian_gradient(thetas: jnp.ndarray, lambda_restriction) -> jnp.ndarray:
    """
    Calculate the total gradient of the energy and the size of the structure in the y direction.

    Args:
        thetas (array): Angles of the system.
    Returns:
        array: The total gradient of the energy and the size of the structure in the y direction.
    """
    energy_grad = potential_gradient(thetas)
    constraint_gradient = constraint_term_gradient(thetas)
    return energy_grad + lambda_restriction * constraint_gradient

def mean_error(x: jnp.ndarray) -> float:
    x = jnp.array(x)
    return np.sqrt(np.mean(x**2))

if __name__ == "__main__":
    thetas = jnp.array([np.pi/32, np.pi/16, np.pi/8, np.pi/4])  # initial sqrt angles of the system
    lambda_restriction =  10 # Lagrange multiplier for the constraint term
    presicion = 1e-7 # Presición para la minimización

    thetas_grad = modified_hamiltonian_gradient(thetas, lambda_restriction)
    error = mean_error(thetas_grad)

    step_size = 0.001  # Tamaño del paso de integración para minimización
    thetas_velocity_CG = jnp.array(np.zeros(len(thetas)))
    alpha_CG = 0 

    iter = 0
    while error > presicion and iter < 300:
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
        print(f"Iteration {iter}, Error: {error}")
    
    print(f"Optimized angles: {thetas*180/np.pi}, sum of angles: {jnp.sum(thetas) * 180 / np.pi}")
    
    square_size = 1.0
    structure, structure_size_x, structure_size_y = draw_total_levels(square_size, thetas)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axis('off')
    print_structure(ax, structure)
    pint_box(ax, structure_size_x, structure_size_y)
    plt.show()

    
    











