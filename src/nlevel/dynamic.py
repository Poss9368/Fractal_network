import jax.numpy as jnp
from jax import grad

def potential(thetas: jnp.ndarray) -> float:
    """
    Calculate the energy of a system given the angles

    Args:
        thetas (array): Angles of the system.
    Returns:
        float: The energy of the system.
    """
    number_of_levels = len(thetas)
    energy = 0.0 
    for i in range(len(thetas) - 1):
        if i == 0:
            energy += (4**(number_of_levels-1-i))*4*thetas[i]**2
        else:
            energy += (4**(number_of_levels-1-i)) * 2 *((thetas[i]-thetas[i-1])**2 + (thetas[i]+thetas[i-1])**2)

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

    size_y = 1.0
    for i in range(len(thetas)):
        size_y *= 2*jnp.cos(thetas[i]*0.5)
    L_y = 0
    return L_y - size_y


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
    size_y_grad = constraint_term_gradient(thetas)
    return energy_grad + lambda_restriction * size_y_grad


def mean_error(x: jnp.ndarray) -> float:
    x = jnp.array(x)
    return np.sqrt(np.mean(x**2))

if __name__ == "__main__":
    import numpy as np
    # Example usage
    thetas = jnp.array([1e-9, 1e-9])  # Example angles for a 4-level system
    N = len(thetas)  # Number of angles
    lambda_restriction = 2
    presicion = 5e-9 # Presición para la minimización

    thetas_grad = -modified_hamiltonian_gradient(thetas, lambda_restriction)
    error = mean_error(thetas_grad)

    step_size = 0.01  # Tamaño del paso de integración para minimización
    thetas_velocity_CG = jnp.array(np.zeros(N))
    alpha_CG = 0 

    while error > presicion:
        thetas_velocity_CG = thetas_grad + alpha_CG * thetas_velocity_CG
        thetas += step_size * thetas_velocity_CG
        thetas_grad = -modified_hamiltonian_gradient(thetas, lambda_restriction)
        new_error = mean_error(thetas_grad)

        if new_error < error:
            alpha_CG = new_error/error
        else:
            alpha_CG = 0.0
        error = new_error
        print(f"error: {error}")

    print(f"Optimized angles: {thetas}")

    











