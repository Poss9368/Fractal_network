from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numba import njit
from matplotlib.animation import FFMpegWriter

try:
    from .plot_nlevels import draw_total_levels, pint_box, print_square, print_structure
except ImportError:  # Mantiene la ejecución directa: python dynamic.py
    from plot_nlevels import draw_total_levels, pint_box, print_square, print_structure

square_color = '#4DD0E1'  # color del cuadrado


def balanced_random_signs(n: int, rng=None) -> np.ndarray:
    """Genera N/2 signos -1 y N/2 signos +1 en orden aleatorio.

    ``n`` debe ser un entero positivo par. ``rng`` puede ser un
    ``numpy.random.Generator`` o un ``numpy.random.RandomState``; si se
    omite, se crea un generador nuevo.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n debe ser un entero positivo par")

    n = int(n)
    if n < 2 or n % 2 != 0:
        raise ValueError("n debe ser un entero positivo par")

    if rng is None:
        rng = np.random.default_rng()

    signs = np.ones(n, dtype=float)
    signs[:n // 2] = -1.0
    rng.shuffle(signs)
    return signs


@njit
def maximum_length_angle(n: int) -> float:
    """Ángulo común que maximiza la longitud para una jerarquía de n niveles."""
    if n < 1:
        raise ValueError("n debe ser positivo")
    t_star = 2.0 / (1.0 + np.sqrt(4.0 * n - 3.0))
    return 2.0 * np.arctan(t_star)

@njit
def rest_thetas_from_noise(noise: np.ndarray) -> np.ndarray:
    """Construye theta^(0) = theta* + noise, con noise medido en radianes."""
    theta_star = maximum_length_angle(len(noise))
    return theta_star + noise


@njit
def rest_deformations_from_thetas(rest_thetas: np.ndarray) -> np.ndarray:
    """Convierte una geometría natural theta^(0) en [r_0^-, r_0^+].

    Se usan las variables angulares reducidas

        r_i^- = theta_i - sum_{j<i} theta_j,
        r_i^+ = theta_i + sum_{j<i} theta_j.

    El ángulo físico phi_i^+ es pi-r_i^+, de modo que especificar r_i^+
    equivale a especificar el ángulo natural de la familia phi+.
    """
    n = len(rest_thetas)
    rest_deformations = np.empty((n, 2))
    sum_rest_thetas = 0.0

    for i in range(n):
        rest_deformations[i, 0] = rest_thetas[i] - sum_rest_thetas
        rest_deformations[i, 1] = rest_thetas[i] + sum_rest_thetas
        sum_rest_thetas += rest_thetas[i]

    return rest_deformations

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
def potential_by_level(
    thetas: np.ndarray,
    k_i: np.ndarray,
    family_weights: np.ndarray,
    rest_deformations: np.ndarray,
) -> np.ndarray:
    """
    Energía por nivel con familias y equilibrios configurables.

    family_weights[i] = [w_i^-, w_i^+] contiene pesos en [0, 1].
    rest_deformations[i] = [r_i^{-,0}, r_i^{+,0}] contiene los valores
    naturales, en radianes, de las variables reducidas r_i^- y r_i^+.

    Los pesos son independientes y no tienen que sumar uno. Por ejemplo,
    [1, 1] conserva ambas familias completas y [1, 0] sólo phi-.
    """
    N = len(thetas)

    if (
        len(k_i) != N
        or family_weights.ndim != 2
        or family_weights.shape[0] != N
        or family_weights.shape[1] != 2
        or rest_deformations.ndim != 2
        or rest_deformations.shape[0] != N
        or rest_deformations.shape[1] != 2
    ):
        raise ValueError("family_weights y rest_deformations deben tener forma (N, 2)")

    sum_thetas = 0.0
    energy_by_levels = np.zeros(N)

    for i in range(N):
        elastic_prefactor = 4.0 ** (N - (i + 1)) * k_i[i]
        weight_minus = family_weights[i, 0]
        weight_plus = family_weights[i, 1]
        if weight_minus < 0.0 or weight_minus > 1.0 or weight_plus < 0.0 or weight_plus > 1.0:
            raise ValueError("los pesos de las familias deben pertenecer a [0, 1]")

        deformation_minus = (
            thetas[i] - sum_thetas - rest_deformations[i, 0]
        )
        deformation_plus = (
            thetas[i] + sum_thetas - rest_deformations[i, 1]
        )
        energy_by_levels[i] = elastic_prefactor * (
            weight_minus * deformation_minus**2
            + weight_plus * deformation_plus**2
        )

        sum_thetas += thetas[i]

    return energy_by_levels

@njit
def potential(
    thetas: np.ndarray,
    k_i: np.ndarray,
    family_weights: np.ndarray,
    rest_deformations: np.ndarray,
) -> float:
    """
    Energía total para pesos y deformaciones naturales independientes.
    """
    n = len(thetas)
    if (
        len(k_i) != n
        or family_weights.ndim != 2
        or family_weights.shape[0] != n
        or family_weights.shape[1] != 2
        or rest_deformations.ndim != 2
        or rest_deformations.shape[0] != n
        or rest_deformations.shape[1] != 2
    ):
        raise ValueError("family_weights y rest_deformations deben tener forma (N, 2)")

    sum_thetas = 0.0
    energy = 0.0
    for i in range(n):
        elastic_prefactor = 4.0 ** (n - i - 1) * k_i[i]
        weight_minus = family_weights[i, 0]
        weight_plus = family_weights[i, 1]
        if weight_minus < 0.0 or weight_minus > 1.0 or weight_plus < 0.0 or weight_plus > 1.0:
            raise ValueError("los pesos de las familias deben pertenecer a [0, 1]")

        deformation_minus = (
            thetas[i] - sum_thetas - rest_deformations[i, 0]
        )
        deformation_plus = (
            thetas[i] + sum_thetas - rest_deformations[i, 1]
        )
        energy += elastic_prefactor * (
            weight_minus * deformation_minus**2
            + weight_plus * deformation_plus**2
        )
        sum_thetas += thetas[i]

    return energy

@njit
def potential_gradient(
    thetas: np.ndarray,
    k_i: np.ndarray,
    family_weights: np.ndarray,
    rest_deformations: np.ndarray,
) -> np.ndarray:
    n = len(thetas)

    if (
        len(k_i) != n
        or family_weights.ndim != 2
        or family_weights.shape[0] != n
        or family_weights.shape[1] != 2
        or rest_deformations.ndim != 2
        or rest_deformations.shape[0] != n
        or rest_deformations.shape[1] != 2
    ):
        raise ValueError("family_weights y rest_deformations deben tener forma (N, 2)")

    direct_terms = np.zeros(n)
    previous_angle_terms = np.zeros(n)
    gradient = np.zeros(n)

    sum_thetas = 0.0

    for i in range(n):
        coefficient = 2.0 * 4.0 ** (n - i - 1) * k_i[i]
        weight_minus = family_weights[i, 0]
        weight_plus = family_weights[i, 1]
        if weight_minus < 0.0 or weight_minus > 1.0 or weight_plus < 0.0 or weight_plus > 1.0:
            raise ValueError("los pesos de las familias deben pertenecer a [0, 1]")

        residual_minus = (
            thetas[i] - sum_thetas - rest_deformations[i, 0]
        )
        residual_plus = (
            thetas[i] + sum_thetas - rest_deformations[i, 1]
        )

        direct_terms[i] = coefficient * (
            weight_minus * residual_minus
            + weight_plus * residual_plus
        )
        previous_angle_terms[i] = coefficient * (
            -weight_minus * residual_minus
            + weight_plus * residual_plus
        )

        sum_thetas += thetas[i]

    # theta_i entra en todos los niveles posteriores: con signo - en r^-
    # y con signo + en r^+.
    future_sum = 0.0

    for i in range(n - 1, -1, -1):
        gradient[i] = direct_terms[i] + future_sum

        future_sum += previous_angle_terms[i]

    return gradient

@njit
def constraint_term(thetas: np.ndarray) -> float:
    """
    Calcula la longitud horizontal X de la estructura.

    Args:
        thetas (array): Angles of the system.
    Returns:
        float: Longitud horizontal conjugada a la fuerza.
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
def modified_hamiltonian(
    thetas: np.ndarray,
    k_i: np.ndarray,
    family_weights: np.ndarray,
    rest_deformations: np.ndarray,
    lambda_restriction,
) -> float:
    """
    Potencial a fuerza impuesta H = E - lambda * X.

    Args:
        thetas (array): Angles of the system.
    Returns:
        float: Valor del potencial mecánico.
    """

    energy = potential(thetas, k_i, family_weights, rest_deformations)
    constraint = constraint_term(thetas)
    return energy - lambda_restriction * constraint

@njit
def modified_hamiltonian_gradient(
    thetas: np.ndarray,
    k_i: np.ndarray,
    family_weights: np.ndarray,
    rest_deformations: np.ndarray,
    lambda_restriction,
) -> np.ndarray:
    """
    Gradiente de H = E - lambda * X respecto de los ángulos theta.

    Args:
        thetas (array): Angles of the system.
    Returns:
        array: Gradiente del potencial mecánico.
    """
    energy_grad: np.ndarray = potential_gradient(
        thetas, k_i, family_weights, rest_deformations
    )
    constraint_gradient: np.ndarray = constraint_term_gradient(thetas)
    return energy_grad - lambda_restriction * constraint_gradient

def mean_error(x: np.ndarray) -> float:
    return np.sqrt(np.mean(x**2))

def conjudate_gradient(
    thetas: np.ndarray,
    k_i: np.ndarray,
    family_weights: np.ndarray,
    rest_deformations: np.ndarray,
    lambda_restriction: float,
) -> np.ndarray:
    presicion: float    = 1e-7    # Presición para la minimización
    step_size: float    = 3e-5   # Tamaño del paso de integración para minimización
    n: np.ndarray       = len(thetas) # Numero de angulos del sistema"
    thetas_grad: np.ndarray = modified_hamiltonian_gradient(
        thetas, k_i, family_weights, rest_deformations, lambda_restriction
    )
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
        thetas_grad = modified_hamiltonian_gradient(
            thetas, k_i, family_weights, rest_deformations,
            lambda_restriction,
        )
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
    n = 4  # Debe ser par para construir ruido binario balanceado.

    # Ruido binario en theta^(0), medido en radianes.
    seed  = 5
    rng = np.random.RandomState(seed)
    
    noise_amplitude = 1.0 * maximum_length_angle(n)
    noise = noise_amplitude * balanced_random_signs(n, rng)
    thetas_init = rest_thetas_from_noise(noise)
    rest_deformations = rest_deformations_from_thetas(thetas_init)
    family_weights = rng.random_sample((n, 2))
    print(f"Noise: {noise}")
    print(f"Family weights [phi-, phi+]:\n{family_weights}")

    # La misma rigidez microscópica en todos los niveles:
    # k_i = 4^(-N), por lo que w_i = 4^(N-i) k_i = 4^(-i).
    microscopic_stiffness = np.exp2(-2.0 * n)  # 4**(-N)
    k_i = np.full(n, microscopic_stiffness, dtype=float)

    # Rango de valores para lambda_restriction
    lambda_values = np.logspace(-5, 1.5, 60) 

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 5),
        gridspec_kw={'width_ratios': [5.2, 4.8]}
    )

    def update(frame):
        lambda_restriction = lambda_values[frame]
        thetas = conjudate_gradient(
            thetas_init, k_i, family_weights, rest_deformations,
            lambda_restriction,
        )
        energy_by_levels = potential_by_level(
            thetas, k_i, family_weights, rest_deformations
        )
        
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

    
