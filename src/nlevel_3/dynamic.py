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
def potential_by_level(thetas: np.ndarray, k_i: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """
    Energía phi- por nivel alrededor de una geometría natural ruidosa.

    El ruido está definido sobre los ángulos geométricos de reposo:
        theta_i^(0) = theta_star(N) + noise[i].

    Todas las bisagras elásticas son de la familia phi- y su ángulo natural
    se obtiene de la misma geometría de reposo:
        phi_i^-     = theta_i     - sum_{j<i} theta_j
        phi_i^{-,0} = theta_i^(0) - sum_{j<i} theta_j^(0).

    ``noise`` contiene perturbaciones angulares en radianes, no selectores
    binarios de las familias phi+ y phi-.
    """
    N = len(thetas)

    if len(k_i) != N or len(noise) != N:
        raise ValueError("thetas, k_i y noise deben tener la misma longitud")

    theta_star = maximum_length_angle(N)
    sum_thetas = 0.0
    sum_rest_thetas = 0.0
    energy_by_levels = np.zeros(N)

    for i in range(N):
        # E_i = w_i (Delta phi_i^-)^2,
        # w_i = 4^(N-(i+1)) k_i en la convención Python i=0,...,N-1.
        # Las dos bisagras phi- cancelan los dos factores 1/2 de sus
        # energías torsionales individuales.
        elastic_prefactor = 4.0 ** (N - (i + 1)) * k_i[i]
        rest_theta = theta_star + noise[i]
        phi_minus = thetas[i] - sum_thetas
        rest_phi_minus = rest_theta - sum_rest_thetas
        deformation = phi_minus - rest_phi_minus
        energy_by_levels[i] = elastic_prefactor * deformation**2

        sum_thetas += thetas[i]
        sum_rest_thetas += rest_theta

    return energy_by_levels

@njit
def potential(thetas: np.ndarray, k_i: np.ndarray, noise: np.ndarray) -> float:
    """
    Energía total usando sólo phi- y ruido en theta^(0).

    theta_i^(0) = theta_star(N) + noise[i], con ``noise`` en radianes.
    La energía de cada nivel es proporcional a
    (phi_i^- - phi_i^{-,0})**2.
    """
    N = len(thetas)

    if len(k_i) != N or len(noise) != N:
        raise ValueError("thetas, k_i y noise deben tener la misma longitud")

    theta_star = maximum_length_angle(N)
    sum_thetas = 0.0
    sum_rest_thetas = 0.0
    energy = 0.0

    for i in range(N):
        # E_i = w_i (Delta phi_i^-)^2, con w_i igual a la multiplicidad
        # geométrica por la rigidez física de cada bisagra.
        elastic_prefactor = 4.0 ** (N - (i + 1)) * k_i[i]
        rest_theta = theta_star + noise[i]
        phi_minus = thetas[i] - sum_thetas
        rest_phi_minus = rest_theta - sum_rest_thetas
        deformation = phi_minus - rest_phi_minus
        energy += elastic_prefactor * deformation**2

        sum_thetas += thetas[i]
        sum_rest_thetas += rest_theta

    return energy

@njit
def potential_gradient(thetas: np.ndarray, k_i: np.ndarray, noise: np.ndarray) -> np.ndarray:
    n = len(thetas)

    if len(k_i) != n or len(noise) != n:
        raise ValueError("thetas, k_i y noise deben tener la misma longitud")

    terms = np.zeros(n)
    gradient = np.zeros(n)

    theta_star = maximum_length_angle(n)
    sum_thetas = 0.0
    sum_rest_thetas = 0.0

    # Cada fila corresponde a r_i = phi_i^- - phi_i^{-,0}.
    # Para E = sum_i w_i r_i^2, terms[i] = 2 w_i r_i.
    for i in range(n):
        coefficient = (
            2.0
            * 4.0 ** (n - i - 1)
            * k_i[i]
        )

        rest_theta = theta_star + noise[i]
        residual = (
            (thetas[i] - sum_thetas)
            - (rest_theta - sum_rest_thetas)
        )

        terms[i] = coefficient * residual
        sum_thetas += thetas[i]
        sum_rest_thetas += rest_theta

    # theta_i aparece con coeficiente -1 en todas las phi_m^- con m>i.
    future_sum = 0.0

    for i in range(n - 1, -1, -1):
        gradient[i] = terms[i] + future_sum

        future_sum -= terms[i]

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
def modified_hamiltonian(thetas: np.ndarray, k_i: np.ndarray, noise: np.ndarray ,lambda_restriction) -> float:
    """
    Potencial a fuerza impuesta H = E - lambda * X.

    Args:
        thetas (array): Angles of the system.
    Returns:
        float: Valor del potencial mecánico.
    """

    energy = potential(thetas, k_i, noise)
    constraint = constraint_term(thetas)
    return energy - lambda_restriction * constraint

@njit
def modified_hamiltonian_gradient(thetas: np.ndarray, k_i: np.ndarray, noise: np.ndarray,lambda_restriction) -> np.ndarray:
    """
    Gradiente de H = E - lambda * X respecto de los ángulos theta.

    Args:
        thetas (array): Angles of the system.
    Returns:
        array: Gradiente del potencial mecánico.
    """
    energy_grad: np.ndarray = potential_gradient(thetas, k_i, noise)
    constraint_gradient: np.ndarray = constraint_term_gradient(thetas)
    return energy_grad - lambda_restriction * constraint_gradient

def mean_error(x: np.ndarray) -> float:
    return np.sqrt(np.mean(x**2))

def conjudate_gradient(thetas: np.ndarray, k_i: np.ndarray, noise: np.ndarray, lambda_restriction: float) -> np.ndarray:
    presicion: float    = 1e-8    # Presición para la minimización
    step_size: float    = 3e-5   # Tamaño del paso de integración para minimización
    n: np.ndarray       = len(thetas) # Numero de angulos del sistema"
    thetas_grad: np.ndarray  = modified_hamiltonian_gradient(thetas, k_i, noise, lambda_restriction) # initial gradient
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
        thetas_grad = modified_hamiltonian_gradient(thetas, k_i, noise, lambda_restriction)
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
    print(f"Noise: {noise}")

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
        thetas = conjudate_gradient(thetas_init, k_i, noise, lambda_restriction)
        energy_by_levels = potential_by_level(thetas, k_i, noise)
        
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

    
