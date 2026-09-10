from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import matplotlib.pyplot as plt

def energy_spectrum(N, lam, alpha=1.0, C=1.0):
    """
    Calcula el espectro de energía promedio <E_k(lambda)> para k=0,...,N-1.

    Parámetros
    ----------
    N : int
        Número de modos / tamaño del sistema.
    lam : float
        Parámetro lambda.
    alpha : float, opcional
        Amplitud del estado inicial. Por defecto 1.0.
    C : float, opcional
        Constante prefactor de la energía. Por defecto 1.0.

    Retorna
    -------
    E : np.ndarray
        Vector de tamaño N con <E_k(lambda)> para cada k.
    """
    k = np.arange(N)
    Lambda_k = 4.0 * np.sin(np.pi * k / N)**2

    E = C * alpha**2 * Lambda_k * (lam / (Lambda_k + lam))**2
    return E

if __name__ == "__main__":
    N = 36
    # Rango de valores para lambda_restriction
    x_min = -5.0
    x_max = 5.0 
    bin = 60
    lambda_values = np.logspace(x_min, x_max, bin)

    fig, ax = plt.subplots(figsize=(10, 4.8))

    def update(frame):
        lam = lambda_values[frame]
        E = energy_spectrum(N, lam)
        x_vals = np.arange(0, N )
        E = E / np.sum(E)  # Normalizar para que sum(E) = 1
        ax.clear()
        ax.plot(x_vals, E, 'o', label=f'f={lam:.2e}')
        ax.set_title(r"$E_k / E$", fontsize=14)
        ax.set_xticks(x_vals)
        ax.set_xlabel("k")
        ax.set_ylim([0, 1])
        ax.legend(loc='upper left')
        for i, y in zip(x_vals, E):
            ax.text(i, y, f"{y:.2f}", fontsize=9, ha='left', va='bottom')   
    
    anim = FuncAnimation(fig, update, frames=len(lambda_values), interval=100)
    #writer = PillowWriter(fps=10)
    #anim.save("energy_spectrum_1D.gif", writer=writer, dpi=250)
    plt.show()
