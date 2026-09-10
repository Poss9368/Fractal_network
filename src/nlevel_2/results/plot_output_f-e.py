from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


PATH = Path(__file__).resolve().parent

if __name__ == "__main__":
    ns = [8, 16]  # Lista de valores de n para los que se generarán los gráficos
    fig, ax1 = plt.subplots(figsize=(8, 6))
    for n in ns:
        file_name = f"{n}_angles_output.csv"
        df = pd.read_csv(PATH / file_name)
        # extraer primera fila (lambda=0) para obtener el tamaño inicial
        structure_size_x_max_0 = df[df['lambda'] == 0]['structure_size_x_max'].values[0]
        structure_size_x_min_0 = df[df['lambda'] == 0]['structure_size_x_min'].values[0]
        structure_size_y_max_0 = df[df['lambda'] == 0]['structure_size_y_max'].values[0]
        structure_size_y_min_0 = df[df['lambda'] == 0]['structure_size_y_min'].values[0]

        lambda_values = df['lambda'].values[1:]  # Excluir lambda=0
        structure_size_x_max = df['structure_size_x_max'].values[1:]  # Excluir lambda=0
        structure_size_x_min = df['structure_size_x_min'].values[1:]  # Excluir lambda=0
        structure_size_y_max = df['structure_size_y_max'].values[1:]  # Excluir lambda=0
        structure_size_y_min = df['structure_size_y_min'].values[1:]  # Excluir lambda=0
        area = df['area'].values[1:]  # Excluir lambda=0
        force = lambda_values
        epsilon = (structure_size_x_max - structure_size_x_max_0)
        ax1.plot(epsilon, force, label=f'n={n}')

    exp1 = 1.0
    x1 = np.logspace(-4.5, -0.5, 100)
    ax1.plot(x1, x1**(exp1)*0.05, 'k--', label=fr'$f \propto \epsilon^{{{exp1}}}$')

    exp2 = 3.0
    x2 = np.logspace(-1.5, 0.5, 100)
    ax1.plot(x2, x2**(exp2)*15, 'k--', label=fr'$f \propto \epsilon^{{{exp2}}}$')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$\epsilon$')
    ax1.set_ylabel(r'$f$')
    ax1.legend()
    ax1.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.show()    