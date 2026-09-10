from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


PATH = Path(__file__).resolve().parent

if __name__ == "__main__":
    ns = [4, 8, 16, 32, 64, 128]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
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
        thetas_mean = df['thetas_mean'].values[1:]  # Excluir lambda=0
        thetas_2_mean = df['thetas_2_mean'].values[1:]
        force = lambda_values/structure_size_x_max
        ax1.plot(thetas_mean, force, label=f'n={n}')
        ax2.plot(thetas_2_mean, force, label=f'n={n}')

        
    exp1 = 1.0
    x1 = np.logspace(-7, 0, 100)
    ax1.plot(x1, x1**(exp1)*30, 'k--', label=f'$f \\propto \\langle \\theta \\rangle^{{{exp1}}}$')
    
    exp2 = 0.5
    x2 = np.logspace(-14, 0, 100)
    ax2.plot(x2, x2**(exp2)*25, 'k--', label=f'$f \\propto \\langle \\theta^2 \\rangle^{{{exp2}}}$')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$\langle \theta \rangle$')
    ax1.set_ylabel(r'$f$')
    ax1.legend()
    ax1.grid(True, which="both", ls="--")

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\langle \theta^2 \rangle$')
    ax2.set_ylabel(r'$f$')
    ax2.legend()
    ax2.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.show()    