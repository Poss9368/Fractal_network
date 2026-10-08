from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


PATH = Path(__file__).resolve().parent

if __name__ == "__main__":
    ns = [16, 64, 256, 1024, 4096, 16384, 65536]
    fig, ax1 = plt.subplots(figsize=(8, 6))
    for n in ns:
        file_name = f"{n}_angles_output.csv"
        df = pd.read_csv(PATH / file_name)
        force = df['lambda'].to_numpy() / (n ** (3/2))
        epsilon = df['strain_median'].to_numpy()
        epsilon_q25 = df['strain_q25'].to_numpy()
        epsilon_q75 = df['strain_q75'].to_numpy()
        valid = (force > 0.0) & (epsilon > 0.0)
        band_valid = valid & (epsilon_q25 > 0.0) & (epsilon_q75 > 0.0)
        ax1.plot(epsilon[valid], force[valid], label=f'n={n}')
        ax1.fill_betweenx(
            force[band_valid], epsilon_q25[band_valid], epsilon_q75[band_valid],
            alpha=0.2,
        )

    exp1 = 2.1
    x1 = np.logspace(-5, -0.3, 100)
    ax1.plot(x1, x1**(exp1)*15, 'k--', label=fr'$f \propto \epsilon^{{{exp1}}}$')

    exp2 = 1
    x2 = np.logspace(-6, -2, 100)
    #ax1.plot(x2, x2**(exp2)*0.01, 'b--', label=fr'$f \propto \epsilon^{{{exp2}}}$')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$\epsilon$')
    ax1.set_ylabel(r'$f / n^{3/2}$')
    ax1.legend()
    ax1.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.savefig(PATH / "force_vs_epsilon.png", dpi=300, bbox_inches="tight")
    plt.show()    
