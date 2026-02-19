from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PATH = Path(__file__).resolve().parent

if __name__ == "__main__":
    ns = [16, 32, 64]
    plt.figure(figsize=(8, 6))
    for n in ns:
        max_l = 2.0**(n/2)
        file_name = f"{n}_angles_output.csv"
        df = pd.read_csv(PATH / file_name)
        lambda_values = df['lambda'].values
        structure_size_x_max = df['structure_size_x_max'].values
        structure_size_x_min = df['structure_size_x_min'].values
        structure_size_y_max = df['structure_size_y_max'].values
        structure_size_y_min = df['structure_size_y_min'].values
        area = df['area'].values
        S = df['S'].values
        Q = df['Q'].values
        Ak = df['Ak'].values
        theta_1 = df['theta_1'].values
        theta_N = df['theta_N'].values
        plt.plot(structure_size_x_max - 1.0, lambda_values/structure_size_x_max , 'o-', label=f'n={n}')
        #plt.plot(structure_size_x_max- 1.0, theta_1) 
        #plt.plot(structure_size_x_max- 1.0, theta_N)
        #plt.plot(structure_size_x_max- 1.0, Ak)
        #plt.plot(structure_size_x_max- 1.0, S)
        #plt.plot(structure_size_x_max- 1.0, Q)
        #plt.plot(lambda_values, 4*Ak*theta_1/(S - Q*theta_1), label=f'n={n}')
    plt.xscale('log')
    plt.yscale('log') 
    #plt.xlim([1e-2, 1e0])
    plt.grid(True, which="both", ls="--")
    plt.show()    