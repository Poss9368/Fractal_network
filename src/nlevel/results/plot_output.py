from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PATH = Path(__file__).resolve().parent

if __name__ == "__main__":
    ns = [4, 8, 16]
    plt.figure(figsize=(8, 6))
    for n in ns:
        file_name = f"{n}_angles_output.csv"
        df = pd.read_csv(PATH / file_name)
        lambda_values = df['lambda'].values
        structure_size_x_max = df['structure_size_x_max'].values
        structure_size_x_min = df['structure_size_x_min'].values
        structure_size_y_max = df['structure_size_y_max'].values
        structure_size_y_min = df['structure_size_y_min'].values
        area = df['area'].values
        plt.plot((structure_size_x_max-2**n)/2**n, lambda_values,'--o')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()    