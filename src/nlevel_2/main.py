from pathlib import Path
import numpy as np
from dynamic import conjudate_gradient, structure_size

PATH = Path(__file__).resolve().parent
RESULTS_PATH = PATH / 'results'


if __name__ == "__main__":
    for i in range(2, 7):
        # Ángulos iniciales del sistema
        n = 2 ** i
        thetas = np.zeros(n)  # Inicializar con ceros
        
        # Rango de valores para lambda_restriction
        x_min = -5.0
        x_max = 0
        bin = int((x_max - x_min)*4 + 1) 
        lambda_values = np.logspace(x_min, x_max, bin)  # Valores de lambda en escala logarítmica

        k_i = []
        for i in range(n):
            k_i.append((2*4**(n-(i+1)))**-1)

        results = []
        thetas = np.zeros(n)  # Inicializar con ceros
        structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min = structure_size(thetas)
        area = structure_size_x_max * structure_size_y_max
        results.append((0, structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min, area))
        for lambda_restriction in lambda_values:
            thetas = conjudate_gradient(thetas, k_i, lambda_restriction)
            structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min = structure_size(thetas)
            area = structure_size_x_max * structure_size_y_max
            results.append((lambda_restriction, structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min, area))

        # Imprimir los resultadospython main.py en .csv y con cabecera
        file_name = f"{n}_angles_output.csv"
        with open(RESULTS_PATH / file_name, 'w') as f:
            f.write("lambda,structure_size_x_max,structure_size_x_min,structure_size_y_max,structure_size_y_min,area\n")
            for result in results:
                f.write(f"{result[0]},{result[1]},{result[2]},{result[3]},{result[4]},{result[5]}\n")


