from pathlib import Path
import numpy as np
from dynamic import conjudate_gradient, structure_size
from secon_order_dynamic import theta_second_order_calculation

PATH = Path(__file__).resolve().parent
RESULTS_PATH = PATH / 'results'
if __name__ == "__main__":
    # Ángulos iniciales del sistema
    n = 1024.  # Número de ángulos
    thetas = np.zeros(n)  # Inicializar con ceros
    
    # Rango de valores para lambda_restriction
    x_min = -2.0
    x_max = 2.0
    bin = int((x_max - x_min)*4 + 1) 
    lambda_values = np.logspace(x_min, x_max, bin)  # Valores de lambda en escala logarítmica

    results = []

    for lambda_restriction in lambda_values:
        thetas = theta_second_order_calculation(n, lambda_restriction)
        structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min = structure_size(thetas)
        area = structure_size_x_max * structure_size_y_max
        results.append((lambda_restriction, structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min, area))

    # Imprimir los resultadospython main.py en .csv y con cabecera
    file_name = f"{n}_angles_output.csv"
    with open(RESULTS_PATH / file_name, 'w') as f:
        f.write("lambda,structure_size_x_max,structure_size_x_min,structure_size_y_max,structure_size_y_min,area\n")
        for result in results:
            f.write(f"{result[0]},{result[1]},{result[2]},{result[3]},{result[4]},{result[5]}\n")
