from pathlib import Path
import numpy as np
from old.nlevel.dynamic import structure_size
from old.nlevel.second_order_dynamic import potential_by_level, theta_second_order_calculation

PATH = Path(__file__).resolve().parent
RESULTS_PATH = PATH / 'results'


if __name__ == "__main__":
    for i in range(2, 8):
        # Ángulos iniciales del sistema
        n = 2 ** i
        thetas = np.zeros(n)  # Inicializar con ceros
        
        # Rango de valores para lambda_restriction
        x_min = -5.0
        x_max = 2.0
        bin = int((x_max - x_min)*4 + 1) 

        lambda_values = np.logspace(x_min, x_max, bin)  # Valores de lambda en escala logarítmica

        k_i = []
        for i in range(n):
            k_i.append(1.0)  

        results = []

        thetas = np.zeros(n)  # Inicializar con ceros
        thetas_mean = np.mean(thetas)
        thetas_2_mean = np.mean(thetas**2)
        structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min = structure_size(thetas)
        area = structure_size_x_max * structure_size_y_max
        results.append((0, structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min, area, thetas_mean, thetas_2_mean))
        for lambda_restriction in lambda_values:
            thetas, S, Q, Ak, theta_1, theta_N = theta_second_order_calculation(n, lambda_restriction, k_i)
            thetas_mean = np.mean(thetas)
            thetas_2_mean = np.mean(thetas**2)
            structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min = structure_size(thetas)
            area = structure_size_x_max * structure_size_y_max
            results.append((lambda_restriction, structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min, area, thetas_mean, thetas_2_mean))

        # Imprimir los resultadospython main.py en .csv y con cabecera
        file_name = f"{n}_angles_output.csv"
        with open(RESULTS_PATH / file_name, 'w') as f:
            f.write("lambda,structure_size_x_max,structure_size_x_min,structure_size_y_max,structure_size_y_min,area,thetas_mean,thetas_2_mean\n")
            for result in results:
                f.write(f"{result[0]},{result[1]},{result[2]},{result[3]},{result[4]},{result[5]},{result[6]},{result[7]}\n")



        # k_i = []
        # for i in range(n):
        #     k_i.append(1.0)  # Puedes ajustar este valor según tus necesidades, o generar una lista de k_i con diferentes valores


        # results = []
        # # largo inicial del sistema
        # thetas_mean = np.mean(thetas)
        # thetas_2_mean = np.mean(thetas**2)
        # structure_size_x_max_0, structure_size_x_min_0, structure_size_y_max_0, structure_size_y_min_0 = structure_size(thetas)
        # area_0 = structure_size_x_max_0 * structure_size_y_max_0
        # results.append((0, structure_size_x_max_0, structure_size_x_min_0, structure_size_y_max_0, structure_size_y_min_0, area_0, thetas_mean, thetas_2_mean))
        # energy_by_levels = potential_by_level(np.asarray(thetas, dtype=float), np.asarray(k_i, dtype=float))
        # print("Energy by levels (lambda=0):", energy_by_levels)
        # #
        # for lambda_restriction in lambda_values:
        #     thetas= theta_second_order_calculation(n, lambda_restriction, np.asarray(k_i, dtype=float) )
        #     #thetas_mean = np.mean(thetas)
        #     #thetas_2_mean = np.mean(thetas**2)
        #     structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min = structure_size(thetas)
        #     area = structure_size_x_max * structure_size_y_max
        #     results.append((lambda_restriction, structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min, area, thetas_mean, thetas_2_mean))

        # # Imprimir los resultadospython main.py en .csv y con cabecera  
        # file_name = f"{n}_angles_output.csv"
        # with open(RESULTS_PATH / file_name, 'w') as f:
        #     f.write("lambda,structure_size_x_max,structure_size_x_min,structure_size_y_max,structure_size_y_min,area,thetas_mean,thetas_2_mean\n")
        #     for result in results:
        #         f.write(f"{result[0]},{result[1]},{result[2]},{result[3]},{result[4]},{result[5]},{result[6]},{result[7]}\n")
