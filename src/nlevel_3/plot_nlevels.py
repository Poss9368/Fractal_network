import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

square_color = '#4DD0E1'  # color del cuadrado
centers = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]])

def init_rotated_square(center, size, angle):
    """crea un cuadrado rotado alrededor de un centro dado."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    half_size = size/2
    square: np.ndarray = np.array([
        [-half_size, -half_size],
        [ half_size, -half_size],
        [ half_size,  half_size],
        [-half_size,  half_size],
        [-half_size, -half_size]
    ])
    rotated_square: np.ndarray = (R @ square.T).T + center
    return rotated_square

def compute_center_position(x, y, theta):
    """Calcula la nueva posición del centro del cuadrado según su rotación."""
    temp_x = rotate_point(np.array([x, 0]),  -theta)
    temp_y = rotate_point(np.array([0, y]),   theta)
    x = temp_x[0] + temp_y[0]
    y = temp_x[1] + temp_y[1]
    return np.array([x, y])

def rotate_point(point, angle):
    """Rota un punto en 2D dado un ángulo en radianes."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return R @ point

def traslate_point(point, translation):
    """Traslada un punto en 2D por un vector de traslación."""
    return point + translation

def rotate_array_of_points(points: np.ndarray, angle):
    """Rota un array de puntos en 2D dado un ángulo en radianes."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return points @ R.T

def traslate_array_of_points(points: np.ndarray, translation):
    """Traslada un array de puntos en 2D por un vector de traslación."""
    return points + translation

# function to make the structure
def draw_level_0(square_size, angle):
    structures = []
    for c in centers:
        unique_square = init_rotated_square([0, 0], square_size , 0) #cuadrado BASE sin rotar y sin desplazar
        p = -c[0]*c[1] # polaridad del la rotación
        theta = angle*0.5*p # rotación del cuadrado 
        center = c * square_size * 0.5 # centro del cuadro si no estuviera rotado
        traslation  = compute_center_position(center[0], center[1], theta) # centro del cuadrado rotado 
        unique_square = rotate_array_of_points(unique_square, theta) # rotar el cuadrado
        unique_square = traslate_array_of_points(unique_square, traslation) # desplazar el cuadrado
        structures.append(unique_square)
    
    #porte de la nueva estructura
    structure_size_y_max     = 2*square_size/np.cos(angle*0.5) + 2*square_size*np.sin(angle*0.5)*(1 - np.tan(angle*0.5)) 
    structure_size_x_min     = 2*square_size*np.sin(angle*0.5)

    structure_size_x_max     = 2*square_size/np.cos(angle*0.5) + 2*square_size*np.sin(angle*0.5)*(1- np.tan(angle*0.5)) 
    structure_size_y_min     = 2*square_size*np.cos(angle*0.5)

    return structures, structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min

def draw_level_n(structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min, angle, init_structure):
    structures = []
    for c in centers:
        p = -c[0]*c[1]
        theta = angle*0.5*p
        new_structure = rotate_array_of_points(init_structure, theta)
        dispacement_x = structure_size_x_max * c[0] * 0.5
        dispacement_y = structure_size_y_min * c[1] * 0.5
        displacement = compute_center_position(dispacement_x, dispacement_y, theta)
        new_structure = traslate_array_of_points(new_structure, displacement)
        structures.append(new_structure)
    
    structures = np.concatenate(structures)

    #porte de la nueva estructura
    structure_size_y_max = 2*structure_size_y_min/np.cos(angle*0.5) + 2*np.sin(angle*0.5)*(structure_size_x_max - structure_size_y_min*np.tan(angle*0.5)) 
    structure_size_x_min = 2*structure_size_y_min*np.sin(angle*0.5)

    structure_size_x_max = 2*structure_size_x_max/np.cos(angle*0.5) + 2*np.sin(angle*0.5)*(structure_size_y_min - structure_size_x_max*np.tan(angle*0.5)) 
    structure_size_y_min = 2*structure_size_y_min*np.cos(angle*0.5)

    return structures, structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min

def draw_total_levels(square_size: float, angles:np.ndarray):
    """Dibuja la estructura total de niveles."""
    # primer nivel
    structures, structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min = draw_level_0(square_size, angles[0])
    # niveles superiores
    for i in range(1, len(angles)):
        structures, structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min = draw_level_n(structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min, angles[i], structures)
    return structures, structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min

# function to print the structure and fill it with color 
def print_square(ax, structure):
    for square in structure:
        ax.fill(square[:, 0], square[:, 1], color=square_color, edgecolor='black')

def pint_box(ax, size_x, size_y):
    """Dibuja un rectángulo en el gráfico."""
    size_x = size_x * 0.5
    size_y = size_y * 0.5
    box = np.array([
        [-size_x, -size_y],
        [ size_x, -size_y],
        [ size_x,  size_y],
        [-size_x,  size_y],
        [-size_x, -size_y]
    ])
    ax.fill(box[:, 0], box[:, 1], color='none', edgecolor='black')

def print_structure(thetas: np.ndarray, ax ,square_size: float = 1.0):
    structure, structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min = draw_total_levels(square_size, thetas)
    n = len(thetas)
    ax.set_aspect('equal')
    ax.axis('off')    
    ax.set_xlim(-(2)**(n-1)*1.9*square_size, (2)**(n-1)*1.9*square_size)
    ax.set_ylim(-(2)**(n-1)*1.9*square_size, (2)**(n-1)*1.9*square_size)
    print_square(ax, structure)
    pint_box(ax, structure_size_x_max, structure_size_y_max)
    return structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min
 
def ejample_animation():
    # Parámetros
    angles = np.array([np.pi/8, np.pi/4])  # ángulos de rotación
    square_size = 1.0
    num_frames = 100  # Número de frames para la animaci
    n_angles = len(angles)
    # Configurar la figura
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axis('off')

    # Función de actualización para la animación
    def update(frame):
        ax.clear()  # Limpiar el gráfico
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim(-(2)**(n_angles-1)*2, (2)**(n_angles-1)*2)
        ax.set_ylim(-(2)**(n_angles-1)*2, (2)**(n_angles-1)*2)
        factor = frame / (num_frames - 1)  # Valor entre 0 y 1
        scaled_angles = angles * factor  # Escalar los ángulos
        structure, structure_size_x_max, structure_size_x_min, structure_size_y_max, structure_size_y_min = draw_total_levels(square_size, scaled_angles)
        print_square(ax, structure)
        pint_box(ax, structure_size_x_max, structure_size_y_max)
        sum_angles = np.sum(scaled_angles)
        print(f"sum angles {sum_angles * 180/np.pi} -  size_x {structure_size_x_max} - size_y {structure_size_y_min}")

    # Crear la animación
    anim = FuncAnimation(fig, update, frames=num_frames, interval=50 )

    # Guardar la animación como un archivo de video (opcional)
    #anim.save('fractal_auxetic_evolution.mp4', writer='ffmpeg', fps=20)

    # Mostrar la animación
    plt.show()

if __name__ == '__main__':
    ejample_animation()

