import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

square_color = '#4DD0E1'  # rojo coral

def draw_rotated_square(center, size, angle):
    """Dibuja un cuadrado rotado dado un centro, tamaño y ángulo en radianes."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    half_size = size / 2
    square = np.array([
        [-half_size, -half_size],
        [ half_size, -half_size],
        [ half_size,  half_size],
        [-half_size,  half_size],
        [-half_size, -half_size]
    ])
    rotated_square = (R @ square.T).T + center
    return rotated_square

def compute_center_position(i, j, size, angle):
    """Calcula la nueva posición del centro del cuadrado según su rotación."""
    spacing = size * np.cos(angle) + size * np.sin(angle)
    x = j * spacing
    y = i * spacing
    return np.array([x, y])

def update(frame):
    """Actualiza la animación en cada frame."""
    ax.clear()
    angle = angles[frame]
    for i in range(n_rows):
        for j in range(n_cols):
            sign = (-1) ** (i + j)
            theta = sign * angle
            center = compute_center_position(i, j, size, angle)
            square = draw_rotated_square(center, size, theta)
            ax.fill(square[:, 0], square[:, 1], color=square_color, edgecolor='black')

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Ángulo: {np.degrees(angle):.1f}°")

    # Fijar límites para mantener escala constante
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


if __name__ == '__main__':
    # Parámetros
    n_rows = 4
    n_cols = 4
    size = 1.0
    n_frames = 60
    max_angle = np.pi / 4  # hasta 45°

    # Generar una secuencia de ángulos desde 0 hasta max_angle
    angles = np.linspace(0, max_angle, n_frames)

    # Precalcular límites fijos del gráfico
    # Usamos la separación máxima posible entre centros
    max_spacing = size * (np.cos(max_angle) + np.sin(max_angle))
    x_max = max_spacing * (n_cols - 1) + size
    y_max = max_spacing * (n_rows - 1) + size
    x_min = -size
    y_min = -size

    # Crear figura y eje
    fig, ax = plt.subplots(figsize=(8, 8))

    # Crear animación
    anim = FuncAnimation(fig, update, frames=n_frames, interval=100, repeat=False)

    plt.show()
