# Fractal Network — Elasticidad auxética jerárquica

Este repositorio contiene modelos numéricos y herramientas de visualización para estudiar la respuesta mecánica de una estructura auxética jerárquica. La estructura se construye recursivamente a partir de cuadrados rotables: cada nivel añade una nueva escala geométrica y un ángulo de deformación.

El objetivo principal es calcular cómo una carga adimensional (`lambda`) modifica los ángulos de los niveles, el tamaño global de la red y la distribución de energía elástica.

La formulación teórica principal está en [`documentation/Hierarchical_Auxetic_Elasticity.tex`](documentation/Hierarchical_Auxetic_Elasticity.tex).

## Contenido

```text
.
├── documentation/                # Derivaciones y documentos de referencia
├── src/
│   ├── 1D/                       # Modelo espectral unidimensional
│   ├── 1level/                   # Visualización de una red de un nivel
│   └── nlevel/                   # Modelo principal de N niveles
│       ├── dynamic.py             # Energía, geometría y minimización iterativa
│       ├── second_order_dynamic.py# Solución recursiva para la red ideal
│       ├── second_order_dynamic_noisy.py
│       │                          # Variante con ángulos iniciales desordenados
│       ├── plot_nlevels.py        # Construcción y dibujo de la estructura
│       ├── main.py                # Barrido de carga para varios tamaños
│       └── results/               # CSV generados y scripts de análisis
└── *.gif                         # Animaciones de ejemplo
```

## Modelo

Para una red con `N` niveles, cada nivel tiene un ángulo `theta_i` y una rigidez `k_i`. La energía elástica pondera los niveles por su multiplicidad geométrica, proporcional a `4^(N-i)`. La carga externa se representa mediante `lambda`.

El solucionador de segundo orden usa una recurrencia para expresar los ángulos como proporcionales a `theta_1`; posteriormente fija la amplitud mediante la condición del último nivel. Con los ángulos resultantes se calculan:

- extensiones máximas y mínimas en `x` e `y`;
- área característica de la estructura;
- promedio y promedio cuadrático de los ángulos;
- energía almacenada en cada nivel.

La variante ruidosa incorpora un estado de referencia `delta_i`, que modela desviaciones geométricas iniciales.

## Requisitos

Se recomienda Python 3.10 o posterior. Instala las dependencias con:

```bash
python3 -m pip install numpy matplotlib scipy pandas numba
```

## Uso

Los módulos emplean importaciones locales, por lo que conviene ejecutar los comandos desde el directorio correspondiente.

### Generar resultados para redes ideales

Desde `src/nlevel`:

```bash
cd src/nlevel
python3 main.py
```

Esto recorre tamaños de `4` a `128` ángulos y guarda archivos CSV en `src/nlevel/results/`. Cada archivo incluye `lambda`, tamaños geométricos, área y estadísticos de los ángulos.

### Visualizar una estructura jerárquica

```bash
cd src/nlevel
python3 plot_nlevels.py
```

### Visualizar la solución de segundo orden

```bash
cd src/nlevel
python3 second_order_dynamic.py
```

### Graficar los CSV generados

```bash
cd src/nlevel/results
python3 plot_output_f-e.py
python3 plot_output_theta-f.py
```

### Ejemplos simples

```bash
python3 src/1D/main.py
python3 src/1level/plot_network_1level.py
```

## Resultados disponibles

El repositorio ya incluye barridos para 4, 8, 16, 32, 64 y 128 ángulos en `src/nlevel/results/`. Los scripts de análisis convierten `lambda` a una fuerza efectiva mediante `f = lambda / Lx_max` y permiten estudiar relaciones fuerza–deformación y fuerza–ángulos.

## Estado actual y notas

- La rama ideal (`main.py` y `second_order_dynamic.py`) es el flujo principal de cálculo.
- `main_noisy.py` está en desarrollo: actualmente importa el solucionador ideal y no coincide con su valor de retorno. Para usar el caso ruidoso debe conectarse con `second_order_dynamic_noisy.py`.
- Los archivos `__pycache__`, `.DS_Store` y los GIF son artefactos locales o visuales; se recomienda ignorarlos mediante `.gitignore` si el repositorio se va a compartir.
- Los valores de prefactores y convenciones de índices en la recurrencia deben contrastarse con el documento teórico antes de publicar resultados finales.

## Referencias internas

- `documentation/Hierarchical_Auxetic_Elasticity.tex`: desarrollo del modelo auxético jerárquico.
- `documentation/Noisy_Elastica_v1.pdf` y `documentation/Noisy_Elastica _v2.pdf`: notas sobre la extensión con desorden.
- `documentation/Non_hookean_mechanics_of_random_slender_chains.tex`: contexto sobre respuesta no hookeana y filtrado geométrico multiescala.

