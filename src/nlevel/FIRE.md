# Minimización FIRE para sistemas grandes

`dynamic.fire_minimize` acepta los mismos cinco argumentos que
`conjudate_gradient`. Minimiza el potencial a **fuerza impuesta**
`H = E - lambda_restriction * X`, con la energía y los pesos actuales del
código. No modifica el archivo LaTeX ni cambia la normalización de la energía.

En este modelo `n` cuenta los **ángulos o niveles jerárquicos**, no cuadrados
individuales. Las pruebas de 4096 y 16384 se refieren a ese número de variables.

## Ejecutar la simulación

Desde la raíz del repositorio:

```bash
python -m src.nlevel.main --sizes 4096 16384 --solver fire --output-dir src/nlevel/results_fire
```

Esto conserva la configuración de `main.py`: `w_i=1`, ambas familias completas,
reposo uniforme `theta0=0.5*theta_star` y 31 fuerzas entre `1e-5` y `1e5`.
Cada fuerza parte de la solución anterior. Los CSV mantienen el formato existente.
FIRE es ahora el optimizador predeterminado; `--solver cg` permite seleccionar
la función anterior. La ejecución directa `python src/nlevel/main.py ...` también
está soportada. `--processes` controla los procesos y nunca se crean más que semillas.

## Usarlo directamente

```python
from src.nlevel.dynamic import fire_minimize

thetas = fire_minimize(
    thetas, w_i, family_weights, rest_deformations, lambda_restriction,
)
```

Para inspeccionar la convergencia:

```python
result = fire_minimize(
    thetas, w_i, family_weights, rest_deformations, lambda_restriction,
    return_info=True,
)
thetas = result.thetas
print(result.iterations, result.gradient_max, result.seconds)
```

La función copia los ángulos de entrada. Por defecto exige
`max(abs(dH/dtheta)) <= 1e-7`, más estricto que el criterio RMS anterior.
El gradiente se evalúa mediante desplazamientos acumulados, para reducir las
pérdidas de precisión al restar sumas grandes. Recalcularlo con las sumas
ordinarias del método anterior puede diferir por redondeo en sistemas grandes.
`rtol=0` significa que no se relaja automáticamente la tolerancia a fuerzas grandes.
Si se activa `rtol`, se añade `rtol*max(abs(lambda*dX/dtheta_inicial))` a `ftol`.

Se lanza `RuntimeError` si no converge. Para estudiar un fallo sin interrumpir
un barrido de diagnóstico, usar conjuntamente `return_info=True` y
`raise_on_failure=False`, y comprobar `result.converged` antes de continuar.

## Parámetros elegidos

| Parámetro | Valor |
|---|---:|
| `dt` | 1.0 |
| `dt_max` | 1.0 |
| `alpha_start` | 0.1 |
| `finc` | 1.1 |
| `fdec` | 0.5 |
| `falpha` | 0.99 |
| `n_min` | 5 |
| `max_step` | 0.05 rad por ángulo |
| `max_steps` | 50000 |
| `ftol` | 1e-7 |
| `precondition` | True |

Son los mejores parámetros **entre los nueve perfiles ensayados para esta
configuración**, no un óptimo universal. El paso temporal corresponde a la
dinámica ficticia precondicionada; no es el `step_size` del método anterior.

Medición local, secuencial y sin incluir la primera compilación Numba:

| Variables | Fuerzas convergidas | Iteraciones totales | Tiempo total |
|---:|---:|---:|---:|
| 4096 | 31/31 | 1308 | 0.162 s |
| 16384 | 31/31 | 1208 | 0.536 s |

Los tiempos dependen del equipo y excluyen escritura de CSV y compilación inicial.
El informe de todos los perfiles está en `fire_benchmark_summary.json`.
No se midió una aceleración respecto del antiguo CG: la comparación de parámetros
es entre variantes FIRE con el mismo precondicionamiento.

Para repetir la selección:

```bash
python -m src.nlevel.benchmark_fire --sizes 4096 16384 --output /tmp/fire_benchmark.json
```

El barrido descarta una continuación cuando un punto falla y registra ese fallo;
no lo cuenta como una solución válida. Se pueden seleccionar perfiles con
`--profiles unit_step standard`, cambiar `--fraction` y limitar `--max-steps`.

## Cómo se consigue el escalamiento

Se usan sumas de desplazamientos `s_i=sum_{j<=i}(theta_j-theta_ref_j)`.
Las deformaciones de bisagra son locales en estas variables: sus variaciones
son `s_i-2*s_(i-1)` para phi- y `s_i` para phi+. La matriz elástica resulta
tridiagonal. Una masa ficticia positiva aproxima esa curvatura y parte de la
curvatura geométrica; una factorización bidiagonal permite transformar fuerzas
y velocidades en O(N). El potencial y la longitud no se aproximan.

La longitud y su gradiente se calculan mediante recurrencia y diferenciación
inversa, evitando divisiones por cosenos cercanos a cero. El bucle FIRE está
compilado con Numba. No se forman matrices N por N.

El algoritmo de mezcla de velocidades y adaptación temporal sigue
[Bitzek et al., Structural Relaxation Made Simple (2006)](https://www.math.uni-bielefeld.de/~gaehler/papers/fire_prl.pdf),
con integración de Euler semiimplícita y el precondicionamiento descrito arriba.
`precondition=False` desactiva la masa aproximada, pero mantiene las coordenadas
acumuladas; en ese caso conviene reducir `dt` y `dt_max`.

## Pruebas

```bash
python -m unittest src.nlevel.test_fire src.nlevel.test_dynamic -v
```

Incluyen el objetivo y su gradiente contra la implementación existente, diferencias
finitas, comparación con BFGS en sistemas pequeños, entradas inválidas, fallos
explícitos y convergencia con 4096 y 16384 variables. La convergencia del gradiente
identifica un punto estacionario de la rama seguida; no certifica un mínimo global.
