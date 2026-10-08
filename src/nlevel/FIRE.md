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
reposo uniforme `theta0=0.5*theta_star` y 34 fuerzas entre `1e-5` y `1e6`.
`main.py` usa ahora `ftol=1e-12`, `rtol=0` y refina también el punto de fuerza cero.
Cada fuerza parte de la solución anterior. Los CSV mantienen el formato existente.
FIRE es ahora el optimizador predeterminado; `--solver cg` permite seleccionar
la función anterior. La ejecución directa `python src/nlevel/main.py ...` también
está soportada. `--processes` controla los procesos y nunca se crean más que semillas.

## Equilibrios con tolerancia absoluta de 1e-12

Para las fuerzas altas se usa una segunda fase **FIRE**, con evaluación
trigonométrica y elástica en aritmética double-double. No es un cambio a
Newton ni una relajación del criterio. El estado se representa mediante dos
arrays float64, `theta_hi` y `theta_lo`, que conservan juntos la precisión.
La fase final está limitada a `|theta| <= 1.5` rad, rango que incluye los
barridos verificados. Fuera de él se informa un error explícito.

```python
result = fire_minimize(
    thetas, w_i, family_weights, rest_deformations, fuerza,
    ftol=1e-12, rtol=0, return_info=True,
)
theta_hi = result.thetas
theta_lo = result.theta_correction
print(result.gradient_max, result.rounded_gradient_max)

# Continuar en la siguiente fuerza conservando el estado preciso:
result = fire_minimize(
    theta_hi, w_i, family_weights, rest_deformations, siguiente_fuerza,
    theta_correction=theta_lo, ftol=1e-12, rtol=0, return_info=True,
)
```

`gradient_max` corresponde al par **hi+lo**. `rounded_gradient_max` muestra
qué sucede si se descarta la corrección. **No sumar los dos arrays en float64
para guardarlos como uno solo:** se perderían las cifras adicionales.
Por esa razón, las tolerancias menores que `1e-10` requieren `return_info=True`.
El valor por defecto de la función directa se mantiene en `1e-10` para conservar
la compatibilidad de las llamadas antiguas que esperan un único array.

`main.py` gestiona estos pares automáticamente y añade, para cada tamaño y
semilla, un archivo como `16384_seed45_fire_states.npz` con:

- `theta_hi` y `theta_lo`: configuraciones precisas de todos los puntos.
- `diagnostics` y `diagnostic_columns`: fuerza, residuo preciso, residuo sin
  corrección, tolerancia e iteraciones.
- Pesos, familias y deformaciones naturales usados en el cálculo.

Los CSV mantienen su formato y las longitudes para las gráficas se evalúan en
float64. El criterio de convergencia se evalúa siempre sobre el estado preciso.
El presupuesto `max_steps` se comparte entre las dos fases. La fase inicial
float64 usa hasta 1000 iteraciones antes de pasar al refinamiento cuando se
solicita una tolerancia menor que `1e-10`; no se añade un presupuesto oculto.

Se verificaron 4096 y 16384 ángulos, con las 34 fuerzas y el punto inicial,
a `ftol=1e-12`. Los puntos finales a fuerza `1e6` se contrastan también con
una evaluación independiente de 55 dígitos en `test_fire_precision.py`.

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
`max(abs(dH/dtheta)) <= 1e-10`, más estricto que el criterio RMS anterior.
El gradiente se evalúa con sumas compensadas y desplazamientos acumulados
que se recentran periódicamente. La convergencia se comprueba otra vez
sobre los ángulos finalmente devueltos. Recalcularlo con las sumas
ordinarias del método anterior puede diferir por redondeo en sistemas grandes.
`rtol=0` significa que no se relaja automáticamente la tolerancia a fuerzas grandes.
Si se activa `rtol`, se añade `rtol*max(abs(lambda*dX/dtheta_inicial))` a `ftol`.

Se lanza `RuntimeError` si no converge. Para estudiar un fallo sin interrumpir
un barrido de diagnóstico, usar conjuntamente `return_info=True` y
`raise_on_failure=False`, y comprobar `result.converged` antes de continuar.

## Parámetros del integrador

| Parámetro | Valor |
|---|---:|
| `dt` | 0.1 |
| `dt_max` | 0.1 |
| `alpha_start` | 0.1 |
| `finc` | 1.1 |
| `fdec` | 0.5 |
| `falpha` | 0.99 |
| `n_min` | 5 |
| `max_step` | 0.05 rad por ángulo |
| `max_steps` | 50000 |
| `ftol` | 1e-12 en main.py; 1e-10 en llamadas directas |
| `precondition` | True |

Estos valores priorizan una tolerancia más exigente y pasos pequeños.
El refinamiento se activa automáticamente cuando la tolerancia efectiva es menor que `1e-10`.
El ajuste rápido anterior era `dt=dt_max=1`, `ftol=1e-7`.
Para configurar la simulación explícitamente:

```bash
python -m src.nlevel.main --sizes 4096 16384 --solver fire --ftol 1e-12 --fire-dt 0.1 --fire-dt-max 0.1 --fire-max-steps 50000 --output-dir src/nlevel/results_fire_precision
```

Una tolerancia solicitada no garantiza convergencia para cualquier configuración
o presupuesto de iteraciones. El código interrumpe con un
error si no converge; no aumenta la tolerancia ni acepta ese estado en silencio.
Los archivos de resultados anteriores no se regeneran automáticamente.

## Corrección de precisión (octubre de 2026)

Reducir únicamente el paso no bastaba: el barrido podía estancarse por
cancelación numérica. Se corrigieron la recuperación de los ángulos, las
sumas de residuos elásticos y la evaluación de la longitud y su gradiente.
FIRE recentra las variables cada 200 iteraciones y antes de aceptar la
convergencia. No se cambió la energía ni se relajó la tolerancia.

Se verificó el barrido completo de **34 fuerzas de 1e-5 a 1e6**, tanto con
4096 como con 16384 ángulos, `c=0.5`, ambas familias y pesos unitarios:
todos los puntos satisfacen `max|grad| <= 1e-10`, con `rtol=0`.
La regresión está incluida en `test_fire.py`; comprueba también el residuo
recalculado sobre los ángulos devueltos. Esto cubre esta configuración,
no garantiza convergencia para cualquier fuerza o familia.
Los errores incluyen ahora el tamaño, la fuerza y el residuo alcanzado.

## Medición histórica del ajuste rápido

Los datos siguientes corresponden a `dt=dt_max=1`, **ftol=1e-7**, no a los
nuevos valores de precisión. Fue el mejor entre nueve perfiles ensayados. El paso temporal corresponde a la
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

Para ángulos pequeños, la longitud y su gradiente usan expresiones
trigonométricas equivalentes con sumas y productos compensados y acumulación
logarítmica del producto de cosenos. Fuera de esa región se mantiene la
recurrencia y su diferenciación inversa, sin dividir por cosenos cercanos a cero. El bucle FIRE está
compilado con Numba. No se forman matrices N por N.

El algoritmo de mezcla de velocidades y adaptación temporal sigue
[Bitzek et al., Structural Relaxation Made Simple (2006)](https://www.math.uni-bielefeld.de/~gaehler/papers/fire_prl.pdf),
con integración de Euler semiimplícita y el precondicionamiento descrito arriba.
`precondition=False` desactiva la masa aproximada, pero mantiene las coordenadas
acumuladas; en ese caso conviene reducir `dt` y `dt_max`.

## Pruebas

```bash
python -m unittest src.nlevel.test_fire_precision src.nlevel.test_fire src.nlevel.test_dynamic -v
```

Incluyen el objetivo y su gradiente contra la implementación existente, diferencias
finitas, comparación con BFGS en sistemas pequeños, entradas inválidas, fallos
explícitos y convergencia con 4096 y 16384 variables. La convergencia del gradiente
identifica un punto estacionario de la rama seguida; no certifica un mínimo global.
