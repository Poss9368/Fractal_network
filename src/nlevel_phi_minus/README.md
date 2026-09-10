# Ensayo jerárquico con elasticidad sólo en las bisagras phi−

Este experimento conserva un ángulo `theta_i` por nivel y la coordenada horizontal recursiva del modelo reducido. La configuración natural es `delta_i = 0`; las bisagras `phi+` tienen rigidez cero. Compara dos cálculos a la **misma fuerza impuesta**: la solución del Hamiltoniano cuadrático y una minimización con la geometría trigonométrica sin expandir.

La carpeta anterior `src/nlevel` y sus resultados permanecen independientes. Este ensayo permite buscar una respuesta no hookeana; no presupone una ley cuadrática fuerza–deformación.

## Idea central: comparación con la cadena discreta 1D

Los dos modelos pueden escribirse con la misma estructura algebraica después de escoger la variable geométrica adecuada. Esto no los hace físicamente idénticos: sus matrices elásticas, estados de referencia y normalizaciones son diferentes. La comparación sirve para identificar el mecanismo que podría producir una respuesta no hookeana.

### Cadena 1D

La cadena tiene `N` eslabones rígidos de longitud `a`. Sus ángulos tangentes actuales y naturales son `theta_j` y `theta_j^(0)`. La energía exacta a fuerza horizontal `F` es

\[
\mathcal H_{1D}=
\frac{\kappa}{2}\sum_j
\left[(\theta_j-\theta_{j-1})-
(\theta_j^{(0)}-\theta_{j-1}^{(0)})\right]^2
-Fa\sum_j\cos\theta_j.
\]

El cuadrado se aplica a la deformación de cada bisagra. Definiendo el operador de diferencia `(D theta)_j=theta_j-theta_(j-1)` y expandiendo el coseno para ángulos pequeños,

\[
\mathcal H_{1D}^{(2)}=
\frac12(\boldsymbol\theta-\boldsymbol\theta^{(0)})^{\mathsf T}
B_{1D}(\boldsymbol\theta-\boldsymbol\theta^{(0)})
+\frac{g}{2}\boldsymbol\theta^{\mathsf T}\boldsymbol\theta
+\text{constante},
\]

\[
B_{1D}=\kappa D^{\mathsf T}D,\qquad g=Fa.
\]

Con condiciones periódicas, `B_1D` es el laplaciano discreto

\[
B_{1D}=\kappa
\begin{pmatrix}
2&-1&0&\cdots&-1\\
-1&2&-1&\cdots&0\\
\vdots&&\ddots&&\vdots\\
-1&0&\cdots&-1&2
\end{pmatrix}.
\]

La base de Fourier lo diagonaliza porque todos los sitios son equivalentes por traslación. Sus rigideces son

\[
b_n^{1D}=4\kappa\sin^2\left(\frac{\pi n}{N}\right).
\]

El modo cero representa una rotación rígida y tiene rigidez nula. Se elimina fijando el ángulo medio a cero. La ecuación de equilibrio es

\[
(B_{1D}+gI)\boldsymbol\theta=B_{1D}\boldsymbol\theta^{(0)}.
\]

En la base propia, cada amplitud se filtra según

\[
\theta_n(g)=\frac{b_n^{1D}}{b_n^{1D}+g}\,\theta_n^{(0)}.
\]

La longitud proyectada es `X=a sum_j cos(theta_j)`. Por ello,

\[
\varepsilon_{1D}
\simeq\frac{1}{2N}\left(
\|\boldsymbol\theta^{(0)}\|^2-\|\boldsymbol\theta\|^2
\right).
\]

El factor `1/N` aparece porque el alargamiento absoluto y la longitud de referencia son ambos extensivos en el número de eslabones: `X_0` es aproximadamente `Na`.

### Jerarquía con sólo phi−

Para el nivel `i`,

\[
S_i=\sum_{m<i}\theta_m,\qquad
\phi_i^-=\theta_i-S_i.
\]

Con `delta_i=0`, `phi_i^(0)=0` y rigidez nula para `phi+`, la energía es

\[
E_-(\theta)=2\sum_i q_i(\theta_i-S_i)^2,
\qquad q_i=4^{N-i}k_i^-.
\]

Introducimos la variable que hace visible la analogía:

\[
\boxed{\alpha_i=1-\frac{\theta_i}{2}},\qquad
u_i=\alpha_i-1=-\frac{\theta_i}{2}.
\]

Sea `M` la matriz triangular que calcula

\[
(M\mathbf u)_i=u_i-\sum_{m<i}u_m.
\]

Como `theta_i-S_i=-2(Mu)_i`, la energía se transforma exactamente en

\[
E_-(\alpha)=
\frac12(\boldsymbol\alpha-\mathbf1)^{\mathsf T}
B_\alpha(\boldsymbol\alpha-\mathbf1),
\]

\[
\boxed{B_\alpha=16M^{\mathsf T}\operatorname{diag}(q_i)M}.
\]

La longitud recursiva correcta tiene la expansión

\[
X\simeq1+\frac12\sum_i\theta_i-\frac18\sum_i\theta_i^2.
\]

Al sustituir `theta_i=2(1-alpha_i)`, obtenemos

\[
X\simeq1+\frac N2-\frac12\sum_i\alpha_i^2,
\]

\[
\boxed{\varepsilon_{\rm jer}=X-1
\simeq\frac12\left(N-\|\boldsymbol\alpha\|^2\right)}.
\]

Aquí no aparece `1/N`: `N` cuenta niveles que actúan multiplicativamente sobre una única dimensión global, en vez de segmentos aditivos de una cadena. Si se quisiera comparar una deformación por nivel podría definirse `epsilon_jer/N`, pero ésa sería otra observable y no la deformación macroscópica `(X-X_0)/X_0`.

El potencial a fuerza horizontal fija, descartando una constante independiente de los ángulos, queda

\[
\boxed{
\mathcal H_{\rm jer}^{(2)}=
\frac12(\boldsymbol\alpha-\mathbf1)^{\mathsf T}
B_\alpha(\boldsymbol\alpha-\mathbf1)
+\frac f2\boldsymbol\alpha^{\mathsf T}\boldsymbol\alpha}.
\]

Su equilibrio es

\[
(B_\alpha+fI)\boldsymbol\alpha=B_\alpha\mathbf1.
\]

En términos de `theta=2(1-alpha)`, la misma ecuación es

\[
(B_\alpha+fI)\boldsymbol\theta=2f\mathbf1.
\]

### Forma común y diferencia física

Los dos Hamiltonianos cuadráticos tienen la forma

\[
\boxed{
\mathcal H(\mathbf z)=
\frac12(\mathbf z-\mathbf z^{(0)})^{\mathsf T}
B(\mathbf z-\mathbf z^{(0)})
+\frac f2\mathbf z^{\mathsf T}\mathbf z}.
\]

La identificación es

| Modelo | Variable `z` | Referencia `z^(0)` | Matriz elástica |
|---|---:|---:|---:|
| Cadena 1D | `theta` | `theta^(0)` | `kappa D^T D` |
| Jerarquía phi− | `alpha` | vector de unos | `16 M^T diag(q_i) M` |

Toda matriz real simétrica `B` admite una base ortonormal de autovectores `v_p`, con `B v_p=b_p v_p`. En esa base,

\[
z_p(f)=\frac{b_p}{b_p+f}z_p^{(0)}.
\]

Por tanto, ambos problemas poseen el mismo filtro racional por modo. Lo que cambia es qué modos existen y cuánto pesa el estado de referencia en cada uno. Para la jerarquía,

\[
P_p=(v_p^{\mathsf T}\mathbf1)^2,
\]

y la deformación cuadrática es

\[
\boxed{
\varepsilon_{\rm jer}(f)=
\frac12\sum_pP_p
\left[1-\left(\frac{b_p}{b_p+f}\right)^2\right]}.
\]

En la cadena aparece la expresión análoga, además del factor `1/N`, con los pesos determinados por el espectro de la geometría natural `theta^(0)`.

Para cualquier sistema finito con rigideces positivas, la expansión a fuerza pequeña es lineal:

\[
\varepsilon(f)\simeq
f\sum_p\frac{P_p}{b_p}
\quad\text{(salvo la normalización propia de cada modelo)}.
\]

Un régimen no hookeano amplio requiere muchos modos con escalas `b_p` distribuidas y pesos `P_p` apreciables. Un solo modo extremadamente blando puede reducir mucho la ventana hookeana y producir una transición de rigidez, pero no basta para demostrar una ley de potencia extendida. Esta distinción es el objetivo principal de los archivos `spectrum.csv`, `curves.csv` y de la pendiente local exportada por la simulación.

Hay además una diferencia conceptual útil. En 1D se necesita una geometría natural no recta, `theta^(0) != 0`, para disponer de holgura angular que la tensión pueda filtrar. En la jerarquía usamos `delta=0`, pero el estado natural de la variable transformada es `alpha^(0)=1`; por eso la carga sí tiene una referencia no nula sobre la cual actuar.

## Corrección de la longitud jerárquica

La fórmula `prod_i[sin(theta_i/2)+cos(theta_i/2)]` usada en el documento original no sigue las recurrencias de las cuatro longitudes para más de un nivel. Las recurrencias conducen a

\[
\boxed{
X(\theta)=
\left[\prod_i\cos(\theta_i/2)\right]
\left[1+\sum_i\tan(\theta_i/2)\right]}.
\]

Para dos niveles, la recurrencia contiene `c1*c2+s1*c2+c1*s2`; la fórmula antigua añade indebidamente `s1*s2`. La implementación evalúa `X` mediante recurrencias en senos y cosenos, de modo que no introduce singularidades artificiales cuando algún coseno se anula.

## Qué se conserva del trabajo anterior

- `../nlevel/main.py` organiza barridos, llama a `second_order_dynamic.py` y calcula las longitudes con `dynamic.structure_size`.
- `../nlevel/second_order_dynamic.py` combina una recurrencia aproximada con una raíz trigonométrica para el último ángulo. Su parámetro es `lambda = f L`.
- `../nlevel/dynamic.py` contiene energía, geometría y minimización. Su término de carga todavía usa el producto `prod(cos(theta_i/2) + sin(theta_i/2))`, aunque las longitudes reportadas se calculan mediante otra recurrencia.
- `../nlevel/main_noisy.py` importa el solucionador ideal con argumentos incompatibles con su firma. Las versiones de `old/` también usan otras normalizaciones y coordenadas de carga.

Aquí se conserva la separación entre modelo, solucionador y barrido, pero se deriva el equilibrio de la nueva energía. La recurrencia elástica anterior no se reutiliza. Todos los observables y las derivadas de la carga proceden de la misma geometría.

## Modelo y convenciones

Los índices matemáticos van de `i = 1` a `N`; los arreglos Python empiezan en cero. Todos los ángulos se expresan en radianes. Se definen

\[
S_i=\sum_{m<i}\theta_m,\qquad
\phi_i^-=\theta_i-S_i,\qquad
q_i=4^{N-i}k_i^-.
\]

La energía interna es

\[
E_-(\theta)=2\sum_{i=1}^N q_i(\theta_i-S_i)^2.
\]

La familia de rigideces que admite el barrido es

\[
k_i^-=k_*4^{s(i-N)},\qquad k_i^+=0.
\]

`--k-scale` fija `k_*` y `--exponents` fija uno o varios valores de `s`. Para `s=1`, todos los pesos efectivos son `q_i=k_*`: se compensa la multiplicidad de cada nivel. Para `s=0`, todas las bisagras individuales tienen la misma rigidez `k_i^-=k_*`. El modelo también admite un arreglo explícito `k_i` desde Python.

La geometría comienza con `X = Y_min = 1`. En cada nivel, usando los valores anteriores y `c_i=cos(theta_i/2)`, `s_i=sin(theta_i/2)`, se calcula

\[
X_i=c_iX_{i-1}+s_iY_{\min,i-1},\qquad
Y_{\min,i}=c_iY_{\min,i-1},
\]

\[
Y_{\max,i}=s_iX_{i-1}+c_iY_{\min,i-1},\qquad
X_{\min,i}=s_iY_{\min,i-1}.
\]

En particular,

\[
X(\theta)=\prod_i\cos(\theta_i/2)
\left[1+\sum_i\tan(\theta_i/2)\right],\qquad X_0=1.
\]

El código evalúa la recurrencia con senos y cosenos, evitando las divisiones por cosenos de esta fórmula cerrada. La coordenada adoptada es una **longitud característica recursiva**: no se ha verificado aquí que sea la caja envolvente de todos los cuadrados o la separación entre puntos de agarre físicos específicos.

La deformación y el potencial de fuerza controlada son

\[
\varepsilon=\frac{X-X_0}{X_0}=X-1,
\qquad H_-=E_--f(X-1).
\]

El término constante respecto de los ángulos se resta para mejorar la precisión numérica. `force` siempre significa `f`, conjugada a `X`; **no es `f X` y no debe dividirse después por la longitud**. Al recuperar una longitud física inicial `D`, la fuerza física `F` cumple `f=F D`; las rigideces deben expresarse en unidades de energía compatibles.

## Cálculo cuadrático y variables alpha

La expansión geométrica que se compara con la recurrencia es

\[
\varepsilon_{\rm quad}=\frac12\sum_i\theta_i
-\frac18\sum_i\theta_i^2.
\]

Con `alpha_i = 1 - theta_i/2`, `u_i = alpha_i - 1` y una matriz triangular `M` tal que `(M u)_i=u_i-sum_{m<i}u_m`,

\[
B_\alpha=16M^{\mathsf T}\operatorname{diag}(q_i)M,
\]

\[
H_{\rm quad}=\frac12(\boldsymbol\alpha-\mathbf1)^{\mathsf T}
B_\alpha(\boldsymbol\alpha-\mathbf1)
+\frac f2\boldsymbol\alpha^{\mathsf T}\boldsymbol\alpha
+\text{constante}.
\]

El equilibrio puede escribirse como

\[
(B_\alpha+fI)\boldsymbol\theta=2f\mathbf1.
\]

Se obtienen los modos a partir de la descomposición en valores singulares de `4 diag(sqrt(q_i)) M`. Así se evita perder precisión en el modo blando al diagonalizar directamente `B_alpha`. Para cada modo se guardan la rigidez `b_p` y el peso `P_p=(v_p^T 1)^2`. El programa rechaza combinaciones cuyo modo blando no se resuelve adecuadamente en doble precisión.

`quadratic` es la solución del **Hamiltoniano aproximado**, aunque el sistema lineal se resuelva directamente. Se pueden evaluar sus ángulos en la geometría sin expandir para medir el error geométrico; eso no los convierte en un equilibrio del Hamiltoniano trigonométrico.

## Minimización y diagnósticos

`nonlinear` minimiza localmente `E_- - f(X-1)`, usando gradiente y Hessiana de la recurrencia. El barrido utiliza continuación desde cargas menores y coordenadas modales escaladas para resolver el modo muy blando. No se imponen límites artificiales a los ángulos. Se registran el residuo relativo, el número de iteraciones, el mensaje del optimizador y el menor autovalor de la Hessiana en las coordenadas escaladas. Una solución se acepta por residuo y estabilidad local, además de conservar el diagnóstico del optimizador; no constituye una prueba de mínimo global.

También se exportan `max_abs_theta`, una bandera `small_angle` para `max(abs(theta_i)) <= 0.2` rad y el error relativo entre ambas expresiones geométricas. El umbral angular es orientativo: hay que mirar también el error acumulado de la geometría. El denominador del error relativo tiene un piso de `1e-12` para evitar una división por cero cerca del estado descargado.

La palabra «exacta», cuando se usa para la geometría, significa **sin expansión trigonométrica dentro de esta coordenada recursiva**. El modelo no incluye contacto entre cuadrados, límites de apertura de las bisagras, estiramiento material, grados de libertad distintos del ángulo de cada nivel ni una verificación independiente de puntos de agarre. La aproximación `alpha=1-theta/2` no habilita extrapolar el Hamiltoniano cuadrático a ángulos grandes.

## Ejecución

Se requieren Python 3.10 o posterior, NumPy, SciPy y Matplotlib. Desde la raíz del proyecto, en tu entorno de Python:

```bash
python3 -m pip install -r src/nlevel_phi_minus/requirements.txt
python3 src/nlevel_phi_minus/main.py --levels 4 8 12 --exponents 1 --k-scale 1 --method both --points 81
```

También se puede ejecutar desde la carpeta del experimento:

```bash
cd src/nlevel_phi_minus
python3 main.py --levels 4 8 12 --exponents 1 --method both
```

Para comparar distintas distribuciones de rigidez:

```bash
python3 src/nlevel_phi_minus/main.py --levels 4 8 12 --exponents 0 0.5 1 --method both --points 81
```

Un barrido acotado explícitamente en fuerza:

```bash
python3 src/nlevel_phi_minus/main.py --levels 4 --exponents 1 --method both --f-min 1e-5 --f-max 1 --points 81 --output /tmp/phi_minus_ensayo_01
```

La ruta de `--output` debe ser nueva; el programa evita sobrescribir un experimento existente. Si se omite, se crea una carpeta de resultados con fecha y hora. Si no se fijan los extremos de fuerza, el barrido elige una escala a partir de la menor rigidez modal de cada modelo. Se añade el estado descargado para fijar la referencia.

`--method quadratic` ejecuta sólo la solución aproximada; `--method nonlinear`, sólo la minimización; `--method both`, ambas. `--no-plots` omite las figuras. `--tol` y `--maxiter` controlan la minimización. Consulta todas las opciones con:

```bash
python3 src/nlevel_phi_minus/main.py --help
```

## Resultados y lectura física

Cada ejecución produce:

- `curves.csv`: fuerza, deformaciones, longitudes, energía y diagnósticos por combinación de parámetros y método.
- `angles.csv`: ángulos de cada nivel para reconstruir las configuraciones y localizar dónde se concentra la deformación.
- `spectrum.csv`: rigideces y pesos modales de cada modelo.
- Metadatos de la ejecución y cuatro figuras PNG, salvo que se use `--no-plots`.

Para buscar un régimen no hookeano, compara la curva fuerza–deformación con la pendiente local `beta=d(log f)/d(log epsilon)`. Hooke corresponde a `beta=1`; una pendiente próxima a `2` sólo es evidencia de una ley aproximadamente cuadrática si permanece sobre un intervalo amplio y bien resuelto. Los puntos sin convergencia o fuera de la validez del cálculo elegido no deben usarse para ajustar exponentes.

Con sólo `phi-` puede aparecer una rigidez extremadamente pequeña asociada a un modo dominante. Su agotamiento puede producir una transición marcada sin generar un régimen multiescala amplio. Por eso conviene estudiar conjuntamente la curva, los ángulos, el espectro y los pesos modales, empezando por pocos niveles. Aumentar `N` o encontrar un cruce puntual de `beta=2` no demuestra por sí mismo la ley no hookeana buscada.
