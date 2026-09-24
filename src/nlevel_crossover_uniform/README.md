# Crossover que desaparece: todos phi+, reposo uniforme no nulo

Esta configuración reproduce los tres comportamientos buscados dentro del
modelo angular de `nlevel_5`: Hooke para cada N finito, un régimen cuadrático
intermedio y un crossover de strain que tiende a cero como 1/N.

## Configuración

Con índices i=1,...,N, usar solamente phi+ y pesos efectivos uniformes:

\[
w_i^-=0,\qquad w_i^+=1,\qquad a_i=4^{N-i}k_iw_i^+=1.
\]

El estado natural, que también sirve como punto inicial del minimizador, es

\[
\theta_i^{(0)}=c\theta_*(N),\quad 0<c<1,\qquad
\theta_*(N)=2\arctan\frac{2}{1+\sqrt{4N-3}}.
\]

El ejemplo principal usa c=0.8; c=0.5 también funciona y deja más elongación
disponible. No fijar solamente la condición inicial del algoritmo: hay que
cambiar también los ángulos naturales de la energía.

En la interfaz de `nlevel_5`, para tamaños que no desborden las potencias de 4:

```python
family_weights = np.column_stack((np.zeros(n), np.ones(n)))
theta_rest = np.full(n, 0.8 * maximum_length_angle(n))
rest_deformations = rest_deformations_from_thetas(theta_rest)
thetas = theta_rest.copy()
k_i = np.exp2(-2.0 * (n - np.arange(1, n + 1)))
```

Aquí se trabaja directamente con a_i=1 para evitar calcular por separado
4^(N-i) y 4^(-(N-i)), cuyo producto es 1 pero cuyos factores desbordan o
subdesbordan a N grande. El Hamiltoniano es el mismo.

## Modelo exacto y observable

\[
E=\sum_{i=1}^N\left[\sum_{j\le i}(\theta_j-\theta_j^{(0)})\right]^2,
\qquad \mathcal H=E-fX,
\]

\[
X=\prod_i\cos(\theta_i/2)\left[1+\sum_i\tan(\theta_i/2)\right],
\qquad \epsilon=\frac{X-X_0}{X_0},\quad X_0=X(\theta^{(0)}).
\]

El strain del gráfico es el strain de ingeniería, sin dividirlo por N ni
por sqrt(N). f es exactamente el multiplicador que aparece en E-fX.

Para c fijo,

\[
X_0\sim c e^{-c^2/2}\sqrt N,\qquad
\epsilon_{\max}\longrightarrow\frac{e^{(c^2-1)/2}}c-1>0.
\]

Por eso no se hace desaparecer el crossover contrayendo toda la ventana
de deformación. La reserva relativa de elongación tiende a 0.04409 para
c=0.8 y a 0.37458 para c=0.5.

## Derivación de los dos regímenes

Escribir theta_i=theta_i^(0)+2v_i/sqrt(N). La energía es exactamente

\[
E=\frac4N\mathbf v^T M\mathbf v,
\qquad M_{ij}=N-\max(i,j)+1.
\]

Para una perturbación de amplitud v_i acotada que ocupa ell << N niveles,
la expansión de la geometría **en torno al reposo no nulo** da

\[
\epsilon=\frac1N\left[d_N\sum_i v_i-\frac12\sum_i v_i^2\right]
+O(\ell^2/N^2+\ell/N^2),
\]

\[
d_N=\sqrt N\left[\frac{\sec^2t_0}{1+N\tan t_0}-\tan t_0\right]
\longrightarrow d=1/c-c,\qquad t_0=c\theta_*/2.
\]

La diagonal geométrica usada aquí es exacta: d²X/dtheta_i²=-X/4.
No se usa la expansión global X=1+sum(theta)/2-sum(theta²)/8,
que no es uniforme en N para esta referencia.

Definir s=fX_0/8. Entonces

\[
(M+sI)\mathbf v=s d_N\mathbf1,
\qquad (I+sL)\mathbf v=s d_N\mathbf e_N,
\]

donde L=M^(-1) es tridiagonal, con diagonal (1,2,...,2) y vecinos -1.
Para una jerarquía larga la solución decae desde el último nivel:

\[
v_{N-j}=d_N r^{j+1},\qquad
r=\frac{2s}{1+2s+\sqrt{1+4s}},\qquad s=\frac r{(1-r)^2}.
\]

Al sumar la contribución de todos los niveles,

\[
\boxed{\epsilon\simeq\frac{d_N^2}{2N}\frac{r(2+r)}{1-r^2}.}
\]

Por lo tanto,

\[
\epsilon\simeq
\begin{cases}
d_N^2s/N,&s\ll1,\\
3d_N^2\sqrt s/(4N),&1\ll s\ll N^2.
\end{cases}
\]

Equivalentemente,

\[
f\simeq\frac{8N}{d_N^2X_0}\epsilon\quad\hbox{(Hooke)},\qquad
f\simeq\frac{128N^2}{9d_N^4X_0}\epsilon^2\quad\hbox{(intermedio)}.
\]

Los límites de la ventana intermedia son asintóticos: requiere una
deformación suficientemente pequeña para que ell/N << 1, pero mayor
que la escala 1/N. Cerca de la máxima longitud la pendiente vuelve a
aumentar; no se afirma una ley cuadrática hasta la saturación.

La cantidad de niveles que responden crece como ell ~ sqrt(s). Mientras
ell es O(1) se recupera sólo O(1/N) de strain; después la participación
progresiva de niveles genera la ley cuadrática. Así,

\[
\boxed{\epsilon_\times\sim N^{-1},\quad f_\times\sim N^{-1/2}.}
\]

La elongación absoluta del crossover también desaparece:
Delta X_cross=X_0 epsilon_cross ~ N^(-1/2).

## Conexión espectral y corrección conceptual

Los autovalores de M y sus vectores normalizados son

\[
\mu_p=\frac1{4\sin^2(\omega_p/2)},\qquad
v_i^{(p)}=\frac2{\sqrt{2N+1}}\cos[(i-1/2)\omega_p],\qquad
\omega_p=\frac{(2p-1)\pi}{2N+1}.
\]

El peso del vector uniforme es

\[
P_p=(\mathbf1^T\mathbf v^{(p)})^2
=\frac{\cot^2(\omega_p/2)}{2N+1}.
\]

La respuesta en la aproximación local tiene la misma estructura de
filtrado modal de la cadena 1D:

\[
\epsilon\simeq\frac{d_N^2}{2N}\sum_p P_p
\left[1-\left(\frac{\mu_p}{\mu_p+s}\right)^2\right].
\]

Una brecha elástica finita (mu_min -> 1/4) no impide por sí sola que
epsilon_cross -> 0. También importan la geometría del reposo, los pesos
modales, la curvatura de la longitud y la definición del strain. La
conclusión previa para theta^(0)=0 no puede extenderse a este reposo.
En la cadena, el mismo mecanismo de filtrado está documentado en
`documentation/Non_hookean_mechanics_of_random_slender_chains.tex`.

## Medición, resultados y reproducción

Se define el crossover por epsilon(f_cross)=0.9*C_N*f_cross, donde C_N
es la susceptibilidad exacta a fuerza cero. Para la referencia uniforme,
C_N=g_0²/(2X_0), con g_0=dX/dtheta_i evaluado en el reposo. Este criterio
es idéntico para todos los tamaños. No se identifica el crossover con
el inicio de una meseta de pendiente exactamente 2.

La teoría da s_cross=0.080527... y
epsilon_cross ~ 0.072474... d²/N. Para c=0.8 esto es ~0.0146761/N.
Hay correcciones finitas de orden N^(-1/2) en d_N y X_0.

```bash
python3 src/nlevel_crossover_uniform/experiment.py
python3 src/nlevel_crossover_uniform/experiment.py --c 0.5 --sizes 64 256 1024 4096 --output src/nlevel_crossover_uniform/results_c05
```

Se guardan curvas, pendientes locales, predicciones de la capa de
deformación y resumen del crossover en CSV, más figuras PNG/PDF. Los
archivos de salida de la carpeta elegida se regeneran en cada ejecución.

La comprobación numérica usa la longitud trigonométrica exacta y un
Newton con Hessiano analítico, tridiagonal más rango dos en sumas
acumuladas. El gradiente y Hessiano se verifican con diferencias finitas,
y la inversión estructurada se contrasta contra una solución densa.
La continuación sigue la rama estable conectada al reposo; no se hace
una búsqueda global de todas las ramas posibles del potencial angular.

En los resultados c=0.8, el exponente local evaluado en fX_0=N pasa de
1.896 para N=64 a 1.996 para N=16384. Es un punto intermedio cuyo ancho
de capa escala como sqrt(N), lejos de los dos cortes asintóticos.

Resultados exactos del barrido guardado (321 fuerzas por tamaño):

| N | strain del crossover 10 % | N por strain del crossover | exponente intermedio |
|---:|---:|---:|---:|
| 64 | 1.71139e-4 | 0.0109529 | 1.8962 |
| 256 | 4.94392e-5 | 0.0126564 | 1.9561 |
| 1024 | 1.33041e-5 | 0.0136233 | 1.9807 |
| 4096 | 3.45004e-6 | 0.0141314 | 1.9911 |
| 16384 | 8.78959e-7 | 0.0144009 | 1.9957 |

El ajuste de los cuatro tamaños mayores da N^(-0.9694); los dos mayores
dan N^(-0.9864), consistente con la aproximación progresiva al exponente -1.
La predicción explícita de la capa de deformación concuerda con la
geometría exacta a s~1 con error relativo de 4.6e-5 a N=16384.
Además, se contrastaron energía y longitud contra las funciones reales de
`nlevel_5.dynamic` para N=8 y 32; un minimizador BFGS independiente dio
las mismas soluciones, con Hessianos positivos en esos puntos.

La figura usa f/N^(3/2) para comparar las curvas porque el prefactor
del régimen cuadrático crece como N^(3/2). Esa normalización vertical
no cambia los exponentes ni la posición horizontal del crossover;
los CSV contienen también la fuerza original.

Estos resultados conciernen al modelo de energía angular y longitud
reducida implementado. No incluyen auto-contacto ni nuevos límites de
rotación de bisagras.

La segunda construcción en `../nlevel_crossover_pairs/README.md` permite
elegir un exponente distinto (por ejemplo 3) mediante pesos alternados,
con el mismo escalamiento epsilon_cross ~ 1/N.
