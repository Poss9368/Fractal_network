# Pares phi+ con exponente elegible y crossover de strain ~1/N

Este experimento usa el mismo Hamiltoniano y la misma longitud trigonométrica
de `nlevel_5`. No fija rígidamente los niveles pares: todos se minimizan.
Complementa la construcción más simple de reposo uniforme en
`../nlevel_crossover_uniform/README.md`.

## Parámetros realizables

Para N=2M y j=1,...,M, todas las bisagras son phi+. Definir los pesos efectivos
a_i=4^(N-i) k_i w_i^+ mediante

\[
a_{2j-1}=\kappa_j=(j/M)^p,\qquad a_{2j}=1,\qquad p>1.
\]

Esto se realiza con k_i=4^(-(N-i)), w_i^-=0, w_(2j)^+=1 y
w_(2j-1)^+=(j/M)^p, todos los pesos de familia dentro de [0,1].
No se usa aleatoriedad.

Sea theta_star el ángulo uniforme de máxima extensión. El reposo es

\[
\theta_{2j-1}^{(0)}=\theta_*+d,\qquad
\theta_{2j}^{(0)}=\theta_*-d,\qquad d=\eta/\sqrt M.
\]

Usar eta=0.5. Todos los ángulos del reposo y de las soluciones comprobadas
son positivos. La restricción suficiente asintótica es eta<sqrt(2); el
código verifica positividad para cada tamaño finito.

En `nlevel_5`, la configuración equivalente es:

```python
m = n // 2
p = 3.0
eta = 0.5
family_weights = np.zeros((n, 2))
family_weights[:, 1] = 1.0
family_weights[::2, 1] = (np.arange(1, m + 1) / m) ** p
theta_rest = np.full(n, maximum_length_angle(n))
theta_rest[::2] += eta / np.sqrt(m)
theta_rest[1::2] -= eta / np.sqrt(m)
rest_deformations = rest_deformations_from_thetas(theta_rest)
thetas = theta_rest.copy()
k_i = np.exp2(-2.0 * (n - np.arange(1, n + 1)))
```

Para N grande no calcular separadamente 4^(N-i) y su inverso: se producen
desbordamientos aunque su producto sea 1. Este experimento usa a_i
directamente y evita ese problema.

## Mecanismo y aproximación controlada

Con S_i=sum_(k<=i) theta_k, definir
x_j=S_(2j-1)-(2j-1)theta_star e y_j=S_(2j)-2j theta_star.
La energía es exactamente

\[
E=\sum_j[\kappa_j(x_j-d)^2+y_j^2].
\]

A cargas pequeñas los niveles pares mantienen y_j aproximadamente nulo,
de modo que los ángulos del par son theta_star+x_j y theta_star-x_j.
La longitud y la extensión de ingeniería tienen los límites

\[
X/X_*\simeq\exp[-\tfrac14\sum_jx_j^2],\qquad
X_0/X_*\to e^{-\eta^2/4},\qquad
\epsilon_{\max}\to e^{\eta^2/4}-1.
\]

Para eta=0.5 la reserva relativa tiende a 0.064494..., que no depende de N.
Sea q=fX/4, donde f multiplica -X en el Hamiltoniano. El equilibrio da

\[
x_j\simeq d\frac{\kappa_j}{\kappa_j+q},\qquad
\log(1+\epsilon)\simeq\frac{\eta^2}{4M}
\sum_{j=1}^M\left[1-\left(\frac{(j/M)^p}{(j/M)^p+q}\right)^2\right].
\]

Son expresiones asintóticas para q<<1 y N grande. Las correcciones por
la movilidad de los niveles pares son de orden q; no se impone rigidez
infinita en las simulaciones. Si se compara a strain pequeño, la fuerza
fX_0 y 4q difieren por el factor 1+epsilon.

Para q<<M^(-p),

\[
\epsilon\simeq\frac{\eta^2}{2}M^{p-1}
\left(\sum_{j=1}^M j^{-p}\right)q.
\]

Para M^(-p)<<q<<1, la suma converge a una integral:

\[
\log(1+\epsilon)\simeq\frac{\eta^2}{4}A_p q^{1/p},\qquad
A_p=\frac{\pi(p+1)}{p^2\sin(\pi/p)}.
\]

A strain pequeño esto implica fX_0 proporcional a epsilon^p. Por tanto,
p=2 genera un régimen cuadrático y p=3 uno cúbico. El modo más blando
deja de responder linealmente a q_cross~M^(-p), con

\[
\boxed{\epsilon_\times\sim\eta^2/M\sim N^{-1}.}
\]

La elongación absoluta en el crossover también disminuye como N^(-1/2)
porque X_0~sqrt(N). La fuerza original del crossover escala como
N^(-(p+1/2)). El efecto combina una distribución de rigideces y una
contracción inicial repartida entre muchos modos; un único modo blando
sin amplitud en el reposo no ofrece este mecanismo.

## Reproducir y leer resultados

```bash
python3 src/nlevel_crossover_pairs/experiment.py --sizes 64 256 1024 4096 8192 --points 401 --output src/nlevel_crossover_pairs/results/verified
```

La referencia predeterminada es `maximum`. La opción `--reference square`
prueba parejas (+d,-d) alrededor de cero, con ángulos negativos, y no es
la configuración principal propuesta aquí.

El solucionador utiliza geometría trigonométrica exacta, gradiente y
Hessiano analíticos, y continuación en fuerza. Trabaja con desviaciones
acumuladas respecto del reposo para no perder precisión al restar grandes
sumas angulares. Antes de ejecutar verifica energía, recurrencia geométrica,
gradiente y solución del Hessiano contra expresiones independientes.

El crossover se mide por epsilon(f_cross)=0.9*C_N*f_cross, con C_N la
susceptibilidad exacta a fuerza cero. El mismo criterio se usa en el
experimento uniforme. En los archivos `results/verified`, para p=3:

| N | strain del crossover | ajuste de exponente intermedio |
|---:|---:|---:|
| 64 | 3.68332e-4 | 2.6980 |
| 256 | 9.26289e-5 | 2.8356 |
| 1024 | 2.31957e-5 | 2.9027 |
| 4096 | 5.80164e-6 | 2.9378 |
| 8192 | 2.90053e-6 | 2.9491 |

El ajuste de epsilon_cross contra N da exponente -0.99851. El exponente
constitutivo de la tabla es un ajuste sobre una ventana amplia, fijada
antes de inspeccionar las pendientes: 100*M^(-p)<fX_0<0.01. Incluye
correcciones de tamaño finito y no da exactamente p. Para p=3 la
pendiente local en q=M^(-3/2), entre ambos cortes, es 2.943, 2.972 y
2.981 para N=1024,4096,8192. La predicción asintótica es 3, no 2.949.

Para p=2 el ajuste amplio aumenta de 1.931 (N=256) a 1.980 (N=8192).
El máximo residuo relativo del gradiente en el barrido p=3,N=8192 fue
2.6e-9. Se sigue la rama conectada al reposo, sin certificar un mínimo
global sobre todas las rotaciones angulares posibles.

Las figuras normalizan verticalmente la fuerza con X_0. El strain
horizontal es siempre (X-X_0)/X_0, sin reescalamiento por N. Los CSV
guardan la fuerza original. Las carpetas de resultados anteriores
contienen barridos exploratorios; `verified` corresponde al solucionador
en desviaciones acumuladas documentado aquí.

Como en el modelo original, no se incluye auto-contacto ni se añaden
restricciones nuevas a los ángulos físicos de las bisagras.
