- #pso | #metaheuristic 

- Es un algoritmo de optimización metaheurístico basado en poblaciones (llamada enjambre).
- Nació fruto de la simulación de las bandadas de pájaros, en un esfuerzo por simular su vuelo coordinado y aleatorio y la forma en la que estos encuentran comida.
- Estas partículas (soluciones) se mueven por el espacio basándose en unas simples fórmulas. Los movimientos de las partículas se guían por su posición más conocida en el espacio de búsqueda y por la posición más conocida de todo el enjambre. Cuando se descubren posiciones mejoradas, éstas pasan a guiar los movimientos del enjambre. El proceso se repite y de este modo se espera, aunque no se garantiza, que finalmente se descubra una solución satisfactoria. 
- Los valores $b_{lo}$ y $b_{up}$ representan los límites del espacio de búsqueda (*lower* y *upper*). El parámetro $w$ es el peso de *inercia*. Los parámetros $\phi_p$ y $\phi_b$ son los llamados *coeficiente cognitivo* y *coeficiente social*.
- Para evitar la divergencia ("explosión"), el peso de inercia debe ser inferior a $1$. Los otros dos parámetros pueden derivarse entonces gracias al enfoque de restricción, o seleccionarse libremente, pero los análisis sugieren dominios de convergencia para restringirlos. Los valores típicos están en $[1,3]$

![[particleSwarmArrowsAnimation.gif|400]]

```
for each particle i = 1, ..., S do
    Initialize the particle's position with a uniformly distributed random vector: xi ~ U(blo, bup)
    Initialize the particle's best-known position to its initial position: pi ← xi
    if f(pi) < f(g) then
        update the swarm's best-known position: g ← pi
    Initialize the particle's velocity: vi ~ U(-|bup-blo|, |bup-blo|)
while a termination criterion is not met do:
    for each particle i = 1, ..., S do
        for each dimension d = 1, ..., n do
            Pick random numbers: rp, rg ~ U(0,1)
            Update the particle's velocity: vi,d ← w vi,d + φp rp (pi,d-xi,d) + φg rg (gd-xi,d)
        Update the particle's position: xi ← xi + vi
        if f(xi) < f(pi) then
            Update the particle's best-known position: pi ← xi
            if f(pi) < f(g) then
                Update the swarm's best-known position: g ← pi
```