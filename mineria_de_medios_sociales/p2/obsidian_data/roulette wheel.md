- #selection | #optimization | #roulettewheel | #mh

- La ruleta (roulette wheel) o tambien conocida como **fitness proportionate selection** es un operador genético usando en [[genetic algorithm|GAs]] para seleccionar potenciales soluciones para la recombinación.
- En este operador selectivo se asigna un valor *fitness* a cada posible solución o cromosoma. Este valor se asocia a una probabilidad de ser seleccionado para recombinación.
- La probabilidad se calcula como: $p_i=\frac{f_i}{\sum_{j=i}^Nf_i}$, donde $N$ es el número total de individuos.