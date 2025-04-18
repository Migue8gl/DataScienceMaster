- #bayesiannetwork | #sampling | #montecarlo | #likelihoodweighting

# Concepto
- Funciona de forma parecida al [[logic sampling|muestreo lógico]], pero cuando llega a un nodo observado $X_{i}=e_{i}$, se procede de la siguiente forma. En vez de simular un valor para $X_{i}$, fija el valor de $X_{i}=e_{i}$.
- Con ello se intenta resolver el problema del alto rechazo del muestreo lógico.
- Los nodos también se ordenan según una ordenación ancestral.
![[likelihood weighting-distitributions.png]]

# Ejemplo
![[likelihood weighting-ej1.png]]
![[likelihood weighting-ej2.png]]
![[likelihood weighting-ej3.png]]
![[likelihood weighting-ej4.png]]
![[likelihood weighting-ej5.png]]