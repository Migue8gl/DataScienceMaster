- #tb | #metaheuristic 

- Las [[local search|búsquedas locales]] toman una potencial solución a un problema y comprueban sus vecinos más cercanos buscando una mejorar solución (estos son prácticamente iguales, excepto por pequeños detalles). Este tipo de búsquedas tienen el inconveniente de que tienen tendencia a quedarse atascados en óptimos locales.
- La búsqueda **tabú** cambia ciertas reglas de la búsqueda local:
	- En cada paso, soluciones peores pueden ser aceptadas si ninguna mejor es encontrada.
	- Las prohibiciones (de ahí el nombre *tabú*) son introducidas. Se utilizan estructuras de memoria para guardar soluciones ya visitadas o un conjunto de reglas introducidas por el usuario. Si una potencial solución ha sido ya visitada dentro de un período de tiempo corto o ciertas reglas han sido violadas, entonces esa solución se marca como *tabú* y el algoritmo no vuelve a considerar esa solución.

![[tabu-search-diagram.png|500]]

```
sBest ← s0
bestCandidate ← s0
tabuList ← []
tabuList.push(s0)
while (not stoppingCondition())
    sNeighborhood ← getNeighbors(bestCandidate)
    bestCandidateFitness ← -∞
    for (sCandidate in sNeighborhood)
        if ( (not tabuList.contains(sCandidate)) and (fitness(sCandidate) > bestCandidateFitness) )
            bestCandidate ← sCandidate
            bestCandidateFitness ← fitness(bestCandidate)
        end
    end
    if (bestCandidateFitness is -∞)
        break;
    end
    if (bestCandidateFitness > fitness(sBest))
        sBest ← bestCandidate
    end
    tabuList.push(bestCandidate)
    if (tabuList.size > maxTabuSize)
        tabuList.removeFirst()
    end
end
return sBest
```