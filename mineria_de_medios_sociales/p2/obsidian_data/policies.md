- #policies | #elastic
### Links
- [ILM](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-lifecycle-management.html)
- [Deletel](https://www.elastic.co/guide/en/elasticsearch/reference/current/ilm-delete.html)
- [Rollover](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-rollover.html)
- [Configure lifecycle policy](https://www.elastic.co/guide/en/elasticsearch/reference/current/set-up-lifecycle-policy.html)

- Para configurar la gestión del ciclo de vida de los [[index|índices]] se pueden especificar **políticas**.
- Esto se hace en `index.lifecycle.name`.
- Las políticas **ILM** (*index lifecycle management*) se guardan en el estado del clúster,
- Para crear la política desde [[kibana]], se abre **Stack Management > Index Lifecycle Policies**.
![[policies-kibana-screen-creation.png|500]]
- Una vez creado la *policy* con las configuraciones necesarias hay que añadirla al índices o índices que se requieran.
	- Se pueden usar *templates* para aplicar una política.
	- Se puede hace directamente la API.
```bash
PUT /.ent-search-engine-documents-fichas-productos-it-se/_settings?pretty
{
  "index": {
    "lifecycle": {
      "name": "policy_unused_indexes_hot_to_cold"
    }
  }
}
```
![[policies-template.png|500]]
- La opción de borrar el índice dado su `age` es posible. Ha de tenerse en cuenta que ese parámetro se refiere al tiempo de vida del índice, NO AL TIEMPO QUE EL ÍNDICE ESTÁ EN UNA FASE. Si el *rollover* está activado, se tiene en cuenta el tiempo desde esa creación.
- No se pueden borrar *engines* desde **ILM**. Para ello se puede hacer un *cron* que compruebe los índices asociados a un *engine* y si están borrados borrar el *engine*.