- #snapshots | #elastic

- Las *snapshots* son copias de seguridad de un índice o índices en un clúster de Elastic.
- En caso de pérdida de pueden usar para restaurar ese índice perdido.
- Estas se pueden almacenar en un repositorio externo, como [[google cloud|Google Cloud]].
- Cada *snapshot* debe tener un nombre único dentro del repositorio.
- Las *snapshots* son automáticamente deduplicadas, es decir, se eliminan redundancias y por ello se pueden hacer *snapshots* frecuentes sin que impacte negativamente en el almacenamiento.
- Son independientes de las demás.