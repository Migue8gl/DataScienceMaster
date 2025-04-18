#processmining | #business | #process

# Concepto
- La minería de procesos se sitúa dentro de las **técnicas basadas en modelos** para gestionar [[business process|procesos de negocio]] y se apoya en los **registros de eventos** que provienen de sistemas empresariales como **ERP**, [[crm|CRM]], etc.

# Funcionalidades:
- **Descubrimiento automático** del proceso real a partir de registros.    
- **Verificación de conformidad** entre lo modelado y lo ejecutado.
- **Análisis de variantes** (diferentes ejecuciones del mismo proceso).
- **Minería del rendimiento**: analizar cuellos de botella, tiempos, etc.

# Registro de eventos
- Son archivos que documentan la ejecución real de los procesos (*logs*) normalmente con formato *csv*, *xes*.
- Suele albergar un **ID** de caso (para agrupaciones de eventos relacionados), la **actividad** y un **timestamp**.
![[process mining-logs-use-cases.png]]
## Traza
- Una traza en un registro de eventos es una ejecución de un proceso de negocio (una instancia de proceso).
- Es decir, una colección de entradas con el mismo **ID** de caso constituyen una traza.
## Mapa de proceso
- Un **mapa de proceso** es un grafo donde:
	- **Nodos** = actividades.
	- **Arcos** = relación de precedencia directa entre actividades.
- Los arcos pueden tener:
	- **Frecuencia absoluta**: cuántas veces ocurre la transición.
	- **Frecuencia relativa**: proporción respecto a ocurrencias anteriores.
	- **Tiempos medios**: útil para minería del rendimiento.
![[process mining-process-map.png]]