#tietjen-moor | #hypothesis | #datascience | #outliersdetection 

# Concepto
- Es una generalización del [[grubbs test|test de Grubb]]. Para la detección de un solo [[outliers]] es equivalente.
- Se requiere que el número de *outliers* sea especificado a priori.

# Proceso
- Se hace el siguiente planteamiento de [[hypothesis testing|hipótesis]]:
	- $H_{0}: \text{No hay outliers en los datos}$.
	- $H_{1}: \text{Hay exactamente K outliers}$.
- El problema con este test es el **Swamping**.
- *Swamping* aparece cuando un test para $k$ *outliers* declara que hay $k$ anomalías y en realidad hay menos. Ocurre cuando hay anomalías muy significativas y extremas, de forma que el estadístico es afectado por estos puntos.