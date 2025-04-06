import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# Cargar datos
df = pd.read_csv("java_network.csv")

# Crear el grafo dirigido
G = nx.DiGraph()
G.add_edges_from(zip(df["Source"], df["Target"]))

# Calcular coeficientes de clustering
clustering_coeffs = list(nx.clustering(G).values())

# Contar frecuencia de cada coeficiente
counts = Counter(clustering_coeffs)
x, y = zip(*sorted(counts.items()))  # Ordenar valores únicos

# Graficar
plt.scatter(x, y, alpha=0.7)
plt.xlabel("Coeficiente de Clustering")
plt.ylabel("Frecuencia")
plt.title("Distribución de Coeficientes de Clustering")
plt.savefig("clustering_distribution_scatter.png", dpi=300)
plt.close()
