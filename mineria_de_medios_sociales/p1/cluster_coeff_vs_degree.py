import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Cargar datos
df = pd.read_csv("java_network.csv")

# Crear el grafo dirigido
G = nx.DiGraph()
G.add_edges_from(zip(df["Source"], df["Target"]))
# Calcula el coeficiente de clustering para cada nodo
clustering_coeffs = nx.clustering(G)

# Calcula el grado de cada nodo
degrees = dict(G.degree())

# Prepara los datos para el gráfico
x = [degrees[node] for node in G.nodes()]  # grados
y = [clustering_coeffs[node] for node in G.nodes()]  # coeficientes de clustering

# Crea el gráfico
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5)
plt.title("Coeficiente de Clustering vs Grado")
plt.xlabel("Grado")
plt.ylabel("Coeficiente de Clustering")
plt.grid(True)
plt.savefig("clustering_distribution_vs_degree.png", dpi=300)
plt.close()
