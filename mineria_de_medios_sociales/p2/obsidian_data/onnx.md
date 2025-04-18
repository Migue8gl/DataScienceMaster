- #onnx | #ml | #dl 
### Links:
- [doc](https://onnx.ai/onnx/intro/index.html)

# Conceptos
- **ONNX** es una especie de lenguaje de programación especializado en funciones matemáticas. Define todas las operaciones necesarias para que un modelo de [[machine learning & data science]] pueda implementar su función de inferencia.
- **ONNX** es un formato que describe el gráfico de operaciones de un modelo (los cálculos y transformaciones que realiza).
- Un ejemplo de [[perceptron#^da9b29|net input]] o regresión lineal es:

```python
def onnx_linear_regressor(X):
    "ONNX code for a linear regression"
    return onnx.Add(onnx.MatMul(X, coefficients), bias)
```

- La intención detrás de este proyecto es la de crear un framework común que sirva para describir los modelos. Gracias a esto también es posible hacer más fácil el despliegue de modelos en producción.
- Un intérprete de **ONNX** puede ser implementado para evaluar modelos creados en distintos frameworks de **ML**.
## ONNX Graph
- Crear un grafo **ONNX** significa implementar una función usando los operadores del lenguaje, pues las operaciones se implementan como nodos y los *inputs* y *outputs* de las funciones son *inputs* y *outputs* en los nodos.
- Los grafos pueden tener un **inicializador**, esto es una variable que nunca cambia. Hacerlo de esta forma es mucho más eficiente y se convertiría en una constante.

![[onnx-graph-example.png|400]]

## Protobuf
- La serialización con **protobuf** se refiere al proceso de convertir un modelo de aprendizaje automático entrenado en un formato compacto y eficiente para ser almacenado y transmitido. En el contexto de despliegue de modelos, **protobuf (Protocol Buffers)** es un formato de serialización eficiente creado por Google, que se usa para transformar datos en bloques binarios que pueden ser fácilmente guardados y transferidos entre sistemas.
- Esto hace que el modelo sea portable, optimizado en tamaño y que su carga sea más rápida
## Metadata
- Es posible almacenar metadatos en el formato de **ONNX** para realizar el seguimiento de datos como versión, autor, etc.

# Tools
- [netron](https://netron.app/) es una herramienta que facilita la visualización de modelos en **ONNX**.
- [onnx2py.py](https://github.com/microsoft/onnxconverter-common/blob/master/onnxconverter_common/onnx2py.py) crea un *script* python que puede crear el mismo grafo **ONNX**. De esta forma hace que sea más sencillo modificar el grafo manualmente.
- [zetane](https://github.com/zetane/viewer) puede cargar el modelo y mostrar resultados intermedios mientras el modelo se ejecuta.

# Converters
- Una librería *convertidora* es capaz de transformar modelos de esa librería al formato **ONNX**.