- #mlbusiness | #linkedin

# CEO
- El **CEO** de Cambrian Intelligence es Cristóbal Esteban. Este hombre es investigador e ingeniero de IA.
- Trabajo como analista en **Accenture** y más tarde creó su empresa, **Cambrian Intelligence**.
- Tiene *papers* interesantes.
## Learning with Memory Embeddings
- El artículo analiza cómo los modelos de representación (o *embeddings*) pueden mapear funciones de memoria humana a sistemas computacionales. Estos modelos, comúnmente usados para representar gráficos de conocimiento, se aplican aquí a memorias cognitivas como la semántica, episódica, sensorial y de trabajo. Los autores proponen hipótesis que relacionan estas memorias con estructuras de tensor y representación latente.
1. **Representaciones latentes y gráficos de conocimiento**:
    - Los gráficos de conocimiento se representan como tensores donde las entidades y relaciones tienen representaciones latentes compartidas.
    - Estas representaciones permiten inferir nuevas conexiones y manejar alta dimensionalidad y esparsidad.
2. **Modelos de memoria**:
    - **Memoria semántica**: Almacena conocimiento general en gráficos de conocimiento (tripletas: sujeto, predicado, objeto).
    - **Memoria episódica**: Captura eventos con una dimensión temporal adicional.
    - **Memoria sensorial**: Procesa información subsimbólica en buffers temporales.
    - **Memoria de trabajo**: Coordina las anteriores para predicciones y toma de decisiones.
3. **Hipótesis principales**:
    - La memoria semántica se describe como tripletas y la episódica como cuádruples (tripletas con tiempo).
    - Las representaciones latentes son universales y compartidas entre las funciones de memoria.
    - La memoria episódica se basa en representaciones temporales latentes derivadas de la entrada sensorial.
    - Existe interdependencia entre las memorias semántica y episódica: la primera es un almacén a largo plazo de la segunda.
4. **Aspectos técnicos**:
    - Uso de descomposiciones tensoriales (PARAFAC, Tucker) para representar memorias.
    - Decodificación semántica mediante muestreo estocástico para recuperar tripletas.
    - Transferencia de memoria episódica a largo plazo a memoria semántica a través de procesos como la consolidación durante el sueño.
5. **Paralelos con la memoria humana**:
    - Modelos como la hipótesis de memoria tensorial emulan funciones cerebrales, sugiriendo cómo memorias como la episódica y semántica interactúan en el cerebro humano.
## Real-valued (medical) time series generation with recurrent conditional GANs
- En este artículo se utilizan Redes Generativas Adversativas para generar datos sintéticos del ámbito médito. El generador y el discriminador son Redes Neuronales Recurrentes (**RNNs**). Se proponen dos tipos, una red generativa recurrente y una red generativa recurrente condicional.
- Específicamente, el discriminador se entrena para minimizar la entropía cruzada negativa media entre sus predicciones *por paso temporal* y las etiquetas de la secuencia.
- La entropía cruzada es una métrica que mide la discrepancia entre dos distribuciones de probabilidad.
- Sea $RNN(X)$ el vector o matriz que incluye los *outputs* $T$ de una **RNN** que recive una secuencia de $T$ vectores $\{x_{t}\}^{T}_{t=1}$ $(x_{t}\in \mathbb{R}^{d})$, y sea $CE(a,b)$ la entropía media entre una secuencia $a$ y $b$. Entonces, la pérdida del discriminador para un par $\{X_{n},y_{n}\}$ (con $X_{n}\in \mathbb{R}^{T\times d}$ y $y_{n}\in\{0,1\}^{T}$) es:
$$D_{loss}(X_{n},y_{n})=-CE(RNN_{D}(X_{n}), y_{n})$$
- El generador tiene la tarea de engañar al discriminador y hacer que sus clasificaciones sean $1$ o $true$. Esto es minimizar la media de la entropía negativa entre las predicciones del discriminador en secuencias generadas y la etiqueta "verdadera".
$$G_{loss}(Z_{n})=-CE(RNN_{D}(RNN_{G}(Z_{n})), 1)$$
![[cambrian intelligence-rgan-example.png|500]]
- Se propone además un método de evaluación nuevo, el *TSTR* (*train on synthetic, test on real*). 
![[cambrian intelligence-tstr.png]]
- Se generan ejemplos sintéticos y se comparan las distancias entre esos ejemplos y los del *dataset* de entrenamiento para ver con *tests* estadísticos si hay filtración de datos. No se dió un resultado estadísticamente significativo, por lo que se rechaza la hipótesis.

# Cambrian Intelligence
- Es una especie de consultora dedicada exclusivamente a dar soluciones basadas en inteligencia artificial y análisis de datos.
- Ayudan con el preprocesamiento de datos, construcción de modelos basados en *IA*, desplegados en la nube y creación de interfaces de configuración que incluyen visualizaciones de datos, predicciones, herramientas de configuración, etc.

## F1-score
- Combina el recall con el acurracy, de forma que hay un equilibrio entre precisión y recall.
## Recall
- Mide la proporción de positivos reales que han identificado correctamente.
## Entropía Cruzada
- Es un término utilizado habitualmente en la teoría de la información, y mide las diferencias entre dos distribuciones de probabilidad que pueden utilizarse para identificar una observación.

https://huggingface.co/intfloat/multilingual-e5-large, 