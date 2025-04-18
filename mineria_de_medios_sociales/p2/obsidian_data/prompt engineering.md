- #promptengineering | #foundationalmodels | #llm

# ChatGPT
- **Principio 1**: Escribir instrucciones claras y específicas.
- **Principio 2**: Darle tiempo al modelo para pensar (y agregar más pasos).
- **Uso de delimitadores**: Pueden ser `““““, <>, <tag></tag>, :`.
- **Solicitar salida estructurada**: Por ejemplo, en formato JSON o HTML.
- **Revisión de condiciones**: Preguntar al modelo si la salida es válida (por ejemplo, que el JSON es válido).
```python
text_1 = f"""
Making a cup of tea is easy! First, you need to get some \ 
water boiling. While that's happening, \ 
grab a cup and put a tea bag in it. Once the water is \ 
hot enough, just pour it over the tea bag. \ 
Let it sit for a bit so the tea can steep. After a \ 
few minutes, take out the tea bag. If you \ 
like, you can add some sugar or milk to taste. \ 
And that's it! You've got yourself a delicious \ 
cup of tea to enjoy.
"""
prompt = f"""
You will be provided with text delimited by triple quotes. 
If it contains a sequence of instructions, \ 
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequence of instructions, \ 
then simply write \"No steps provided.\"

\"\"\"{text_1}\"\"\"
```
- **Añadir ejemplos**: Añadir técnicas de *few-shot*.
- **Especificar pasos**: Especificar pasos en formato de listado.
```bash
 Perform the following actions: 
1 - Summarize the following text delimited by triple \
backticks with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the following \
keys: french_summary, num_names.

Separate your answers with line breaks.
```
- **Auto-corrección**: Pedir al modelo resolver la tarea y que después valide la solución indicando el por qué.
## Proceso iterativo de prompt
- Básicamente, necesitamos probar el prompt de manera exhaustiva a través de muchos ejemplos. En cada uno, intentamos resolver el problema añadiendo contexto al prompt.
- **"IDEA >> IMPLEMENTACIÓN >> EXPERIMENTO >> ANÁLISIS >> CICLO"**
	- Si el texto es demasiado largo, indicarle que utilice un máximo de N palabras.
	- Si no se está enfocando en la parte correcta del contexto, decírselo.
	- Si necesitamos que corrija el formato, indicarle qué formato debe seguir o qué debe evitar.
- Para resúmenes, debemos indicarle la cantidad máxima de palabras / párrafos y decirle en qué debe centrarse del texto. Por ejemplo, si el texto trata sobre la venta de un objeto, indicarle que se enfoque en el precio y el valor percibido.
- A veces es mejor pedirle que "extraiga" en lugar de "resumir". A veces, utilizar palabras distintas para especificar una tarea o un concepto puede funcionar mejor. Es un enfoque iterativo, pero puede ayudar a extraer un poco más de información de la tarea actual.
- También podemos pedirle que realice múltiples tareas a la vez, como análisis de sentimientos, detección de palabrotas, entre otras. No obstante, debemos indicarle que esas son tareas diferentes y que debe responder en un formato adecuado que nos permita distinguir la solución de cada tarea.