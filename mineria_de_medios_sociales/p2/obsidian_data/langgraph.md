- #langchain | #ml | #llm | #nlp

# Concepto
- LangGraph permite definir **agentes** controlados por una **LLM**.
## Agente
- Un agente es un flujo de control definido por una **LLM**
![[langchain-agent-diagram.png|500]]
- Pueden ser totalmente autónomos o tener cierto tipo de control (**router**). A más control sobre el agente, menos fiable es.
##  Grafo
- El agente manejará un grafo de "acciones" por el cual podrá moverse.
- Un grafo de este tipo está compuesto por:
	- Estado: Sirve como el esquema inicial para todos los nodos y aristas del grafo. Pueden existir **estados privados o intermedios** para nodos concretos que necesitan ese tipo de estado, pero que cuya información no es necesario para el resto del grafo.
	- Aristas **condicionales**: Estos aristas representan una decisión que llevara a un nodo u otro, una acción u otra.
	- Aristas **normales**: Estas aristas simplemente concatenan acciones entre si (nodos).
	- Nodos: Cada nodo Es una función de Python, donde el primer argumento es el estado. Por defecto, cada nodo sobreescribe el estado anterior.

```python
graph_builder = StateGraph(State) # Estado compartido del grafo
  
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile()
```
## Herramientas
- Las herramientas son funciones que pueden añadirse a un modelo para hacer que este pueda controlar código o hacer llamadas a API's externas.
![[langgraph-tools.png]]
- Con la interfaz de `tool_calling` se puede agregar al modelo una serie de *tools* para que genere el *payload* necesario para llamar a la función.
- De esta forma: **mensajes** -> **inputs**.

```python
# Esta clase permite la ejecución de herramientas
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

tool_node = BasicToolNode(tools=tools) # Se crea un nodo de ejecución en el grafo principal
```
- En el ejemplo proporcionado, el nodo básico de ejecución de tareas guarda las herramientas por nombre. Si se llama a este nodo y se recibe un *input* en el que se encuentra que el agente razonador ha devuelto una entrada para una llamada a herramienta, estas son invocadas.
### ToolMessage
- Los *ToolMessage* son mensajes que contienen el resultado de ejecución de una herramienta y son pasados al modelo.
- El *tool_call_id* relaciona la petición de llamada de la herramienta con la respuesta de la herramienta. Esto es útil cuando un modelo es capaz de realizar múltiples peticiones de herramientas en paralelo. Por entenderlo mejor, cada ejecución de una misma herramienta genera un *id*, por lo que el *id* no identifica a la herramienta, **sino a la ejecución de la herramienta.**
## Human-in-the-loop
- A menudo es necesario que el usuario intervenga durante la ejecución del grafo, ya sea para realizar acciones sensibles o para confirmar determinadas operaciones.
- La ejecución puede ser pausada y reanudada una vez el humano haya introducido el *input* correspondiente. Se utiliza para ello el objeto *Command*, que contiene los datos requeridos por la herramienta de parada.

```python
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

human_command = Command(resume={"data": human_response})
events = graph.stream(human_command, config, stream_mode="values")
```
- Cuando se utiliza la función *interrup*, se espera a que se pase un mensaje de tipo *Command* para proseguir. Se puede utilizar *Command* para actualizar el estado del grafo con el parámetro *update* o incluso obligar con *goto* a navegar al siguiente nodo especificado en el parámetro.
## Reducers
- Las funciones *reducer* permiten cambiar la forma en la que se actualiza el estado del grafo.
- Si por ejemplo se quiere tener un historial de mensajes (lista de *strings*), por defecto el grafo sobreescribre el  estado. Para ello se puede usar el *reducer* `add_messages` para que añada al estado en vez de crear de nuevo la clase.
## Router
- Un *router* es una especie de agente. Es capaz de mapear la salida de un *tool_call* con la función a la que los argumentos pertenecen y así generar una respuesta. 
- Es un agente básico capaz de ejecutar código basandose en lenguaje natural.
- Usa los nodos de tipo **ToolNode** y las aristas de tipo **tools_condition**.
## Agent
- Un agente completo difiere del *router* en que la salida de una arista **tools_condition** vuelve a conectar con el modelo. De esta forma, el modelo puede seguir razonando y llamando a su kit de herramientas hasta que considere terminada la tarea.
- Esto permite instrucciones complejas en lenguaje natural y concatenación de herramientas externas bajo el uso de una "inteligencia" que controla el flujo de forma automatizada.
## Memory
- Cada invocación del grafo hace que se resete el estado. Por ello, para poder persistir el estado entre ejecuciones es necesario el uso de **memoria**.
- Usando `MemorySaver` se crean *checkpoints* que pueden accederse mediante su *id*. Este *id* o *thread_id* permite identificar a este grafo y continuar la conversación.
- Todo esto se consigue mediante *checkponting*.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}
# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
```
- Es tan sencillo como definir la memoria, darle un *id* a la configuración del grafo y comenzar a *stremear* mensajes.
- El *checkpointer* puede ser una base de datos de tipo persistente, como *postgres* de forma que se puede reanudar este tipo de interacciones siempre, incluso cuando el grafo deja de funcionar durante un tiempo, ya que se utilizaría **memoria persistente**.
- **DEBUG** -> para poder *debuggear* es posible utilizar `snapshot = graph.get_state(config)` y comprobar los *checkpoints* del grafo.
# Langsmith
- Es una plataforma que permite conectar los agentes de un proyecto para monitorizar la cadena de acciones de estos agentes. Permite un análsis profundo del comportamiento del agente.

# Técnicas avanzadas
## Información sensible
- Cuando se utiliza un parámetro que no debe ver la **IA** ya que contiene información sensible y debe inyectarse directamente. Para eso se usan los objetos o herramientas de *LangGraph* de tipo `Injected*`.
## Otros
- **Recorte**: Se puede limitar el número de *tokens*.
- **Reducers** para memoria eficiente: Se pueden usar tećnicas que usan los *reducers* para ir eliminando datos innecesarios de la memoria o para usar solo algunos de estos datos.
- **Memoria externa**: Estados de tipo persistente.
- **Streaming**: Soporta *streaming*, esto es que, se pueden devolver, actualizar o *debuggear* los estados de los nodos o sus *outputs*.
	- **values** -> con *values* vemos el estado completo.
	- **update** -> se ven solo las actualizaciones.
	- También se pueden *streamear* los *tokens*.
- **Breakpoints**: Se utilizan para introducir el *human-in-loop*. Puede usarse para *approval*, *editing* y *debugging*. También se puede generar *breakpoints* dinámicos, estos interrumpen la ejecución del grafo si ocurre una condición especial que se haya captado (estados que queremos que sean ilegales, errores, etc).
- **Paralelización**.
- **Subgrafos**: La idea es tener varios subgrafos que realizan tareas concretas y que luego se comunican con el padre mediantes **claves superpuestas** para unificar todo. 