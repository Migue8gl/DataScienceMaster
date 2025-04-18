\section{Teoría de la computación}

La teoría de la computación es un campo que estudia los fundamentos matemáticos y los límites de la computación. En esencia, se ocupa de preguntas fundamentales sobre qué se puede y qué no se puede computar, cómo se pueden resolver problemas de manera eficiente y qué tan difíciles son ciertos problemas en términos de recursos computacionales.\\[6pt]

Suele usarse la máquina de Turing, como modelo abstracto de computadora, para las demostraciones realizadas, creando un marco sencillo en el que entender cómo se ejecutan los algoritmos y cómo se resuelven estos.

\subsection{Máquina de Turing}

Una máquina de Turing, cuyo nombre proviene de su creador Alan Turing, es un modelo de computación matemático que es capaz de ejecutar algoritmos mediante el control de una serie de símbolos en la tira de una cinta y de acuerdo a una serie de reglas establecidas~\cite{stone1972}.\\[6pt]

Es capaz de general un modelo abstracto mediante el cual representar todo de algoritmos que resuelvan problemas computables.

  

\subsection{Grupos de complejidad P y NP}\label{complexity}

Se puede definir un lenguaje como una serie de cadenas dentro de un alfabeto, siendo este último un conjunto finito de símbolos~\cite{johnjeffery_automata}. \\[6pt]

Dada esta definición se dice que un lenguaje que es \textit{reconocible}, es decir, que una máquina de Turing puede determinar si una cadena pertenece al lenguaje o no, en un tiempo ``polinómico determinista" son conocidos como problema de la clase $P$. Que un problema sea resoluble en tiempo polinómico determinista en una máquina de Turing, significa que para cada estado solo existe una posible acción y que su tiempo de resolución puede expresarse como una función polinómica~\cite{johnjeffery_automata}. \\[6pt]

En contraposición, un problema de la clase $NP$ no es resoluble en tiempo polinómico determinista, sino en tiempo polinómico no determinista. Esto es, que puede resolverse expresando el tiempo como una función polinómica, pero con la diferencia de que el no determinismo implica que para pasar de un estado a otro es posible considerar múltiples acciones.\\[6pt]

La diferencia entre los problemas de las clases $P$ y $NP$ es fundamental en la teoría de la complejidad computacional. Mientras que los problemas de la clase $P$ son eficientemente resueltos, los problemas de la clase $NP$ pueden ser verificados en tiempo polinómico, pero su resolución en tiempo polinómico no está demostrada y tiende a ser exponencial.

  

\subsection{NP-Hard}

Un problema de decisión $H$ se considera $NP$-Hard o duro si, para cada problema $L$ en $NP$, hay una reducción de muchos a uno de tiempo polinómico de $L$ a $H$~\cite{leeuwen_algorithms_1998,johnjeffery_automata}. Esto significa que cualquier problema en $NP$ puede ser transformado en $H$ en tiempo polinómico, lo que sugiere que $H$ es al menos tan difícil como cualquier problema en $NP$.

  

\begin{figure}[H]

\begin{center}

\includegraphics[width=0.6\textwidth]{imagenes/np_subsets.png}

\end{center}

\caption[Relación entre clases de complejidad]{En esta figura se puede ver la relación de pertenencia entre clases de complejidad. Puede observarse que $NP$-Hard no es un subconjunto de $NP$. Una solución para un problema $NP$-Hard podría transformarse mediante reducción y solucionar un problema $NP$, pero no siempre es así al revés.}

\end{figure}
