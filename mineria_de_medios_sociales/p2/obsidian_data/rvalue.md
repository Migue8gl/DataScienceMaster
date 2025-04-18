- #rvalue

- En C++, un rvalue es una expresión que se evalúa en un valor, pero no tiene una ubicación en memoria asignable. En otras palabras, un rvalue es una expresión que representa un valor temporal que no se puede asignar a otro valor.

- Estos valores a su vez se dividen en dos, #pvalue, que es un rvalue puro, como por ejemplo `true` o `10`. También las  [[runtime-modernc++#Expresiones Lambda]]. Las variables temporales devueltas por no-referencias, las variables temporales generadas por expresiones de operación, los literales originales y las expresiones lambda son todos valores rvalue puros.

- #xvalue es un rvalue que introduce las referencias a rvalues. Es un valor que es destruido, pero puede ser movido.