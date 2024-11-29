"Crea un vector de número impares entre el 1 y el 30 (ambos inclusive)"
c(seq(1, 30, 2), 30)


"Crea los siguientes vectores:
  1- un vector del 1 al 20
  2- un vector del 20 al 1
  3- un vector con el patrón 1,2,3,...,19,20,19,19,...,2,1"
seq(1,20)
seq(20,1)
c(seq(1,20), seq(19,1))
c(1:20, 19:1) # Otra forma

"Crea una secuencia de números del 1 al 30 con un incremento del 0.5"
seq(1,30,0.5)

"Crea una secuencia que contenga las cuatro primeras letras del abecedario 
en minúscula 3 veces cada una (a a a b b b c c c d d d)"
rep(letters[1:3], each=3)

"Crea el vector numérico x con los valores 2.3, 3.3, 4.3 y accede al 
segundo elemento del vector"
x <- seq(2.3,4.3,1)
x
x[2]

"Crea un vector numérico z que contenga los números del 1 al 10. 
Cambia la clase del vector forzando que sea de tipo carácter. Después cambia 
el vector z a numerico otra vez"
z <- c(1:10)
z
z <- as.character(z)
z
z <- as.numeric(z)
z

"Crea un vector numérico con valores no ordenados usando la función sample(). 
Una vez creado ordena el vector de forma ascendente usando la función sort(). 
¿Si quisieras invertir el orden de los elementos del vector que 
función utilizarías?"

x <- sample(10)
x
sort(x)
sort(x)[length(x):1] # First way
rev(sort(x)) # Second way

"Crea un vector x que contenga los elementos -5,-1,0,1,2,3,4,5,6. 
Escribe un código del tipo x[algo], para extraer:
    1- elementos de x menores que 0,
    2- elementos de x menores o igual que 0,
    3- elementos of x mayor o igual que 3,
    4- elementos de x menor que 0 o mayor que 4,
    5- elementos de x mayor que 0 y menor que 4,
    6- elementos de x distintos de 0"

x <- c(-5,-1,0,1,2,3,4,5,6)
x[x<0]
x[x<=0]
x[x>=3]
x[x<0|x>4]
x[x>0&x<4]
x[x!=0]

"Crea los siguientes vectores x<-month.name[1:6] y z<-month.name[4:10] a 
partir del vector original month.name. Recupera los valores idénticos 
entre los vectores x y z usando %in%"

x <- month.name[1:6]
z <- month.name[4:10]
x[x %in% z]

"R permite extraer elementos de un vector que satisfacen determinadas 
condiciones usando la función subset(). Para el vector x <- c(6,1:3,NA,12) 
calcula los elementos mayores que 5 en x usando:
    1- el filtrado normal, es decir, con el operador >
    2- la función subset()"

x <- c(6,1:3,NA,12)
x[x>5]
subset(x, x>5)



