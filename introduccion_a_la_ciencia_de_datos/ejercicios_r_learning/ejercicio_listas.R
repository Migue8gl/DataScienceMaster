"Crea una lista llamada mi_lista que contenga los siguientes elementos: 
un vector numérico de 15 elementos, un vector de caracteres de 5 elementos 
y un vector de valores booleanos de 10 elementos todos TRUE"

mi_lista <- list(sample(15), sample(letters,5), rep(TRUE, 10))
mi_lista

"Dada la siguiente lista my_list <- list(name=\"Fred\", wife=\"Mary\", 
no.children=3, child.ages=c(4,7,9)):
    1- Imprime los nombres de todos los componentes de la lista
    2- Devuelve el segundo componente de la lista
    3- Recupera el segundo elemento del cuarto componente de la lista
    4- Imprime la longitud del cuarto elemento de la lista
    5- Reemplaza el cuarto elemento de la lista por un vector de 12 numeros del 1 al 12
    6- Elimina el componente wife
    7- Añade un componente más a la lista llamado pepe"

my_list <- list(name="Fred", wife="Mary", no.children=3, child.ages=c(4,7,9))
names(my_list)
my_list[2]
my_list[[4]][2]
length(my_list[[4]])
my_list[[4]] <- 1:12
my_list[[4]]
my_list["wife"] <- NULL
my_list
append(my_list, "Pepe") # Si se quiere un elemento nuevo
my_list$Pepe <- 1 # Una clave llamada Pepe con algún valor
my_list

"Convertir un vector de 30 números postivos y negativos en una lista con 
[1ra componente: los 2 primeros elementos, 2da componente: los 5 siguientes 
pero como caracteres, 3ra componente: los elementos restantes que sean valores 
positivos, 4ta componente: una lista de caracteres con tu nombre y apellidos 
(ej: \"Rocio\" \"Romero\" \"Zaliz\")]
    1- Una vez creado ponles nombre (ej: \"1ro\", \"2do\" etc)
    2- Accede al tercer elemento por su nombre
    3- Fusiona el primer y cuarto componente en un quito componente y 
    borra los originales"

x <- sample(-50:50, 30)
x <- list(
    "1ro"=x[1:2], 
    "2do"=as.character(x[3:7]), 
    "3ro"=x[8:30][x[8:30] > 0],
    "4to"=list("Miguel", "García", "López")
  )
x["3ro"]
x[["5to"]] = c(x[["1ro"]],x[["4to"]])
x
x["1ro"] <- NULL
x["4to"] <- NULL

"Crea una nueva lista cuyos componentes sean las listas de los ejericios 
anteriores (OJO: Su longitude debería ser 3)"

another_list = list(mi_lista, my_list, x)

"Crea una nueva lista que concatene las listas de los tres 
primeros ejericios. ¿Qué longitud tiene?"

another_list = c(mi_lista, my_list, x)
