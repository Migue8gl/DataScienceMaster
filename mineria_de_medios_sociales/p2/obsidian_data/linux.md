- #linux | #bash

## Estructura de directorios
- [[linux directory]]

## Tips
- Para crear instrucciones que puedan ser llamadas desde la terminal:
	- Crear script bash con funcionalidad.
	- Si la funcionalidad implica cambios en algún directorio, añade rutas absolutas en el script para hacer `cd`.  Para obtener ruta absoluta -> `pwd`.
	- identificar un directorio que aparezca en la variable de entorno **PATH**. Para ver **PATH** actual -> `echo $PATH`.
	- Escoger uno, como por ejemplo `/usr/local/bin/` y mover el script a ese directorio.
	- Se puede llamar al script desde terminal invocando su nombre.

## Terminal
- `bg` -> para ver procesos en segundo plano.
- `xdg-open` -> para abrir imagenes desde terminal.
- `pgrep -af "python3 src/main.py"`:
	- pgrep busca los procesos que cumplan con el criterio de búsqueda.
	- -a hace que se muestre la línea completa del comando.
	- -f busca un patrón en los procesos.
- `ps aux | grep common_name | awk '{print $2}' | xargs kill` -> Encontrar procesos que se hayan creado a partir de la misma instrucción y matarlos todos.

## Herramientas
- [[tmux]]
- zsh
- zoxide