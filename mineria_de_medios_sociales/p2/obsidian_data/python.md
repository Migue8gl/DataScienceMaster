- #python 

- [[ptc]] - Asignatura de programación técnica y científica.
- [[pytest]] - Test en python.
- [[cookiecutter]] - Dependencia para crear templates de proyectos.

# Tips
- Usar `python -m pip` en vez de `pip` directamente, así se asegura que se instala usando la versión de pip asociada al interprete que se desee.
- Los parámetros de una función se almacenan como objetos individuales en memoria, de forma que no se crean más de uno cada vez que se llama a una función o se crean varias. De esta forma, los parámetros inmutables cuando son modificados internamente (como un número entero) en realidad no son modificados, sino que se crea otro objeto en memoria.
- Las listas son mutables, por tanto, esta si se modificaría directamente, lo cual puede llevar a comportamientos extraños, véase:
```python
def surprise(my_list=[]): 
	print(my_list) 
	my_list.append("X")
```
- Esto hace que cada vez que se llama a la función, añada un elemento string a la lista, pues la lista `my_list` es la misma en cada llamada.
## Crear requiriments.txt
```
pip install pipreqs
pipreqs /path/to/project
```
- También se puede lo siguiente, pero [es mejor](https://stackoverflow.com/questions/31684375/automatically-create-file-requirements-txt) pipreqs: 
```
python -m pip freeze > requirements.txt
```
## Crear un entorno aislado
- Instalar -> `sudo apt install python3-venv`
- Crear el entorno -> `python3 -m venv venv`
- Se crea la siguiente estructura:
	```
		$tree -L 3
		└── venv
		    ├── bin
		    │   ├── activate
		    │   ├── activate.csh
		    │   ├── activate.fish
		    │   ├── Activate.ps1
		    │   ├── easy_install
		    │   ├── easy_install-3.8
		    │   ├── pip
		    │   ├── pip3
		    │   ├── pip3.8
		    │   ├── python -> python3
		    │   └── python3 -> /usr/bin/python3
		    ├── include
		    ├── lib
		    │   └── python3.8
		    ├── lib64 -> lib
		    ├── pyvenv.cfg
		    └── share
		        └── python-wheels
	```
- Para activar el entorno -> `source venv/bin/activate`
- La shell mostrará el siguiento *prompt* -> `(env)`
- Para desactivar al terminar -> `deactivate`
## Cambiar versión de python local
- `pyenv install <py-version>`
- `pyenv local`