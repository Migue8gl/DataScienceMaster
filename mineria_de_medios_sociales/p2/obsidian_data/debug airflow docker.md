- #aicore | #docker | #airflow

- Para poder *debuggear* en airflow una **etl** corriendo [[docker]] en PyCharm seguir los siguientes pasos:
	- Crear configuración para docker.
		- Server: Docker
		- Compose files: compose.yaml
	- Crear configuración de Python Debug Server.
		- IDE host name: 172.17.0.1
		- Port: 9000
		- Path Maps: 
		![[path-map-debug-airflow-pycharm.png]]
	- Donde se quiera *debuggear* añadir:
	```
				import subprocess  
				try:
					import pydevd_pycharm    
				except ImportError:    
				    subprocess.run(["pip", "install", "pydevd-pycharm~=241.17890.14"], check=True)    
				    import pydevd_pycharm    
				pydevd_pycharm.settrace(    
				    '172.17.0.1',    
				    port=9000,    
				    stdoutToServer=True,    
				    stderrToServer=True    
				)
	```
	-  Con ello se parará donde se haya puesto ese código y podremos depurar.
	- Corremos la configuración de docker que hemos creado. Después corremos el debug que hemos creado de Python Server.
	- Accediendo a *localhost:8080* accedemos al airflow local y podemos ejecutar el dag para depurarlo.
- **IMPORTANTE** -> partes del código que sean sensibles deben comentarse y adaptarse un poco el código para aislar aquello que vayamos a depurar.