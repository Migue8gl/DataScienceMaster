- #nginx

- Nginx es un servidor web ligero de alto rendimiento que puede ser utilizado como [[reverse proxy]], [[load balancer]], mail proxy y [[http cache]].

### Instalación
- Para instalar Nginx debermos introducir las siguientes instrucciones en la terminal de nuestro SO (estas instrucciones en este caso son para Ubuntu) -> `sudo apt-get update && sudo apt-get dist-upgrade && sudo apt-get autoremove`, `sudo apt-get install nginx`,  `sudo systemctl start nginx`.

### Configuración
##### Primeros pasos
- Nginx utiliza la directiva **proxy_pass** para especificar la dirección URL del servidor web al que se deben reenviar las solicitudes del cliente. En nuestro caso podemos usar la directiva **upstream** para redirigir el tráfico a una granja web.
- Si no queremos la funcionalidad web de nginx debemos desactivarla dentro del siguiente archivo: **/etc/nginx/nginx.conf** ya que viene por defecto activa.

##### Puerto 80
- Antes de editar el archivo de configuración debemos aclarar que nginx necesita el puerto 80 libre de procesos o servicios que lo utilicen. Por ello podemos comprobarlo con el siguiente comando -> `netstat –tulpn | grep :80`.

##### Configuración round-robin
- [[round-robin]]
- Editamos el archivo **/etc/nginx/conf.d/default.conf** añadiendo lo siguiente:

```
upstream balanceo_usuarioUGR {  
	server ip_maquinaM1;  
	server ip_maquinaM2;  
}  
server{  
	listen 80;  
	server_name balanceador_usuarioUGR;  
	access_log /var/log/nginx/balanceador_usuarioUGR.access.log;  
	error_log /var/log/nginx/balanceador_usuarioUGR.error.log;  
	root /var/www/;  
	location /  
	{  
		proxy_pass http://balanceo_usuarioUGR;  
		proxy_set_header Host $host;  
		proxy_set_header X-Real-IP $remote_addr;  
		proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;  
		proxy_http_version 1.1;  
		proxy_set_header Connection "";  
	}  
}
```

- En detalle, el archivo de configuración explicado hace lo siguiente:  
	- Establece la directiva upstream para definir los servidores web que manejarán las solicitudes entrantes. Los servidores web se especifican con las direcciones IP de dos máquinas diferentes: ip maquinaM1 y ip maquinaM2. Define un servidor virtual (server) que escucha en el puerto 80 y se llama balanceador usuarioUGR. Esta configuración es para redirigir el tráfico entrante al balanceador de carga.  
	 - Configura los registros de acceso y error para este servidor virtual. Configura la ruta del balanceador (root) para que sea /var/www/.  
	- Establece la directiva location para configurar la ubicaci ́on del servidor virtual.  Todas las solicitudes a la ubicación / se manejarán con un proxy inverso al grupo de servidores definidos en la directiva upstream balanceo usuarioUGR.  
	- La directiva proxy set header se utiliza para establecer la información del encabezado que se debe pasar a los servidores web. En este caso, se establece el host, la dirección IP real del cliente (X-Real-IP) y la dirección IP de reenvío (X-Forwarded-For).  
	- La directiva proxy http version se establece en 1.1 para utilizar HTTP 1.1.  
	- La directiva proxy set header Connection se establece en una cadena vacía para permitir que la conexión siga abierta después de que se haya completado la solicitud. Esto permite que el balanceador de carga mantenga una conexión abierta con el servidor para reducir la sobrecarga de conexión.

##### Otras opciones
- Existen más configuraciones, por prioridad, por ip, con cookies, con timeout etc. 
- Para más información consultar la [documentación](http://nginx.org/en/docs/).
- Otros enlaces de interés: [link1](http://www.cyberciti.biz/tips/using-nginx-as-reverse-proxy.html) y [link2](http://nginx.org/en/docs/http/ngx_http_upstream_module.html).
