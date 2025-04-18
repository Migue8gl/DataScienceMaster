- #xmlhttprequest | #xhr

- El objeto XHR proporciona una API para crear solicitudes HTTP asincrónicas y recibir respuestas del servidor sin tener que recargar la página.

- Para utilizar el objeto XMLHttpRequest, los desarrolladores pueden crear una instancia del objeto XMLHttpRequest y luego establecer los parámetros de la solicitud, como la URL de la solicitud, el método HTTP utilizado y los datos que se enviarán con la solicitud. La solicitud se envía mediante el método `send()` del objeto XMLHttpRequest, y cuando se recibe la respuesta del servidor, se maneja utilizando los métodos `onload()` y `onreadystatechange()` del objeto.

- El XHR cliente tiene un estado (no confundir con el status de la petición, ej: 200) y ese estado o #readystatus cambia cuando:
	- `0` -> objeto creado.
	- `1` -> se ha llamado a `open()`.
	- `2` -> se ha llamado a `send()`.
	- `3` -> la respuesta está siendo recibida.
	- `4` -> operación completada.