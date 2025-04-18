- #ajax

- AJAX (Asynchronous JavaScript and XML) es una técnica de programación web que permite actualizar una página web sin tener que recargarla completamente. Se basa en el uso de tecnologías web estándar, como [[html|HTML]], CSS, JavaScript y XML, para crear aplicaciones web interactivas y dinámicas.

- La técnica de AJAX se basa en la capacidad de JavaScript para enviar y recibir datos en segundo plano, sin la necesidad de recargar toda la página. En lugar de enviar una solicitud al servidor web y esperar la respuesta completa, una aplicación web AJAX envía pequeñas solicitudes al servidor en segundo plano, utilizando el objeto [[xmlhttprequest]] de JavaScript, y actualiza solo las partes de la página que necesitan ser actualizadas.

- Para utilizar AJAX en una aplicación web, el desarrollador debe utilizar JavaScript para capturar la acción del usuario, enviar una solicitud AJAX al servidor, procesar la respuesta y actualizar la página dinámicamente sin tener que recargarla completamente.

- Además de XML, AJAX también puede utilizar otros formatos de datos, como JSON, para enviar y recibir datos del servidor. Esto permite a los desarrolladores crear aplicaciones web altamente interactivas y dinámicas que pueden funcionar de manera fluida y sin interrupciones.

- Una de las claves de AJAX es el uso del objeto [[xmlhttprequest]]

- Ej:
```javascript
function executeFunction() {
  var xhttp = new XMLHttpRequest();
  xhttp.onreadystatechange = function() {
    if (this.readyState == 4 && this.status == 200) {
      // Handle the response from the server
      console.log(this.responseText);
    }
  };
  xhttp.open("GET", "example.php?function=doSomething", true);
  xhttp.send();
}
```

- Ejemplo en el que al clickar un botón ejecutamos una función de php.