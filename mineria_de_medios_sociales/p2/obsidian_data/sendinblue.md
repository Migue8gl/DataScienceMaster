# Concepto
- *Sendinblue* (*ahora se llama Brevo*) es una plataforma de *marketing* digital todo en uno que ofrece una amplia gama de herramientas para ayudar a las empresas a conectarse con sus clientes y aumentar su base de clientes. Ofrece soluciones de *email* *marketing*, *SMS* *marketing*, chat en vivo, automatización de marketing, publicidad en *Facebook*, y más. Además, Sendinblue también tiene una **API** abierta que permite la integración con otras herramientas de marketing y sistemas de gestión de relaciones con clientes ([[crm]]). Con su enfoque en la automatización y la personalización, *Sendinblue* es una herramienta popular para pequeñas y medianas empresas que buscan mejorar su marketing y su relación con los clientes.
- Utiliza una [[rest api]] que nos permite ejecutar una cantidad muy amplia de operaciones.

# Funcionalidad
- La API de #sendinblue nos permite:
	- *Email* API: crear y mandar emails que se disparan con acciones tales como confirmaciones de pedido, reseteo de contraseñas y confirmación de emails.
	- Campaña de *Email*: crear y mandar campañas con buen diseño. También permite generar reportes para las campañas previamente programadas y compartirlos con direcciones de correo específicas.
	- Otras: *sms*, contactos, automatización etc.
## Acceso a la API
- La API es accedida mediante [[http requests]] a una versión específica de la **URL** del [[endpoint]]. Solo se accede mediante [[ssl]].
- Las peticiones deben contener un paquete *JSON*, las cabeceras `content-type:  application/json` y una `api-key`.
## Documentación API
- [sendinblue](https://developers.sendinblue.com/docs/getting-started)
### Ejemplo acceso con PHP
```php
<?php
// Endpoint que deseas probar
$url = "https://api.sendinblue.com/v3/account";

// Headers necesarios para autenticación y formato de la respuesta
$headers = array(
    "Content-Type: application/json",
    "api-key: tu_api_key"
);

// Inicializa el objeto cURL
$ch = curl_init();

// Configura la URL y los headers
curl_setopt($ch, CURLOPT_URL, $url);
curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);

// Establece las opciones adicionales para la solicitud
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);

// Ejecuta la solicitud
$response = curl_exec($ch);

// Verifica si la solicitud fue exitosa
if ($response === false) {
    echo "Error: " . curl_error($ch);
} else {
    // Procesa la respuesta
    $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    if ($http_code == 200) {
        // La petición ha sido exitosa, imprime la respuesta
        echo $response;
    } else {
        // La petición no ha sido exitosa, imprime el código de error
        echo "Error: " . $http_code;
    }
}

// Cierra la conexión cURL
curl_close($ch);
?>
```
- Obviamente lo recomendado es usar la **SDK** oficial, proporcionando una interfaz más sencilla. Ej:
```php
<?php
require_once(__DIR__ . '/vendor/autoload.php'); // Path donde se encuentra el archivo autoload.php

use SendinBlue\Client\Configuration;
use SendinBlue\Client\ApiException;
use SendinBlue\Client\Api\AccountApi;

// Configura tu clave de API
$config = Configuration::getDefaultConfiguration()->setApiKey('api-key', 'tu_api_key');

// Crea una instancia del cliente API
$apiInstance = new AccountApi(null, $config);
try {
    // Llama al método getAccount para obtener información de tu cuenta
    $result = $apiInstance->getAccount();

    // Imprime la respuesta
    print_r($result);
} catch (ApiException $e) {
    echo 'Exception when calling AccountApi->getAccount: ', $e->getMessage(), PHP_EOL;
}
?>
```