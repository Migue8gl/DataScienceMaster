- #elastic | #queries

# Búsqueda por frase (match phrase)
- Cuando se quiere hacer un *match* con una frase en concreto, donde el orden de las palabras viene definido por la propia frase. Se utiliza **match_phrase**:
```bash
GET /megacorp/employee/_search
{
"query":{
    "match_phrase":{
      "about": "rock climbing "
    }
  }
}
```
- Incluso podemos hacer un **highlight** de la parte con la cual coincide esta frase, en concreto solamente tendríamos que añadir el siguiente campo:
```bash
"highlight":{
  "fields":{
    "about" : {}
  }
}
```
- Se necesita la opción `index_options: positions` en el índice. No se puede cambiar a no ser que se reindexe todo en otro índice con otra configuración.

# Analíticas (Aggregations)
- [Agregaciones](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations.html)
- Se pueden analizar los valores que posee un índice en algunos de sus campos por medio de operaciones de **agregación**. Por ejemplo, operaciones como `avg`:
```bash
GET /megacorp/employee/_search
{
 "aggs" : {
  "all_interests" : {
    "terms" : { "field" : "interests" },
    "aggs" : {
    "avg_age" : {
        "avg" : { "field" : "age" }
      }
    }
   }
 }
}
```

# Actualizar información sin reindexar
- [Painless scripting](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-scripting-using.html)
- Se puede actualizar información de un documento sin tener que reindexarlo. Esto se hace por medio de los *scripts* de *painless*. Ejemplo de llamada a un *script*:
```bash
POST /website/blog/1/_update
{
 "script" : "ctx._source.tags+=new_tag",
 "params" : {
 "new_tag" : "search"
 }
}
```