- #iam | #policy | #gcp

# Características
- Las **políticas IAM** (*Identity and Access Management*) en *Google Cloud* son reglas de control de acceso que determinan qué permisos tienen los usuarios (o entidades) sobre los recursos en la plataforma. Estas políticas ayudan a gestionar la seguridad y el acceso, permitiendo definir quién puede hacer qué con los diferentes servicios y datos en *Google Cloud*.
## Roles y permisos
- **Permisos**: Son las acciones específicas que se pueden realizar en un recurso. Ejemplos: `compute.instances.list` (listar instancias de Compute Engine) o `storage.buckets.create` (crear un bucket en *Google Cloud* Storage).
- **Roles**: Son conjuntos de permisos. En lugar de asignar permisos uno a uno, se agrupan en roles, lo que facilita la gestión. *Google Cloud* ofrece:
    - **Roles predefinidos**: Diseñados para tareas específicas, como `Editor` o `Viewer`.
    - **Roles básicos**: Acceso a nivel amplio como `Owner`, `Editor`, y `Viewer`.
    - **Roles personalizados**: Roles definidos por el usuario para necesidades específicas de permisos.
## Componentes de una política IAM
- **Principals**: Son las entidades que reciben permisos. Pueden ser usuarios, grupos, cuentas de servicio o dominios. Se identifican por correo electrónico.
- **Bindings**: Es la asignación entre un "principal" y un rol. Un `binding` vincula un conjunto de "principals" con un rol específico en un recurso.
- **Conditions**: Son reglas opcionales que limitan cuándo se puede aplicar un permiso específico. Por ejemplo, puedes restringir accesos a determinadas horas o ubicaciones.
## Niveles de políticas
- Las políticas se pueden definir en distintos niveles de la jerarquía de recursos de *Google Cloud*:
    - **Organización**: Afecta a todos los recursos en la cuenta de organización.
    - **Folder**: Afecta a los recursos en una carpeta específica.
    - **Proyecto**: Afecta solo a los recursos dentro de un proyecto.
    - **Recurso**: Afecta a un recurso específico, como una máquina virtual o un bucket.