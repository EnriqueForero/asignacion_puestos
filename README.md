Ciertamente, aquí tienes un README.md bien estructurado y un archivo de requerimientos para tu proyecto:

```markdown
# Optimización de Asignación de Puestos de Trabajo

## Descripción
Este proyecto implementa un sistema de optimización para la asignación de puestos de trabajo a diferentes equipos, considerando preferencias de ubicación, utilidad y tamaño de los equipos. Utiliza programación lineal para encontrar la asignación óptima y presenta los resultados a través de una interfaz interactiva creada con Streamlit.

## Características
- Optimización de asignaciones basada en múltiples criterios
- Interfaz de usuario interactiva para ajustar parámetros de equipos
- Visualización gráfica de las asignaciones
- Análisis de cumplimiento de preferencias y agrupación de equipos

## Instalación

### Requisitos previos
- Python 3.7+
- pip

### Pasos de instalación
1. Clonar el repositorio:
   ```
   git clone https://github.com/tu-usuario/optimizacion-puestos-trabajo.git
   cd optimizacion-puestos-trabajo
   ```

2. Crear un entorno virtual (opcional pero recomendado):
   ```
   python -m venv venv
   source venv/bin/activate  # En Windows use `venv\Scripts\activate`
   ```

3. Instalar las dependencias:
   ```
   pip install -r requirements.txt
   ```

## Uso
Para ejecutar la aplicación:
```
streamlit run app.py
```

Después de ejecutar el comando, se abrirá una ventana del navegador con la interfaz de la aplicación.

## Estructura del proyecto
- `app.py`: Contiene el código principal de la aplicación Streamlit.
- `requirements.txt`: Lista de dependencias del proyecto.
- `README.md`: Este archivo, con información sobre el proyecto.

## Contribuir
Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios mayores antes de hacer un pull request.

## Licencia
[MIT License](https://opensource.org/licenses/MIT)
