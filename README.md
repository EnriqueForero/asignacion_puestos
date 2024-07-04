# Optimización de asignación de puestos de trabajo

## Descripción
Este proyecto implementa un sistema de optimización para la asignación de puestos de trabajo a diferentes equipos, considerando preferencias de ubicación, utilidad y tamaño de los equipos. Utiliza programación lineal para encontrar la asignación óptima y presenta los resultados a través de una interfaz interactiva creada con Streamlit.

## Características
- **Optimización basada en múltiples criterios:** Asigna puestos de trabajo considerando diversas preferencias y restricciones.
- **Interfaz de usuario interactiva:** Permite ajustar parámetros de los equipos de manera dinámica.
- **Visualización gráfica:** Presenta las asignaciones de manera clara y visualmente atractiva.
- **Análisis de cumplimiento:** Evalúa cómo se cumplen las preferencias y cómo se agrupan los equipos.

## Instalación

### Requisitos previos
- Python 3.7+
- pip

### Pasos de instalación
1. Clonar el repositorio:
   ```sh
   git clone https://github.com/EnriqueForero/asignacion_puestos.git
   cd optimizacion-puestos-trabajo
   ```

2. Crear un entorno virtual (opcional pero recomendado):
   ```sh
   python -m venv venv
   source venv/bin/activate  # En Windows use `venv\Scripts\activate`
   ```

3. Instalar las dependencias:
   ```sh
   pip install -r requirements.txt
   ```

## Uso
Para ejecutar la aplicación:
```sh
streamlit run app.py
```

Después de ejecutar el comando, se abrirá una ventana del navegador con la interfaz de la aplicación.

## Estructura del proyecto
```
optimizacion-puestos-trabajo/
│
├── app.py                # Archivo principal de la aplicación Streamlit
├── requirements.txt      # Lista de dependencias del proyecto
├── README.md             # Documentación del proyecto

```

Contribuir a este proyecto es bienvenido y agradecido. Si tienes alguna sugerencia, problema o mejora, no dudes en abrir un 'issue' o enviar un 'pull request'.
