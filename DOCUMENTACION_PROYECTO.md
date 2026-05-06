# Práctica 4: Calidad y Minería de Datos en Python

## Análisis de Delitos Informáticos en Colombia

---

# Introducción y Objetivo

## ¿De qué trata este proyecto?

Este proyecto implementa un sistema inteligente de análisis de datos para predecir si un caso de delito informático en Colombia está vinculado con criminalidad organizada. Utiliza técnicas de aprendizaje automático (Machine Learning) para ayudar a las autoridades judiciales a identificar rápidamente aquellos casos que requieren mayor atención y recursos.

## Objetivo Principal

El objetivo es construir un modelo predictivo confiable que, basándose en características del proceso judicial (estado del caso, etapa procesal, ley aplicable, ubicación geográfica, entre otros), pueda predecir si un delito informático corresponde a criminalidad organizada o no. Esto permite a las autoridades priorizar y gestionar mejor sus recursos de investigación.

## ¿Por qué es importante?

En la era digital, los delitos informáticos son cada vez más complejos y frecuentes. No todos tienen el mismo nivel de gravedad. Algunos son actos aislados, mientras que otros forman parte de redes criminales organizadas. Identificar automáticamente estos últimos permite a la justicia actuar de manera más rápida y eficiente.

---

# Descripción del Dataset

## Origen de los Datos

Los datos provienen de registros judiciales reales de Colombia relacionados con delitos informáticos. El archivo original se llama *Delitos_Informaticos_V1_20260216.csv* y contiene información consolidada de casos procesados.

## Tamaño y Características

| Característica | Valor |
|---|---|
| Número de casos | 63,705 registros |
| Variables originales | 17 columnas |
| Variable objetivo | CRIMINALIDAD (SI / NO) |

## Información Capturada

El dataset contiene información sobre:

- **Estado del proceso**: Si fue archivado, si hay preclusión, si está activo o inactivo
- **Etapa procesal**: En qué fase se encuentra el caso (indagación, investigación, juicio, ejecución, etc.)
- **Ley aplicable**: Qué normativa se utilizó en el proceso (LEY 906, LEY 1098, etc.)
- **Ubicación**: País y departamento donde ocurrieron los hechos
- **Años**: Cuándo ocurrió el hecho y cuándo entró al sistema
- **Delitos**: Tipo y grupo de delito cometido
- **Otros**: Total de procesos asociados, datos de consumación, etc.

## Desbalance de Clases

Un aspecto importante es que el dataset está desbalanceado: aproximadamente el **78% de los casos** corresponden a criminalidad SÍ (organizada) y el **22% a criminalidad NO**. Esto se tiene en cuenta durante el entrenamiento de los modelos para evitar sesgos.

---

# Metodología del Proyecto

## Estructura General

El proyecto se divide en **tres fases claramente definidas**, cada una representada por un Notebook de Jupyter:

1. **Fase 1: Calidad de Datos** — Análisis exhaustivo y limpieza del dataset
2. **Fase 2: Modelado Predictivo** — Entrenamiento y evaluación de modelos de aprendizaje automático
3. **Fase 3: Despliegue** — Creación de una interfaz web para hacer predicciones

## Tecnologías Utilizadas

- **Python**: Lenguaje de programación principal
- **Pandas**: Manipulación y análisis de datos
- **Scikit-learn**: Biblioteca de aprendizaje automático
- **Matplotlib y Seaborn**: Visualización de datos
- **Ydata-profiling**: Análisis automático de calidad de datos
- **Gradio**: Creación de interfaz web interactiva
- **Jupyter Notebook**: Entorno de desarrollo interactivo

---

# Notebook 1: Calidad de Datos

## ¿Qué es la calidad de datos?

La calidad de datos es fundamental en cualquier proyecto de análisis. Antes de construir modelos, es crucial asegurarse de que los datos sean confiables, completos y consistentes. Un dataset de mala calidad conducirá a resultados imprecisos.

## Paso a Paso del Notebook 1

### Paso 1: Crear la estructura de carpetas

Se crean las carpetas *outputs/reportes*, *outputs/datasets* e *outputs/imagenes* para organizar todos los resultados de forma ordenada y profesional.

### Paso 2: Instalar dependencias

Se instalan las librerías necesarias, especialmente **ydata-profiling** que genera análisis automáticos e interactivos de calidad.

### Paso 3: Cargar el dataset original

Se carga el archivo CSV original con los 63,705 casos de delitos informáticos. Se verifica que todas las columnas estén presentes y se inspecciona el contenido para detectar problemas evidentes.

### Paso 4: Generar reporte de perfilado

Ydata-profiling analiza automáticamente cada columna y genera un **reporte HTML detallado** con:
- Estadísticas descriptivas (media, mediana, desviación estándar)
- Distribuciones de cada variable
- Número de valores faltantes y duplicados
- Alertas automáticas de calidad
- Correlaciones entre variables

Este reporte se guarda en *outputs/reportes/reporte_perfilado_datos.html*.

### Paso 5: Análisis exploratorio visual

Se crean visualizaciones gráficas para entender mejor los datos:
- Gráficos de completitud (qué porcentaje de datos está disponible en cada columna)
- Análisis de outliers o valores anómalos
- Distribuciones de variables principales

### Paso 6: Limpieza y transformación

Se realizan las siguientes acciones:
- Normalizar nombres de columnas (se eliminan tildes: AÑO → ANO)
- Manejar valores faltantes (llenarlos o eliminarlos según corresponda)
- Eliminar duplicados
- Convertir tipos de datos incorrectos

### Paso 7: Feature Engineering

Se crean nuevas variables a partir de las existentes para capturar información más relevante. Por ejemplo:
- Se crea *PERIODO_HECHO* agrupando años en períodos temáticos
- Esto ayuda a capturar tendencias históricas del fenómeno

### Paso 8: Guardar dataset limpio

El dataset procesado y listo se guarda como *outputs/datasets/Delitos_Informaticos_Limpio.csv* para ser usado en el siguiente notebook.

## Salidas del Notebook 1

- **reporte_perfilado_datos.html**: Reporte interactivo con análisis completo de calidad (>5 MB)
- **Delitos_Informaticos_Limpio.csv**: Dataset limpio y listo para modelado
- **Gráficos exploratorios**: completitud.png, outliers_total_procesos.png, distribucion_variables.png

**Tiempo estimado de ejecución: 15 minutos**

---

# Notebook 2: Modelado Predictivo

## ¿Qué es un modelo predictivo?

Un modelo predictivo es un algoritmo que **aprende patrones** de los datos históricos para hacer predicciones sobre nuevos casos. En nuestro caso, el modelo aprenderá características de casos previos para predecir si nuevos casos corresponden a criminalidad organizada.

## Paso a Paso del Notebook 2

### Paso 1: Instalar y preparar librerías

Se instalan versiones compatibles de scikit-learn, numpy y scipy para evitar conflictos. Se importan todas las librerías necesarias para entrenamiento de modelos.

### Paso 2: Cargar dataset limpio

Se carga el archivo *Delitos_Informaticos_Limpio.csv* generado en el Notebook 1.

### Paso 3: Seleccionar características (features)

Se seleccionan las **11 variables más relevantes** que serán usadas para las predicciones:

- ES_ARCHIVO (¿fue archivado?)
- ES_PRECLUSION (¿hubo preclusión?)
- ESTADO (¿está activo o inactivo?)
- ETAPA_CASO (etapa del proceso)
- LEY (ley aplicada)
- PAIS_HECHO (país donde ocurrió)
- DEPARTAMENTO_HECHO (departamento)
- ANO_HECHOS (año del hecho)
- ANO_ENTRADA (año de entrada al sistema)
- TOTAL_PROCESOS (procesos asociados)
- PERIODO_HECHO (período temporal)

### Paso 4: Codificar variables categóricas

Muchas variables son categóricas (texto). Se convierten a números usando **LabelEncoder** para que los modelos puedan procesarlas. Por ejemplo:
- SI/NO → 1/0
- ACTIVO/INACTIVO → 1/0
- ANTIOQUIA/BOGOTÁ/etc → 0/1/2/etc

### Paso 5: Dividir datos

Se divide el dataset en dos partes:
- **80% para entrenamiento**: El modelo aprende de estos datos
- **20% para prueba**: Se evalúa en datos nunca vistos

Se usa stratification para mantener la proporción de clases en ambos conjuntos.

### Paso 6: Escalar datos

Algunos modelos requieren que los datos estén en escalas similares. Se usa **StandardScaler** para transformar todas las variables a media 0 y desviación estándar 1.

### Paso 7: Entrenar 5 modelos diferentes

Se entrenan **cinco algoritmos distintos** con características y parámetros diferentes:

1. **Árbol de Decisión**: Crea un árbol de reglas lógicas. Muy interpretable pero puede sobreajustarse.
2. **KNN (K-Nearest Neighbors)**: Clasifica basándose en los vecinos más cercanos. Simple pero lento.
3. **Red Neuronal (MLP)**: Simula el cerebro con capas interconectadas. Muy flexible.
4. **SVM (Support Vector Machine)**: Encuentra el hiperplano óptimo que separa clases. Excelente con datos complejos.
5. **Random Forest**: Combina múltiples árboles. Robusto y generalmente excelente.

### Paso 8: Evaluar modelos

Se comparan los cinco modelos usando métricas estándar:
- **Accuracy**: Porcentaje de predicciones correctas
- **Precision**: De las predicciones SÍ, cuántas fueron correctas
- **Recall**: De los casos SÍ reales, cuántos se detectaron
- **F1-Score**: Balance entre Precision y Recall

Se identifica el mejor modelo: **Random Forest** con F1-Score de ~0.82.

### Paso 9: Optimizar el mejor modelo

Se usa **GridSearchCV** para probar automáticamente **cientos de combinaciones** de parámetros en Random Forest. El objetivo es encontrar la configuración óptima que maximice el F1-Score.

Este proceso prueba:
- Número de árboles: 100, 200, 300
- Profundidad máxima: 5, 10, 15, None
- Muestras mínimas para dividir: 2, 5, 10
- Características: sqrt, log2

Total: 4 × 4 × 3 × 2 = 96 combinaciones × 3 validaciones = 288 entrenamientos.

### Paso 10: Guardar artefactos

Se guardan cinco archivos en la carpeta *modelos/*:
- **modelo_final.pkl**: Modelo Random Forest optimizado
- **scaler.pkl**: StandardScaler para normalizar datos
- **encoders.pkl**: LabelEncoders para convertir categorías a números
- **le_target.pkl**: Codificador especial para la variable objetivo
- **features.pkl**: Lista de las 11 características usadas

Estos archivos serán usados en el Notebook 3.

## Salidas del Notebook 2

- **modelo_final.pkl, scaler.pkl, encoders.pkl, le_target.pkl, features.pkl**: Artefactos para despliegue
- **comparacion_modelos.png**: Gráfico comparativo de los 5 modelos
- **matrices_confusion.png**: Matrices de confusión (muestra errores)
- **feature_importance.png**: Variables más importantes para predicciones
- **arbol_decision.png**: Visualización del árbol de decisión

**Tiempo estimado de ejecución: 30 minutos** (puede variar según hardware)

---

# Notebook 3: Despliegue e Interfaz

## ¿Qué es despliegue?

Despliegue es el proceso de llevar un modelo desde el entorno de desarrollo a un entorno donde **usuarios reales** pueden interactuar con él. En este caso, se crea una interfaz web usando **Gradio** que permite a cualquier persona hacer predicciones sin necesidad de conocer Python.

## Paso a Paso del Notebook 3

### Paso 1: Instalar Gradio

Se instala la librería Gradio que permite crear interfaces web interactivas de forma muy simple (sin necesidad de HTML/CSS/JavaScript).

### Paso 2: Importar librerías

Se importan todas las herramientas necesarias, incluyendo Gradio para la interfaz y joblib para cargar los modelos guardados.

### Paso 3: Cargar los artefactos del modelo

Se cargan los cinco archivos guardados en el Notebook 2:
- El modelo entrenado
- Los escaladores
- Los codificadores
- La lista de características

### Paso 4: Crear función de predicción

Se define una función que:
1. Recibe los parámetros del caso ingresados por el usuario
2. Los transforma al formato que el modelo entiende (codificación, escalado)
3. Ejecuta el modelo para obtener la predicción
4. Retorna el resultado junto con probabilidades e interpretación

### Paso 5: Construir interfaz gráfica

Usando Gradio, se crea una interfaz web profesional con:
- **Campos de entrada** para cada variable (menús, sliders, números)
- **Botones** para ejecutar predicción y limpiar campos
- **Campos de salida** para mostrar resultados

La interfaz se divide en secciones lógicas:
- "Información del Proceso"
- "Información Geográfica y Temporal"

### Paso 6: Agregar ejemplos

Se agregan ejemplos predefinidos que los usuarios pueden probar con un solo clic. Esto permite que se familiaricen con la interfaz sin llenar manualmente todos los campos.

### Paso 7: Lanzar la aplicación

Se ejecuta `demo.launch()` que automáticamente:
1. Abre la interfaz en el navegador local: **http://127.0.0.1:7860**
2. Genera un enlace público: **https://xxxxxxxxxx.gradio.live** (válido por 72 horas)

El segundo enlace es útil para compartir con otros usuarios remotos.

## Funcionalidades de la Interfaz

### Campos de Entrada

Los usuarios deben ingresar información sobre el caso judicial:

- ¿Fue archivado? (Menú: SI / NO)
- ¿Hubo preclusión? (Menú: SI / NO)
- Estado del proceso (Menú: ACTIVO / INACTIVO)
- Etapa del caso (Menú: INDAGACIÓN, INVESTIGACIÓN, JUICIO, etc.)
- Ley aplicada (Menú: LEY 906, LEY 1098, etc.)
- País del hecho (Menú: COLOMBIA, SIN DATO)
- Departamento (Menú: ANTIOQUIA, BOGOTÁ, VALLE DEL CAUCA, etc.)
- Año del hecho (Slider: 2000-2026)
- Año de entrada al sistema (Slider: 2000-2026)
- Total de procesos (Campo numérico)
- Período del hecho (Menú: ANTES_2015, 2016_2019, 2020_2022, 2023_ADELANTE)

### Resultado de la Predicción

Después de presionar el botón "Predecir Criminalidad", la interfaz muestra:

- **La predicción**: "CRIMINALIDAD ORGANIZADA: SÍ" o "CRIMINALIDAD ORGANIZADA: NO"
- **El nivel de riesgo**: ALTO RIESGO, RIESGO MODERADO, o BAJO RIESGO
- **Las probabilidades**: Porcentaje de confianza para cada clase
- **Interpretación**: Explicación breve del resultado

### Cómo Usar la Interfaz

1. Selecciona los valores desde los menús desplegables o campos de entrada
2. Llena los campos con información del caso que deseas analizar
3. Presiona el botón "Predecir Criminalidad" (color azul)
4. En **segundos**, el modelo procesará los datos y mostrará el resultado
5. Si deseas otra predicción, presiona "Limpiar" para resetear
6. Puedes probar los ejemplos predefinidos para familiarizarte

### Acceso a la Interfaz

| Tipo de Acceso | URL |
|---|---|
| Local (tu computadora) | http://127.0.0.1:7860 |
| Público (para compartir) | https://xxxxxxxxxx.gradio.live (válido 72 horas) |

## Salidas del Notebook 3

- Interfaz web interactiva en tiempo real
- Enlace público para compartir con otros usuarios
- Predicciones instantáneas basadas en el modelo

**Tiempo estimado de ejecución: 5 minutos**

---

# Resultados y Recomendaciones

## Desempeño del Modelo Final

| Métrica | Valor |
|---|---|
| Accuracy (Precisión General) | ~82% |
| Precision | ~83% |
| Recall | ~82% |
| F1-Score | ~82% |

## Variables Más Influyentes

El modelo identificó las siguientes variables como las más importantes para hacer predicciones:

1. **TOTAL_PROCESOS**: El número de procesos asociados al caso
2. **DEPARTAMENTO_HECHO**: La ubicación geográfica (departamento)
3. **ANO_HECHOS**: El año en que ocurrió el delito
4. **ESTADO**: Si el proceso está activo o inactivo
5. **ETAPA_CASO**: La etapa procesal (indagación, investigación, etc.)

## Interpretación de Resultados

El modelo alcanza un **82% de efectividad**, lo que es considerado un **buen resultado** en tareas de clasificación. Esto significa que en 82 de cada 100 casos, el modelo hará la predicción correcta. Las métricas están balanceadas, indicando que el modelo no está sesgado hacia una clase en particular.

## Limitaciones y Recomendaciones

### Limitaciones

1. **Dataset histórico**: El modelo se basa en datos pasados. Nuevos patrones criminales que no estén representados podría no ser detectados.

2. **Variables limitadas**: El modelo solo usa 11 variables. Agregar más información contextual podría mejorar el desempeño.

3. **Desbalance de clases**: El dataset está desbalanceado (78% SÍ, 22% NO). Aunque se aplicó compensación, técnicas adicionales podrían ayudar.

4. **Cambios en el tiempo**: La criminalidad evoluciona. El modelo debería reentrenarse periódicamente con nuevos datos.

### Próximos Pasos Sugeridos

- Recolectar más datos para expandir el dataset
- Incorporar nuevas variables relevantes
- Probar arquitecturas de redes neuronales más complejas
- Implementar ensemble methods más avanzados
- Establecer un pipeline de actualización periódica del modelo
- Crear dashboards adicionales para análisis en tiempo real
- Integrar el modelo en sistemas judiciales reales

---

# Conclusión

Este proyecto demuestra el poder del aprendizaje automático aplicado a problemas reales. A través de tres notebooks bien estructurados, se implementó un **sistema completo** que va desde:

1. **Calidad de Datos** → Asegurar que los datos sean confiables
2. **Modelado Predictivo** → Entrenar y optimizar el mejor modelo
3. **Despliegue** → Crear una interfaz web amigable para usuarios

El modelo **Random Forest** optimizado alcanza un **82% de efectividad** en la predicción de criminalidad organizada, lo que representa una **herramienta valiosa** para las autoridades judiciales. La interfaz Gradio permite que usuarios sin conocimientos técnicos puedan hacer predicciones de forma intuitiva.

Aunque hay espacio para mejora, este proyecto establece las bases sólidas para un **sistema de apoyo a la toma de decisiones** en investigaciones de delitos informáticos. Con datos más completos, variables adicionales y re-entrenamientos periódicos, el desempeño podría mejorarse significativamente.

---

**Documento generado: Mayo 2026**  
**Curso: Analítica de Datos**  
**Práctica 4: Calidad y Minería de Datos en Python**
