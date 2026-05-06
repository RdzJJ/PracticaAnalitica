# Práctica 4: Calidad y Minería de Datos en Python

**Curso:** Analítica de Datos  
**Tema:** Calidad de Datos, Modelado Predictivo y Despliegue  
**Valor:** 15% de la nota final

---

## 📋 Descripción del Proyecto

Este proyecto implementa un pipeline completo de **minería de datos** sobre delitos informáticos en Colombia:

1. **Calidad de Datos** — Perfilado, diagnóstico y limpieza del dataset
2. **Modelado Predictivo** — Entrenamiento de 5 modelos de clasificación
3. **Despliegue** — Interfaz web interactiva con Gradio para predicciones en tiempo real

### Dataset
- **Fuente:** Delitos_Informaticos_V1_20260216.csv
- **Tamaño:** 63,705 casos × 17 características
- **Variable Objetivo:** `CRIMINALIDAD` (SI / NO)
- **Problema:** Clasificación binaria balanceada

---

## 🎯 Objetivo

Predecir si un caso de **delito informático** en Colombia tiene carácter de **criminalidad organizada**, permitiendo a las autoridades priorizar recursos de investigación en casos con mayor probabilidad de estar ligados a redes criminales organizadas.

---

## 📁 Estructura del Proyecto

```
PracticaAnalitica/
├── notebook1_calidad_datos.ipynb       # Perfilado y limpieza
├── notebook2_modelos.ipynb             # Entrenamiento de modelos
├── notebook3_despliegue.ipynb          # Interfaz Gradio
├── Instrucciones.md                    # Guía de ejecución
├── README.md                           # Este archivo
│
├── outputs/
│   ├── datasets/
│   │   ├── Delitos_Informaticos_V1_20260216.csv  # Dataset original
│   │   └── Delitos_Informaticos_Limpio.csv       # Dataset limpio
│   ├── reportes/
│   │   └── reporte_perfilado_datos.html          # Análisis de calidad
│   └── imagenes/
│       ├── completitud.png                       # Gráficos exploratorios
│       ├── outliers_total_procesos.png
│       ├── distribucion_variables.png
│       ├── arbol_decision.png                    # Visualización modelos
│       ├── comparacion_modelos.png
│       ├── matrices_confusion.png
│       └── feature_importance.png
│
├── modelos/
│   ├── modelo_final.pkl                # Random Forest optimizado
│   ├── scaler.pkl                      # StandardScaler
│   ├── encoders.pkl                    # LabelEncoders
│   ├── le_target.pkl                   # Encoder variable objetivo
│   └── features.pkl                    # Lista de features

```

---

## 🚀 Cómo Ejecutar

### Paso 1: Instalar Dependencias

```bash
pip install ydata-profiling scikit-learn pandas numpy matplotlib seaborn gradio joblib
```

### Paso 2: Ejecutar Notebook 1 (Calidad de Datos)

```
Abre: notebook1_calidad_datos.ipynb
Ejecuta: Todas las celdas en orden

Genera:
- outputs/reportes/reporte_perfilado_datos.html
- outputs/datasets/Delitos_Informaticos_Limpio.csv
- outputs/imagenes/*.png (gráficos exploratorios)
```

### Paso 3: Ejecutar Notebook 2 (Modelos)

```
Abre: notebook2_modelos.ipynb
Ejecuta: Todas las celdas en orden

Entrena 5 modelos:
1. Árbol de Decisión (max_depth=5)
2. KNN (k=7)
3. Red Neuronal MLP (100,50)
4. SVM (rbf kernel, muestra=15000)
5. Random Forest (200 árboles)

Genera:
- modelos/*.pkl (artefactos para despliegue)
- outputs/imagenes/*.png (visualizaciones de modelos)
```

### Paso 4: Ejecutar Notebook 3 (Despliegue)

```
Abre: notebook3_despliegue.ipynb
Ejecuta: Todas las celdas

Abre automáticamente:
http://127.0.0.1:7860  ← Local
https://xxxxxxxxxx.gradio.live  ← URL pública (72 horas)
```

---

## 📊 Resultados del Mejor Modelo

**Modelo:** Random Forest (optimizado con GridSearchCV)

### Métricas en Test Set

| Métrica | Valor |
|---------|-------|
| Accuracy | ~0.82 |
| Precision | ~0.83 |
| Recall | ~0.82 |
| F1-Score | ~0.82 |

### Features Más Importantes

1. `TOTAL_PROCESOS` — Total de procesos asociados
2. `DEPARTAMENTO_HECHO` — Ubicación geográfica
3. `ANO_HECHOS` — Año del hecho
4. `ESTADO` — Estado del proceso (ACTIVO/INACTIVO)
5. `ETAPA_CASO` — Etapa procesal

### Configuración del Modelo Optimizado

```python
RandomForestClassifier(
    n_estimators=200,        # 200 árboles
    max_depth=10,            # Profundidad máxima
    min_samples_split=5,     # Muestras mínimas para dividir
    class_weight='balanced', # Pesa clases desbalanceadas
    random_state=42
)
```

---

## ⚙️ Tecnologías Utilizadas

- **Python 3.x**
- **pandas** — Manipulación de datos
- **scikit-learn** — Modelos de ML
- **numpy** — Computación numérica
- **matplotlib / seaborn** — Visualizaciones
- **ydata-profiling** — Análisis de calidad automático
- **Gradio** — Interfaz web interactiva
- **joblib** — Serialización de modelos

---

## 📝 Notas de Ejecución

### Tiempo Estimado
- **Notebook 1:** ~15 minutos (generación de perfil HTML)
- **Notebook 2:** ~30 minutos (GridSearch puede tardar)
- **Notebook 3:** ~5 minutos (interfaz web)

### Posibles Ajustes si hay Lentitud

**Si SVM es muy lento:**
```python
n_sample = 8000  # Reducir de 15000
```

**Si GridSearch es muy lento:**
```python
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt']
}
```

---

## 🔍 Validación de Entrega

Antes de entregar, verifica que existan:

✅ Notebooks 1-3 con todas las celdas ejecutadas  
✅ `outputs/reportes/reporte_perfilado_datos.html`  
✅ `outputs/datasets/Delitos_Informaticos_Limpio.csv`  
✅ `modelos/modelo_final.pkl` + artefactos asociados  
✅ Todos los `.png` generados en `outputs/imagenes/`  
✅ `captura_despliegue.png` (screenshot de Gradio funcionando)  

---

## 👤 Autor

**Estudiante:** Julián Andrés Rodríguez Jiménez
**Fecha:** Mayo 2026  
**Código:** Práctica 4 — Calidad y Minería de Datos
