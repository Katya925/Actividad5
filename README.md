# Actividad 5 — Clasificación de tumores mamarios

Proyecto de clasificación binaria con **Regresión Logística** y **Random Forest**, utilizando el conjunto de datos **Breast Cancer Wisconsin (Diagnostic)** de Scikit-Learn. Incluye preparación de datos, búsqueda de hiperparámetros, evaluación comparativa y seguimiento de experimentos con MLflow.

## 1. Datos

- **Fuente:** `sklearn.datasets.load_breast_cancer()`; origen: UCI Machine Learning Repository.
- **Tamaño:** 569 observaciones y 30 variables predictoras numéricas.
- **Clases originales:** `0 = maligno`, `1 = benigno`.
- **Codificación utilizada en el proyecto:** `0 = benigno`, `1 = maligno`, mediante `y = 1 - target`.
- **Distribución:** 357 casos benignos y 212 malignos.
- **Diccionario:** diez características celulares (`radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, `concave points`, `symmetry` y `fractal dimension`), cada una expresada como `mean`, `error` y `worst`.

**Supuestos:** las observaciones y sus etiquetas se consideran válidas para el ejercicio; los ceros en medidas de concavidad pueden ser legítimos y no se eliminan automáticamente. Los valores atípicos se revisan, pero no se interpretan por sí solos como errores.

## 2. Tecnologías

- Python en Google Colab (también puede ejecutarse en Jupyter con las dependencias instaladas).
- NumPy, Pandas, Scikit-Learn, SciPy, Matplotlib y Seaborn.
- **MLflow 3.17.1** para registrar parámetros, métricas, modelos y artefactos.
- **SQLite** para el almacenamiento de seguimiento en la versión ajustada para Colab.

## 3. Reproducción del proyecto

1. Abrir `Actividad5.ipynb` en Google Colab.
2. Instalar las dependencias; para MLflow ejecutar:

   ```python
   %pip install -q "mlflow==3.17.1"
   ```

3. En la celda de configuración de MLflow, utilizar la **misma base de datos SQLite** tanto para registrar como para consultar los experimentos:

   ```python
   import mlflow
   from pathlib import Path

   db = (Path.cwd() / "mlflow.db").resolve()
   mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
   mlflow.set_experiment("Actividad5_Breast_Cancer_Clasificacion")
   ```

4. Ejecutar las celdas en orden: carga y validación de datos; partición; definición de pipelines; entrenamiento y ajuste de ambos modelos; comparación y curvas ROC; y generación de evidencias para consultar las ejecuciones reales de MLflow y exportar tablas.
5. Guardar el notebook ejecutado, el archivo `mlflow.db`, los artefactos de MLflow y los archivos de evidencia. En Colab, estos archivos pueden perderse al terminar la sesión; descargarlos o respaldarlos en Drive.

## 4. Flujo de preparación y entrenamiento

1. **Calidad de datos:** revisar tipos numéricos, valores faltantes, duplicados y mediciones inválidas; conservar ceros legítimos.
2. **Separación:** 80 % entrenamiento y 20 % prueba, con `stratify=y` y semilla `42`.
3. **Preprocesamiento:** imputación por mediana y estandarización dentro de un `Pipeline`, ajustado solo con datos de entrenamiento para evitar fuga de información.
4. **Optimización:** `GridSearchCV` con validación cruzada estratificada de cinco particiones y F1-score como criterio principal.
5. **Evaluación:** accuracy, precision, recall, F1-score, ROC-AUC, matrices de confusión y curvas ROC.
6. **Comparación:** análisis de diferencias entre modelos y pruebas estadísticas exploratorias sobre puntuaciones de validación cruzada, interpretadas con cautela.

## 5. Decisiones técnicas

| Decisión | Justificación |
|---|---|
| Regresión Logística y Random Forest | Comparar un modelo lineal interpretable con uno capaz de capturar relaciones no lineales. |
| Clase positiva = maligno | Facilitar la interpretación de recall y falsos negativos en casos malignos. |
| División estratificada 80/20 | Conservar proporciones de clase y reservar datos para evaluación final. |
| Imputación y escalado dentro del pipeline | Evitar fuga de información y reproducir transformaciones en validación y prueba. |
| GridSearchCV, 5 folds, F1 | Buscar sistemáticamente hiperparámetros equilibrando precision y recall. |
| Semilla 42 | Reducir variación debida a procesos aleatorios. |
| MLflow con SQLite | Registrar y consultar experimentos sin depender del panel web en Colab. |

### Hiperparámetros explorados

| Modelo | Parámetro | Valores |
|---|---|---|
| Regresión Logística | `C` | `0.01`, `0.1`, `1`, `10` |
| Regresión Logística | `penalty` / `solver` | `l2` / `lbfgs` |
| Random Forest | `n_estimators` | `50`, `100`, `200` |
| Random Forest | `max_depth` | `None`, `5`, `10` |
| Random Forest | `min_samples_split` | `2`, `5` |

Las configuraciones óptimas deben consultarse en las salidas de `GridSearchCV` o en los parámetros registrados en MLflow de la ejecución definitiva.

## 6. Evidencias de MLflow

La interfaz web de MLflow **no es indispensable** para verificar los registros: se puede usar la API desde Colab.

```python
import mlflow

# Ejecutar tras configurar el mismo tracking URI que en el entrenamiento.
runs = mlflow.search_runs(search_all_experiments=True)
print("Ejecuciones encontradas:", len(runs))
if not runs.empty:
    cols = [c for c in ["run_id", "tags.mlflow.runName", "status", "metrics.f1", "metrics.roc_auc"] if c in runs.columns]
    display(runs[cols])
```

Para el informe, conservar evidencia de:

1. **Ejecuciones:** identificador real (`run_id`), nombre del modelo y estado `FINISHED`.
2. **Métricas:** accuracy, precision, recall, F1 y ROC-AUC.
3. **Parámetros:** configuraciones seleccionadas por GridSearchCV.
4. **Artefactos y modelos:** curvas ROC, matrices de confusión, huella del dataset y URI del modelo; comprobar, cuando sea posible, que el modelo se puede cargar.
5. **Trazabilidad:** fecha, versión del dataset, semilla y versiones de dependencias.

No usar identificadores provisionales como `local_run`: seleccionar los `run_id` reales mediante `mlflow.search_runs()`.

## 7. Archivos y versionamiento

| Archivo o recurso | Uso |
|---|---|
| `Actividad5.ipynb` | Código, análisis y resultados. |
| `dataset_raw.csv` | Copia de datos antes de la limpieza. |
| `dataset_cleaned.csv` | Datos tras las reglas iniciales de limpieza; no necesariamente escalados. |
| `CHANGELOG.md` | Historial de cambios, si fue generado y actualizado. |
| `dependencias.json` | Versiones de dependencias, si fue generado. |
| `mlflow.db` | Base de datos de seguimiento en SQLite. |
| Artefactos de MLflow | Gráficas, modelos y archivos asociados a los experimentos. |
| `Evidencias_MLflow_Actividad5.zip` | Exportación opcional de evidencias desde la celda 9. |

**Nota:** el archivo SQLite por sí solo puede no contener todos los archivos de modelos y artefactos. Conservar también el almacenamiento de artefactos asociado. Antes de publicar en GitHub, evitar subir archivos voluminosos, datos sensibles o resultados temporales sin revisarlos.

## 8. Limitaciones y mejoras futuras

- El dataset es académico; no confirma que el modelo esté listo para uso clínico.
- La evaluación depende de la partición y del tamaño de la muestra; las pruebas sobre cinco folds tienen capacidad estadística limitada.
- Se recomienda automatizar verificaciones de datos, fijar todas las versiones de paquetes, conservar hashes y artefactos, y monitorear el desempeño si se utilizaran nuevos datos.
- Las métricas altas no eliminan el riesgo de falsos negativos.

## 9. Referencias

Scikit-learn developers. (s. f.). *sklearn.datasets.load_breast_cancer*. Scikit-learn. https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

MLflow. (s. f.). *MLflow tracking*. https://mlflow.org/docs/latest/ml/tracking/

Scikit-learn developers. (s. f.). *GridSearchCV*. Scikit-learn. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
