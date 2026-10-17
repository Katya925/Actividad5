# CHANGELOG

## v1.1 - Preparación reproducible
- Fuente: sklearn.datasets.load_breast_cancer.
- Eliminación de filas duplicadas.
- Negativos en mediciones: NaN; ceros legítimos conservados.
- Imputación y escalado solo dentro del pipeline entrenado.
- RAW y CLEANED exportados con hash SHA256.
