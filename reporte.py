import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Crear carpeta de gráficos si no existe
# No hay variables de usuario que cambiar aquí, 'os' es un módulo.
os.makedirs("Graficos", exist_ok=True)
print("[INFO] Carpeta 'Graficos' creada o ya existente.")

# 1. Cargar archivos Excel con datos de cada año
try:
    # Cambiamos df_ AÑO por datos_ AÑO
    datos_2019 = pd.read_excel("data/totales-finales-de-ingresos-ano-2019-municipal.xls", skiprows=4, decimal=',')
    datos_2020 = pd.read_excel("data/totales-finales-de-ingresos-ano-2020-municipal.xls", skiprows=4, decimal=',')
    datos_2021 = pd.read_excel("data/totales-finales-de-ingresos-ano-2021municipal.xls", skiprows=4, decimal=',')
    print("[INFO] Archivos Excel cargados correctamente.")
except FileNotFoundError as e:
    print(f"[ERROR] Archivo no encontrado: {e}")
    exit(1)

# 2. Limpieza de datos
# Cambiamos nombre de función y parámetros
def procesar_archivo_anual(datos_entrada, anio_actual):
    print(f"[INFO] Procesando archivo del año {anio_actual}...")
    # Usamos una variable interna con nombre en español
    datos_procesados = datos_entrada.dropna(how='all')  # Eliminar filas totalmente vacías
    # Los nombres de las columnas ya estaban mayormente en español, los mantenemos
    datos_procesados.columns = ["mes", "etiqueta", "presupuesto_inicial", "presupuesto_vigente",
                                "recaudado", "recaudado_anterior", "porcentaje", "recaudado_real",
                                "recaudado_real_anterior", "porcentaje_real", "porcentaje_cambio"]
    # Filtramos usando la columna 'etiqueta'
    datos_procesados = datos_procesados[datos_procesados["etiqueta"] == "TOTALES FINALES"]
    # Añadimos columna 'Anio' (evitamos ñ para compatibilidad)
    datos_procesados['Anio'] = anio_actual
    print(f"[INFO] Archivo {anio_actual} procesado con {len(datos_procesados)} filas.")
    return datos_procesados

# Llamamos a la función con los nuevos nombres y guardamos en variables renombradas
datos_procesados_2019 = procesar_archivo_anual(datos_2019, 2019)
datos_procesados_2020 = procesar_archivo_anual(datos_2020, 2020)
datos_procesados_2021 = procesar_archivo_anual(datos_2021, 2021)

# Unir todos los años
# Cambiamos df_total a datos_totales
datos_totales = pd.concat([datos_procesados_2019, datos_procesados_2020, datos_procesados_2021])
print(f"[INFO] Dataset total unificado con forma: {datos_totales.shape}")

# 3. Análisis por mes
print("[INFO] Analizando ingresos por mes...")
# Asegurarse de que 'recaudado_real' sea numérico antes de la agregación
# Usamos la variable renombrada datos_totales
datos_totales['recaudado_real'] = pd.to_numeric(datos_totales['recaudado_real'], errors='coerce')

# Renombramos variables de análisis
ingresos_mensuales_acumulados = datos_totales.groupby("mes")["recaudado_real"].sum().sort_values(ascending=False)
print("\n[INFO] Meses con mayores ingresos acumulados:\n", ingresos_mensuales_acumulados)

# Usamos 'Anio' como columna y renombramos la variable resultado
variabilidad_por_anio = datos_totales.groupby("Anio")["recaudado_real"].std()
print("\n[INFO] Variabilidad mensual (desviación estándar) por año:\n", variabilidad_por_anio)

# 4. Visualización
print("[INFO] Generando visualizaciones...")

# a. Gráfico de líneas
plt.figure(figsize=(12,6))
# Usamos datos_totales y la columna 'Anio'
sns.lineplot(data=datos_totales, x="mes", y="recaudado_real", hue="Anio")
plt.title("Ingresos Recaudados Reales Mensuales por Año")
plt.xlabel("Mes") # Añadida etiqueta X
plt.ylabel("Ingresos Recaudados Reales") # Añadida etiqueta Y
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Graficos/lineas_por_año.png")
print("[INFO] Gráfico de líneas por año guardado.")

# b. Gráfico de barras agrupadas
plt.figure(figsize=(12,6))
# Usamos datos_totales y la columna 'Anio'
sns.barplot(data=datos_totales, x="mes", y="recaudado_real", hue="Anio")
plt.title("Comparación de Ingresos Reales por Mes y Año")
plt.xlabel("Mes") # Añadida etiqueta X
plt.ylabel("Ingresos Recaudados Reales") # Añadida etiqueta Y
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Graficos/barras_agrupadas.png")
print("[INFO] Gráfico de barras agrupadas guardado.")

# c. Mapa de calor
print("[INFO] Generando mapa de calor...")
# Renombramos pivot_df a tabla_pivotante
tabla_pivotante = datos_totales.pivot_table(index='mes', columns='Anio', values='recaudado_real', aggfunc='sum')
plt.figure(figsize=(10,6))
# Usamos la variable renombrada tabla_pivotante
sns.heatmap(tabla_pivotante, annot=True, fmt=".0f", cmap='YlGnBu')
plt.title("Mapa de calor de ingresos reales por mes y año")
plt.xlabel("Año") # Añadida etiqueta X
plt.ylabel("Mes") # Añadida etiqueta Y
plt.tight_layout()
plt.savefig("Graficos/mapa_calor.png")
print("[INFO] Mapa de calor guardado.")

# 5. Clustering con KMeans
print("[INFO] Iniciando clustering con KMeans...")

# Renombramos df_kmeans a datos_para_kmeans
datos_para_kmeans = datos_totales.groupby(["Anio", "mes"]).agg({"recaudado_real": "sum"}).reset_index()

# Renombramos mes_a_num a mes_a_numero
mes_a_numero = {
    'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
    'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
}

# Limpiar la columna 'mes': convertir a mayúsculas y eliminar espacios
# Renombramos las columnas creadas
datos_para_kmeans["mes_limpio"] = datos_para_kmeans["mes"].str.upper().str.strip()
datos_para_kmeans["mes_num"] = datos_para_kmeans["mes_limpio"].map(mes_a_numero)

print("Valores únicos en la columna 'mes' original:")
print(datos_para_kmeans['mes'].unique())
# Usamos la columna renombrada
print("Valores únicos en la columna 'mes_limpio' después de limpieza:")
print(datos_para_kmeans['mes_limpio'].unique())
# Usamos la columna renombrada
print("\nNúmero de valores nulos en 'mes_num':", datos_para_kmeans['mes_num'].isnull().sum())
print("\nFilas con valores nulos en 'mes_num':")
print(datos_para_kmeans[datos_para_kmeans['mes_num'].isnull()])

# Asegurarse de que 'recaudado_real' en datos_para_kmeans sea numérico para KMeans
datos_para_kmeans['recaudado_real'] = pd.to_numeric(datos_para_kmeans['recaudado_real'], errors='coerce')
# Renombramos df_kmeans_no_na a datos_kmeans_validos
datos_kmeans_validos = datos_para_kmeans.dropna(subset=['recaudado_real', 'mes_num']) # Incluir 'mes_num'

print("\ndatos_para_kmeans:")
print(datos_para_kmeans)
print("\ndatos_kmeans_validos después de eliminar NaN:")
print(datos_kmeans_validos)
print("\nDescripción estadística de 'recaudado_real' en datos_kmeans_validos:")
print(datos_kmeans_validos['recaudado_real'].describe())

if datos_kmeans_validos.empty:
    print("[ERROR] No hay datos válidos para realizar el clustering.")
else:
    # Aplicar KMeans solo si hay datos válidos
    # Renombramos kmeans a modelo_kmeans
    modelo_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # n_init=10 es el valor por defecto y evita warnings
    # Renombramos la columna 'cluster' a 'grupo'
    datos_kmeans_validos["grupo"] = modelo_kmeans.fit_predict(datos_kmeans_validos[["recaudado_real"]])
    print("\nValores únicos en la columna 'grupo':", datos_kmeans_validos['grupo'].unique())
    print("[INFO] Clustering completado.")

    # Visualizar clusters
    plt.figure(figsize=(10, 6))
    # Usamos las variables y columnas renombradas
    sns.scatterplot(data=datos_kmeans_validos, x="mes_num", y="recaudado_real", hue="grupo", palette="Set2", style="Anio")
    plt.title("Grupos (Clusters) de Ingresos Recaudados Reales")
    plt.xlabel("Mes (número)")
    plt.ylabel("Ingresos Reales (CLP)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Graficos/clusters_ingresos.png")
    print("[INFO] Gráfico de clustering guardado.")

# Este mensaje se imprime si el clustering falla por datos vacíos O si tiene éxito
# Se podría mejorar la lógica aquí para ser más específico
if datos_kmeans_validos.empty:
  print("[WARN] No se pudo realizar el clustering debido a la falta de datos numéricos válidos después de la limpieza.")

print("\n[INFO] Script finalizado.")