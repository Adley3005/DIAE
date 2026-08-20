import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Carga de datos desde la URL oficial
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-18-2022.csv"
df = pd.read_csv(url)

# 2. Exploración básica
print("--- Primeras 10 filas ---")
print(df.head(10))

print("\n--- Estructura y Tipos de Datos ---")
df.info()

print("\n--- Valores Faltantes por Columna ---")
print(df.isnull().sum())

# 3. Totales por País
country_totals = df.groupby('Country_Region')[['Confirmed', 'Deaths', 'Recovered', 'Active']].sum()

# 4. Totales por País y Provincia
country_province_totals = df.groupby(['Country_Region', 'Province_State'])[['Confirmed', 'Deaths', 'Recovered', 'Active']].sum()

# 5. Filtrar y ordenar provincias de China por casos confirmados
china_provinces = df[df['Country_Region'] == 'China'].groupby('Province_State')[['Confirmed', 'Deaths']].sum()
china_sorted = china_provinces.sort_values(by='Confirmed', ascending=False)
print("\n--- Provincias de China ordenadas por Confirmados ---")
print(china_sorted.head())

# 6. Top 10 Países Confirmados y Fallecidos
top10_confirmed = country_totals.sort_values(by='Confirmed', ascending=False).head(10)
top10_deaths = country_totals.sort_values(by='Deaths', ascending=False).head(10)

highest_death_country = top10_deaths.index[0]
lowest_death_in_top10 = top10_deaths.index[-1]

print(f"\nTop 10 Fallecidos - Máximo: {highest_death_country} ({top10_deaths['Deaths'].iloc[0]})")
print(f"Top 10 Fallecidos - Mínimo (puesto 10): {lowest_death_in_top10} ({top10_deaths['Deaths'].iloc[-1]})")

# 7. Muestreo, eliminación de columnas y exportación a Excel
sample_50 = df.sample(n=50, random_state=42)
# Eliminación de columnas por índice posicional: 0, 1, 5, 6, 11
cols_to_drop = [sample_50.columns[i] for i in [0, 1, 5, 6, 11]]
sample_cleaned = sample_50.drop(columns=cols_to_drop)

sample_cleaned.to_excel("muestra_50.xlsx", index=False)

# Validación de lectura
df_excel_check = pd.read_excel("muestra_50.xlsx")
print(f"\nArchivo Excel validado correctamente. Filas: {len(df_excel_check)}, Columnas: {len(df_excel_check.columns)}")




# 1. Configurar visualización completa temporal en pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Visualización rápida de prueba (primeras 5 filas con todas las columnas visibles)
print(df.head())

# Restaurar configuración predeterminada
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

# Configuración visual global
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.autolayout': True})

# 2. Gráfico de líneas (Países con Deaths > 2500)
df_line = country_totals[country_totals['Deaths'] > 2500].sort_values(by='Confirmed', ascending=False)

plt.figure(figsize=(14, 6))
plt.plot(df_line.index, df_line['Confirmed'], label='Confirmados', marker='o', linewidth=1.5)
plt.plot(df_line.index, df_line['Deaths'], label='Fallecidos', marker='s', color='crimson')
plt.plot(df_line.index, df_line['Recovered'], label='Recuperados', marker='^', color='green')
plt.plot(df_line.index, df_line['Active'], label='Activos', marker='d', color='orange')
plt.xticks(rotation=90, fontsize=8)
plt.yscale('log')  # Escala logarítmica para comparar magnitudes dispares
plt.title('Incidencia COVID-19 por País (Fallecidos > 2,500) - Escala Logarítmica')
plt.xlabel('País')
plt.ylabel('Cantidad (Escala Log)')
plt.legend()
plt.show()

# 3. Gráfico de barras: Fallecidos por estado en EE. UU. (Top 20 para legibilidad)
us_states = df[df['Country_Region'] == 'US'].groupby('Province_State')['Deaths'].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=us_states.head(20).index, y=us_states.head(20).values, palette='Reds_r')
plt.xticks(rotation=75)
plt.title('Top 20 Estados de EE. UU. con Mayor Número de Fallecidos')
plt.xlabel('Estado')
plt.ylabel('Total Fallecidos')
plt.show()

# 4. Gráfico de sectores (Pie chart): Colombia, Chile, Perú, Argentina y México
latam_countries = ['Colombia', 'Chile', 'Peru', 'Argentina', 'Mexico']
latam_deaths = country_totals.loc[country_totals.index.intersection(latam_countries), 'Deaths']

plt.figure(figsize=(7, 7))
plt.pie(latam_deaths, labels=latam_deaths.index, autopct='%1.1f%%', startangle=140, 
        colors=sns.color_palette('Set2', len(latam_deaths)), explode=[0.05]*len(latam_deaths))
plt.title('Distribución de Fallecidos en Países Seleccionados de LATAM')
plt.show()

# 5. Histograma de fallecidos por país
plt.figure(figsize=(10, 5))
sns.histplot(country_totals['Deaths'], bins=30, kde=True, color='darkred')
plt.title('Distribución del Total de Fallecidos por País')
plt.xlabel('Fallecidos')
plt.ylabel('Frecuencia (Número de países)')
plt.show()

# 6. Boxplots de variables principales
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
vars_to_plot = ['Confirmed', 'Deaths', 'Recovered', 'Active']

for ax, col in zip(axes.flatten(), vars_to_plot):
    sns.boxplot(y=country_totals[col], ax=ax, color='skyblue')
    ax.set_title(f'Distribución de {col}')
    ax.set_yscale('log')  # Ayuda a observar la dispersión y outliers sin aplastar la escala

plt.suptitle('Boxplots de Métricas Globales (Escala Logarítmica)', fontsize=14)
plt.show()