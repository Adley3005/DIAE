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
top20_states = us_states.head(20)
sns.barplot(x=top20_states.index, y=top20_states.values, hue=top20_states.index, palette='Reds_r', legend=False)
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
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
vars_to_plot = ['Confirmed', 'Deaths']

for ax, col in zip(axes, vars_to_plot):
    sns.boxplot(y=country_totals[col][country_totals[col] > 0], ax=ax, color='skyblue')
    ax.set_title(f'Distribución de {col}')
    ax.set_yscale('log')

plt.suptitle('Boxplots de Confirmados y Fallecidos (Escala Logarítmica)', fontsize=13)
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# 1. Agrupación y Cálculo de CFR por País
country_data = df.groupby('Country_Region')[['Confirmed', 'Deaths']].sum()
# Filtrar países con al menos 100 casos confirmados para evitar ruido estadístico
country_data = country_data[country_data['Confirmed'] >= 100].copy()
country_data['CFR'] = (country_data['Deaths'] / country_data['Confirmed']) * 100

# 2. Intervalo de Confianza al 95 % para el CFR (Método Wilson)
ci_low, ci_upp = proportion_confint(
    count=country_data['Deaths'],
    nobs=country_data['Confirmed'],
    alpha=0.05,
    method='wilson'
)
country_data['CI_Lower_95'] = ci_low * 100
country_data['CI_Upper_95'] = ci_upp * 100

print("--- Muestra de Países con Intervalos de Confianza (95%) ---")
print(country_data[['Confirmed', 'Deaths', 'CFR', 'CI_Lower_95', 'CI_Upper_95']].head(10))

# 3. Test de Hipótesis de 2 Proporciones (Ejemplo: Perú vs México)
p1_name, p2_name = 'Peru', 'Mexico'
count = np.array([country_data.loc[p1_name, 'Deaths'], country_data.loc[p2_name, 'Deaths']])
nobs = np.array([country_data.loc[p1_name, 'Confirmed'], country_data.loc[p2_name, 'Confirmed']])

z_stat, p_val = proportions_ztest(count, nobs, alternative='two-sided')

print(f"\n--- Prueba de Hipótesis: CFR {p1_name} vs {p2_name} ---")
print(f"CFR {p1_name}: {country_data.loc[p1_name, 'CFR']:.2f}% | CFR {p2_name}: {country_data.loc[p2_name, 'CFR']:.2f}%")
print(f"Estadístico Z: {z_stat:.4f}")
print(f"P-valor: {p_val:.4e}")
if p_val < 0.05:
    print("Conclusión: Se rechaza H0. Existe diferencia estadísticamente significativa entre las tasas de letalidad.")
else:
    print("Conclusión: No se rechaza H0. No hay evidencia estadística suficiente de diferencia.")

# 4. Detección de Outliers (IQR y Z-score sobre Deaths)
# Método IQR
Q1 = country_data['Deaths'].quantile(0.25)
Q3 = country_data['Deaths'].quantile(0.75)
IQR = Q3 - Q1
upper_iqr = Q3 + 1.5 * IQR
outliers_iqr = country_data[country_data['Deaths'] > upper_iqr]

# Método Z-score
country_data['Z_Score_Deaths'] = stats.zscore(country_data['Deaths'])
outliers_z = country_data[np.abs(country_data['Z_Score_Deaths']) > 3]

print(f"\n--- Detección de Outliers en Fallecidos ---")
print(f"Outliers detectados por IQR: {len(outliers_iqr)} países")
print(f"Outliers detectados por Z-Score (>3σ): {len(outliers_z)} países")
print("Top outliers (Z-score):", list(outliers_z.index[:5]))

# 5. Gráfico de Control Shewhart (3 Sigma) para Fallecidos por País
mean_deaths = country_data['Deaths'].mean()
std_deaths = country_data['Deaths'].std()
ucl = mean_deaths + 3 * std_deaths
lcl = max(0, mean_deaths - 3 * std_deaths)

plt.figure(figsize=(14, 6))
plt.plot(country_data.index, country_data['Deaths'], marker='o', linestyle='-', color='steelblue', label='Fallecidos')
plt.axhline(mean_deaths, color='green', linestyle='--', label=f'Media: {mean_deaths:.0f}')
plt.axhline(ucl, color='red', linestyle='--', label=f'LSC (UCL 3σ): {ucl:.0f}')
plt.axhline(lcl, color='orange', linestyle='--', label=f'LIC (LCL 3σ): {lcl:.0f}')

plt.xticks(rotation=90, fontsize=6)
plt.title('Carta de Control (3-Sigma) de Fallecidos Totales por País')
plt.xlabel('País')
plt.ylabel('Fallecidos')
plt.legend()
plt.tight_layout()
plt.show()













