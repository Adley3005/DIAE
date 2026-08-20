import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

# ==========================================
# PARTE 1: EXPLORACIÓN Y VISUALIZACIÓN
# ==========================================

# 1. Carga de datos
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-18-2022.csv"
df = pd.read_csv(url)

print("--- Primeras 10 filas ---")
print(df.head(10))

print("\n--- Estructura del Dataset ---")
df.info()

print("\n--- Valores Faltantes ---")
print(df.isnull().sum())

# Agrupaciones
country_totals = df.groupby('Country_Region')[['Confirmed', 'Deaths', 'Recovered', 'Active']].sum()
country_province_totals = df.groupby(['Country_Region', 'Province_State'])[['Confirmed', 'Deaths', 'Recovered', 'Active']].sum()

# Provincias de China
china_sorted = df[df['Country_Region'] == 'China'].groupby('Province_State')[['Confirmed', 'Deaths']].sum().sort_values(by='Confirmed', ascending=False)
print("\n--- Provincias de China ordenadas por Confirmados ---")
print(china_sorted.head())

# Top 10 Países
top10_deaths = country_totals.sort_values(by='Deaths', ascending=False).head(10)
print(f"\nTop 10 Fallecidos - Máximo: {top10_deaths.index[0]} ({top10_deaths['Deaths'].iloc[0]})")
print(f"Top 10 Fallecidos - Mínimo: {top10_deaths.index[-1]} ({top10_deaths['Deaths'].iloc[-1]})")

# Muestra de 50 filas a Excel
sample_50 = df.sample(n=50, random_state=42)
cols_to_drop = [sample_50.columns[i] for i in [0, 1, 5, 6, 11]]
sample_cleaned = sample_50.drop(columns=cols_to_drop)
sample_cleaned.to_excel("muestra_50.xlsx", index=False)
print("\nArchivo Excel 'muestra_50.xlsx' guardado correctamente.")

# Configuración visual
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.autolayout': True})

# Gráfica 1: Líneas (Países con Deaths > 2500)
df_line = country_totals[country_totals['Deaths'] > 2500].sort_values(by='Confirmed', ascending=False)
plt.figure(figsize=(14, 5))
plt.plot(df_line.index, df_line['Confirmed'], label='Confirmados', marker='o', linewidth=1.5)
plt.plot(df_line.index, df_line['Deaths'], label='Fallecidos', marker='s', color='crimson')
plt.xticks(rotation=90, fontsize=8)
plt.yscale('log')
plt.title('Incidencia COVID-19 por País (Fallecidos > 2,500) - Escala Log')
plt.xlabel('País')
plt.ylabel('Cantidad')
plt.legend()
plt.show()

# Gráfica 2: Barplot Top 20 Estados EE. UU. (Ajustado sin warning)
us_states = df[df['Country_Region'] == 'US'].groupby('Province_State')['Deaths'].sum().sort_values(ascending=False).head(20)
plt.figure(figsize=(12, 5))
sns.barplot(x=us_states.index, y=us_states.values, hue=us_states.index, palette='Reds_r', legend=False)
plt.xticks(rotation=75)
plt.title('Top 20 Estados de EE. UU. con Mayor Número de Fallecidos')
plt.xlabel('Estado')
plt.ylabel('Total Fallecidos')
plt.show()

# Gráfica 3: Pie chart Países LATAM
latam_countries = ['Colombia', 'Chile', 'Peru', 'Argentina', 'Mexico']
latam_deaths = country_totals.loc[country_totals.index.intersection(latam_countries), 'Deaths']
plt.figure(figsize=(6, 6))
plt.pie(latam_deaths, labels=latam_deaths.index, autopct='%1.1f%%', startangle=140, 
        colors=sns.color_palette('Set2', len(latam_deaths)), explode=[0.05]*len(latam_deaths))
plt.title('Distribución de Fallecidos en Países Seleccionados de LATAM')
plt.show()

# Gráfica 4: Histograma de Fallecidos
plt.figure(figsize=(9, 4))
sns.histplot(country_totals['Deaths'], bins=30, kde=True, color='darkred')
plt.title('Distribución del Total de Fallecidos por País')
plt.xlabel('Fallecidos')
plt.ylabel('Número de Países')
plt.show()

# Gráfica 5: Boxplots (Ajustado solo para variables con datos > 0)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col in zip(axes, ['Confirmed', 'Deaths']):
    sns.boxplot(y=country_totals[col][country_totals[col] > 0], ax=ax, color='skyblue')
    ax.set_title(f'Distribución de {col}')
    ax.set_yscale('log')
plt.suptitle('Boxplots de Confirmados y Fallecidos (Escala Logarítmica)')
plt.show()

# ==========================================
# PARTE 2: ESTADÍSTICA DESCRIPTIVA Y AVANZADA
# ==========================================

# CFR e Intervalos de Confianza (Wilson 95%)
country_data = country_totals[country_totals['Confirmed'] >= 100].copy()
country_data['CFR'] = (country_data['Deaths'] / country_data['Confirmed']) * 100

ci_low, ci_upp = proportion_confint(country_data['Deaths'], country_data['Confirmed'], alpha=0.05, method='wilson')
country_data['CI_Lower_95'] = ci_low * 100
country_data['CI_Upper_95'] = ci_upp * 100

print("\n--- Muestra de Intervalos de Confianza 95% (CFR) ---")
print(country_data[['Confirmed', 'Deaths', 'CFR', 'CI_Lower_95', 'CI_Upper_95']].head(5))

# Prueba de Hipótesis: Perú vs México
count = np.array([country_data.loc['Peru', 'Deaths'], country_data.loc['Mexico', 'Deaths']])
nobs = np.array([country_data.loc['Peru', 'Confirmed'], country_data.loc['Mexico', 'Confirmed']])
z_stat, p_val = proportions_ztest(count, nobs, alternative='two-sided')

print(f"\n--- Test Z de Proporciones (Peru vs Mexico) ---")
print(f"Z-stat: {z_stat:.4f} | P-valor: {p_val:.4e}")

# Outliers
Q1, Q3 = country_data['Deaths'].quantile(0.25), country_data['Deaths'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = country_data[country_data['Deaths'] > (Q3 + 1.5 * IQR)]
country_data['Z_Score_Deaths'] = stats.zscore(country_data['Deaths'])
outliers_z = country_data[np.abs(country_data['Z_Score_Deaths']) > 3]

print(f"Outliers IQR: {len(outliers_iqr)} países | Outliers Z-score (>3σ): {len(outliers_z)} países")

# Carta de Control 3-Sigma
mean_d, std_d = country_data['Deaths'].mean(), country_data['Deaths'].std()
ucl, lcl = mean_d + 3 * std_d, max(0, mean_d - 3 * std_d)

plt.figure(figsize=(14, 5))
plt.plot(country_data.index, country_data['Deaths'], marker='o', linestyle='-', color='steelblue', label='Fallecidos')
plt.axhline(mean_d, color='green', linestyle='--', label=f'Media: {mean_d:.0f}')
plt.axhline(ucl, color='red', linestyle='--', label=f'LSC (3σ): {ucl:.0f}')
plt.axhline(lcl, color='orange', linestyle='--', label=f'LIC (3σ): {lcl:.0f}')
plt.xticks(rotation=90, fontsize=6)
plt.title('Carta de Control (3-Sigma) de Fallecidos por País')
plt.xlabel('País')
plt.ylabel('Fallecidos')
plt.legend()
plt.show()