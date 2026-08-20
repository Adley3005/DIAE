import pandas as pd

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
