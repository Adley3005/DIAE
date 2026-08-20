import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# 1. Carga de la serie temporal histórica oficial (Casos Confirmados Globales)
url_time_series = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
ts_raw = pd.read_csv(url_time_series)

# 2. Filtrar y estructurar la serie temporal para un país (ejemplo: Peru)
target_country = 'Peru'
country_ts = ts_raw[ts_raw['Country/Region'] == target_country].iloc[:, 4:].sum(axis=0)
country_ts.index = pd.to_datetime(country_ts.index, format='%m/%d/%y')

# Casos diarios nuevos y suavizado de 7 días (Media Móvil)
daily_cases = country_ts.diff().dropna()
daily_cases[daily_cases < 0] = 0  # Ajuste de anomalías
daily_smoothed = daily_cases.rolling(window=7).mean().dropna()

# Asignar frecuencia diaria explícita primero
daily_smoothed = daily_smoothed.asfreq('D')

# 3. División Entrenamiento / Prueba (heredan freq='D')
forecast_steps = 14
train = daily_smoothed.iloc[:-forecast_steps]
test = daily_smoothed.iloc[-forecast_steps:]

# 4. Ajuste del Modelo SARIMA(1, 1, 1)x(1, 0, 1, 7)
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
fitted_model = model.fit(disp=False)

# 5. Predicción y Bandas de Confianza (95%)
forecast_res = fitted_model.get_forecast(steps=forecast_steps)
predictions = forecast_res.predicted_mean
ci_forecast = forecast_res.conf_int(alpha=0.05)

# 6. Evaluación del Desempeño
mae = mean_absolute_error(test, predictions)
mape = mean_absolute_percentage_error(test, predictions) * 100

print(f"--- Desempeño SARIMA ({target_country}) ---")
print(f"MAE  : {mae:.2f} casos")
print(f"MAPE : {mape:.2f}%")

# 7. Gráfico de Backtesting con Bandas de Confianza
plt.figure(figsize=(12, 6))
plt.plot(train.index[-60:], train.iloc[-60:], label='Entrenamiento (Suavizado 7d)', color='steelblue')
plt.plot(test.index, test, label='Valores Reales (Test)', color='black', marker='o')
plt.plot(test.index, predictions, label='Pronóstico SARIMA', color='crimson', linestyle='--')
plt.fill_between(test.index, ci_forecast.iloc[:, 0], ci_forecast.iloc[:, 1], color='pink', alpha=0.4, label='Banda de Confianza 95%')

plt.title(f'Proyección COVID-19 a 14 Días - {target_country} (SARIMA)')
plt.xlabel('Fecha')
plt.ylabel('Casos Diarios')
plt.legend()
plt.tight_layout()
plt.show()

