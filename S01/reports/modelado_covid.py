import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# 1. Carga y preparación de serie temporal
url_time_series = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
ts_raw = pd.read_csv(url_time_series)

target_country = 'Peru'
country_ts = ts_raw[ts_raw['Country/Region'] == target_country].iloc[:, 4:].sum(axis=0)
country_ts.index = pd.to_datetime(country_ts.index, format='%m/%d/%y')

# Casos diarios y suavizado 7 días
daily_cases = country_ts.diff().dropna()
daily_cases[daily_cases < 0] = 0
daily_smoothed = daily_cases.rolling(window=7).mean().dropna().asfreq('D')

# 2. Acotar ventana reciente (últimos 90 días) para evitar shocks estructurales pasados
recent_series = daily_smoothed.iloc[-90:]

# 3. Transformación Logarítmica: log(y + 1)
log_series = np.log1p(recent_series)

# 4. División Train / Test (14 días)
forecast_steps = 14
train_log = log_series.iloc[:-forecast_steps]
test_actual = recent_series.iloc[-forecast_steps:]

# 5. Ajuste SARIMA sobre escala logarítmica
model = SARIMAX(train_log, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7), 
                enforce_stationarity=False, enforce_invertibility=False)
fitted_model = model.fit(disp=False)

# 6. Predicción e Intervalos de Confianza
forecast_res = fitted_model.get_forecast(steps=forecast_steps)
pred_log = forecast_res.predicted_mean
ci_log = forecast_res.conf_int(alpha=0.05)

# Revertir transformación: exp(y) - 1
predictions = np.expm1(pred_log)
ci_lower = np.maximum(0, np.expm1(ci_log.iloc[:, 0]))  # Límite inferior no negativo
ci_upper = np.expm1(ci_log.iloc[:, 1])

# 7. Métricas de Evaluación
mae = mean_absolute_error(test_actual, predictions)
mape = mean_absolute_percentage_error(test_actual, predictions) * 100

print(f"--- Desempeño SARIMA Optimizado ({target_country}) ---")
print(f"MAE  : {mae:.2f} casos")
print(f"MAPE : {mape:.2f}%")

# 8. Gráfico con bandas ajustadas y positivas
plt.figure(figsize=(12, 6))
plt.plot(recent_series.index[:-forecast_steps], recent_series.iloc[:-forecast_steps], 
         label='Entrenamiento Suavizado (7d)', color='steelblue')
plt.plot(test_actual.index, test_actual, label='Valores Reales (Test)', color='black', marker='o')
plt.plot(test_actual.index, predictions, label='Pronóstico SARIMA Optimizado', color='crimson', linestyle='--')
plt.fill_between(test_actual.index, ci_lower, ci_upper, color='pink', alpha=0.4, label='Banda de Confianza 95% (Ajustada)')

plt.ylim(bottom=0)  # Forzar eje Y en 0
plt.title(f'Proyección COVID-19 Optimizada (14 Días) - {target_country}')
plt.xlabel('Fecha')
plt.ylabel('Casos Diarios')
plt.legend()
plt.tight_layout()
plt.show()