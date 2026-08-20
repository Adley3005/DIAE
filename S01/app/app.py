import os
os.environ['OMP_NUM_THREADS'] = '1'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Configuración general de la página
st.set_page_config(
    page_title="COVID-19 Analytics Dashboard",
    page_icon="🔬",
    layout="wide"
)

# ==============================================================================
# CARGA Y CACHÉ DE DATOS OFICIALES JHU
# ==============================================================================
@st.cache_data
def load_daily_data():
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-18-2022.csv"
    data = pd.read_csv(url)
    return data

@st.cache_data
def load_time_series_data():
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    ts = pd.read_csv(url)
    return ts

df = load_daily_data()
ts_raw = load_time_series_data()

# ==============================================================================
# PANEL LATERAL (SIDEBAR) - FILTROS GLOBALES
# ==============================================================================
st.sidebar.title("🎛️ Filtros del Dashboard")

all_countries = sorted(df['Country_Region'].dropna().unique().tolist())
default_selection = [c for c in ['Peru', 'Colombia', 'Chile', 'US'] if c in all_countries]

selected_countries = st.sidebar.multiselect(
    "Selecciona País(es)",
    options=all_countries,
    default=default_selection
)

min_cases = st.sidebar.number_input(
    "Umbral mínimo de Casos Confirmados",
    min_value=0,
    max_value=10000000,
    value=100000,
    step=50000
)

# Agrupación base por país
country_totals = df.groupby('Country_Region')[['Confirmed', 'Deaths', 'Recovered', 'Active']].sum().reset_index()
country_totals['CFR'] = (country_totals['Deaths'] / country_totals['Confirmed']) * 100

# Filtrado reactivo
filtered_df = country_totals[country_totals['Confirmed'] >= min_cases]
if selected_countries:
    filtered_df_selected = country_totals[country_totals['Country_Region'].isin(selected_countries)]
else:
    filtered_df_selected = filtered_df

# Encabezado Principal y KPIs
st.title("🔬 Dashboard Interactivo COVID-19")
st.caption("Integración de Análisis Exploratorio, Estadística, Series Temporales y Machine Learning.")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Confirmados Globales", f"{df['Confirmed'].sum():,}")
kpi2.metric("Fallecidos Globales", f"{df['Deaths'].sum():,}")
global_cfr = (df['Deaths'].sum() / df['Confirmed'].sum()) * 100
kpi3.metric("CFR Global Promedio", f"{global_cfr:.2f}%")
kpi4.metric("Países que superan umbral", f"{len(filtered_df)}")

st.markdown("---")

# ==============================================================================
# ESTRUCTURA DE PESTAÑAS (ST.TABS)
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Vista General (EDA)",
    "📈 Análisis Estadístico",
    "🔮 Modelado Temporal",
    "🎯 Clustering y PCA",
    "🛠️ Calidad de Datos"
])

# ------------------------------------------------------------------------------
# TAB 1: VISTA GENERAL (EDA)
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Análisis Exploratorio de Datos (EDA)")
    
    col_top1, col_top2 = st.columns(2)
    with col_top1:
        top10_conf = country_totals.sort_values(by='Confirmed', ascending=False).head(10)
        fig_conf = px.bar(
            top10_conf, x='Country_Region', y='Confirmed',
            color='Confirmed', color_continuous_scale='Blues',
            title="Top 10 Países con más Confirmados"
        )
        st.plotly_chart(fig_conf, use_container_width=True)
        
    with col_top2:
        top10_dead = country_totals.sort_values(by='Deaths', ascending=False).head(10)
        fig_dead = px.bar(
            top10_dead, x='Country_Region', y='Deaths',
            color='Deaths', color_continuous_scale='Reds',
            title="Top 10 Países con más Fallecidos"
        )
        st.plotly_chart(fig_dead, use_container_width=True)
        st.info(f"En este top 10, **{top10_dead.iloc[0]['Country_Region']}** tiene la cifra más alta y **{top10_dead.iloc[-1]['Country_Region']}** la más baja.")

    st.markdown("#### Totales por país (Fallecidos > 2500)")
    df_over_2500 = country_totals[country_totals['Deaths'] > 2500].sort_values(by='Confirmed', ascending=False)
    fig_line = px.line(
        df_over_2500, x='Country_Region', y=['Confirmed', 'Deaths', 'Recovered', 'Active'],
        markers=True, title="Incidencia Multivariable en Países con > 2,500 Fallecidos"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### Fallecidos por Estado (US)")
        us_states = df[df['Country_Region'] == 'US'].groupby('Province_State')['Deaths'].sum().sort_values(ascending=False).head(15).reset_index()
        fig_us = px.bar(us_states, x='Province_State', y='Deaths', title="Top 15 Estados de EE. UU. por Fallecidos")
        st.plotly_chart(fig_us, use_container_width=True)
        
    with col_g2:
        st.markdown("#### Fallecidos en Países Seleccionados (LatAm)")
        latam_list = ['Colombia', 'Chile', 'Peru', 'Argentina', 'Mexico']
        latam_df = country_totals[country_totals['Country_Region'].isin(latam_list)]
        fig_pie = px.pie(latam_df, names='Country_Region', values='Deaths', hole=0.4, title="Distribución Relativa en LatAm")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### Mapa Global de Casos Confirmados")
    geo_data = df.dropna(subset=['Lat', 'Long_'])
    fig_geo = px.scatter_geo(
        geo_data, lat='Lat', lon='Long_', color='Deaths', size='Confirmed',
        hover_name='Combined_Key', size_max=35, projection='natural earth',
        title="Dispersión Geoespacial Global"
    )
    st.plotly_chart(fig_geo, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: ESTADÍSTICA DESCRIPTIVA Y AVANZADA
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Estadística Descriptiva y Avanzada")
    
    st.markdown("##### Estimación de Tasa de Letalidad (CFR) e Intervalos de Confianza (Wilson 95%)")
    ci_low, ci_upp = proportion_confint(
        count=filtered_df['Deaths'],
        nobs=filtered_df['Confirmed'],
        alpha=0.05,
        method='wilson'
    )
    stat_table = filtered_df[['Country_Region', 'Confirmed', 'Deaths', 'CFR']].copy()
    stat_table['IC_Inferior_95%'] = ci_low * 100
    stat_table['IC_Superior_95%'] = ci_upp * 100
    st.dataframe(stat_table.sort_values(by='CFR', ascending=False), use_container_width=True)

    st.markdown("##### Prueba de Hipótesis (Diferencia de Proporciones de Letalidad)")
    col_h1, col_h2, col_h3 = st.columns([2, 2, 2])
    with col_h1:
        p_a = st.selectbox("País A", all_countries, index=all_countries.index("Peru") if "Peru" in all_countries else 0)
    with col_h2:
        p_b = st.selectbox("País B", all_countries, index=all_countries.index("Mexico") if "Mexico" in all_countries else 1)
        
    c_a = country_totals[country_totals['Country_Region'] == p_a].iloc[0]
    c_b = country_totals[country_totals['Country_Region'] == p_b].iloc[0]
    
    counts = np.array([c_a['Deaths'], c_b['Deaths']])
    nobs = np.array([c_a['Confirmed'], c_b['Confirmed']])
    z_stat, p_val = proportions_ztest(counts, nobs, alternative='two-sided')
    
    with col_h3:
        st.metric("Estadístico Z", f"{z_stat:.4f}")
        st.metric("P-Valor", f"{p_val:.4e}")

    if p_val < 0.05:
        st.error(f"Conclusión: Se rechaza H0. Existe diferencia estadísticamente significativa entre el CFR de {p_a} ({c_a['CFR']:.2f}%) y {p_b} ({c_b['CFR']:.2f}%).")
    else:
        st.success(f"Conclusión: No se rechaza H0. No hay evidencia estadística suficiente de diferencia entre {p_a} y {p_b}.")

    st.markdown("##### Distribución y Valores Atípicos (Boxplots)")
    fig_box = px.box(
        filtered_df.melt(id_vars=['Country_Region'], value_vars=['Confirmed', 'Deaths']),
        x='variable', y='value', color='variable', log_y=True,
        title="Diagrama de Cajas (Escala Logarítmica)"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("##### Gráfico de Control Global (3σ) - Fallecidos por País")
    mean_d = filtered_df['Deaths'].mean()
    std_d = filtered_df['Deaths'].std()
    ucl = mean_d + 3 * std_d
    lcl = max(0, mean_d - 3 * std_d)

    fig_ctrl = go.Figure()
    fig_ctrl.add_trace(go.Scatter(x=filtered_df['Country_Region'], y=filtered_df['Deaths'], mode='lines+markers', name='Fallecidos'))
    fig_ctrl.add_hline(y=mean_d, line_dash="dash", line_color="green", annotation_text=f"Media: {mean_d:.0f}")
    fig_ctrl.add_hline(y=ucl, line_dash="dash", line_color="red", annotation_text=f"LSC (3σ): {ucl:.0f}")
    fig_ctrl.add_hline(y=lcl, line_dash="dash", line_color="orange", annotation_text=f"LIC (3σ): {lcl:.0f}")
    fig_ctrl.update_layout(title="Carta de Control Shewhart (3-Sigma)", xaxis_title="País", yaxis_title="Fallecidos")
    st.plotly_chart(fig_ctrl, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: MODELADO Y PROYECCIONES TEMPORALES (SARIMA OPTIMIZADO)
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Modelado Predictivo y Proyecciones a 14 Días")
    
    country_forecast = st.selectbox("Selecciona un país para el pronóstico", all_countries, index=all_countries.index("Peru") if "Peru" in all_countries else 0)
    
    c_ts = ts_raw[ts_raw['Country/Region'] == country_forecast].iloc[:, 4:].sum(axis=0)
    c_ts.index = pd.to_datetime(c_ts.index, format='%m/%d/%y')
    
    # Preprocesamiento de la serie temporal
    daily = c_ts.diff().dropna()
    daily[daily < 0] = 0
    smoothed = daily.rolling(window=7).mean().dropna().asfreq('D')
    
    # Ventana reciente para evitar saltos estructurales antiguos
    recent_smoothed = smoothed.iloc[-90:]
    log_smoothed = np.log1p(recent_smoothed)
    
    steps = 14
    train_log = log_smoothed.iloc[:-steps]
    test_actual = recent_smoothed.iloc[-steps:]
    
    with st.spinner("Ajustando modelo de series de tiempo..."):
        try:
            model_sarima = SARIMAX(train_log, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
                                  enforce_stationarity=False, enforce_invertibility=False)
            res_sarima = model_sarima.fit(disp=False)
            
            fc_log = res_sarima.get_forecast(steps=steps)
            pred_log = fc_log.predicted_mean
            ci_log = fc_log.conf_int(alpha=0.05)
            
            # Revertir escala logarítmica
            pred_vals = np.expm1(pred_log)
            ci_lower = np.maximum(0, np.expm1(ci_log.iloc[:, 0]))
            ci_upper = np.expm1(ci_log.iloc[:, 1])
            
            mae_m = mean_absolute_error(test_actual, pred_vals)
            mape_m = mean_absolute_percentage_error(test_actual, pred_vals) * 100
            
            cm1, cm2 = st.columns(2)
            cm1.metric("MAE (Error Absoluto Medio)", f"{mae_m:.2f} casos")
            cm2.metric("MAPE (Error % Absoluto Medio)", f"{mape_m:.2f}%")
            
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=recent_smoothed.index[:-steps], y=recent_smoothed.iloc[:-steps], name='Entrenamiento Suavizado (7d)'))
            fig_fc.add_trace(go.Scatter(x=test_actual.index, y=test_actual, mode='lines+markers', name='Valor Real (Test)', line=dict(color='black')))
            fig_fc.add_trace(go.Scatter(x=test_actual.index, y=pred_vals, name='Pronóstico SARIMA', line=dict(color='red', dash='dash')))
            fig_fc.add_trace(go.Scatter(
                x=test_actual.index.tolist() + test_actual.index.tolist()[::-1],
                y=ci_upper.tolist() + ci_lower.tolist()[::-1],
                fill='toself', fillcolor='rgba(255, 0, 0, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Bandas de Confianza 95%'
            ))
            fig_fc.update_layout(title=f"Proyección a 14 Días - {country_forecast}", xaxis_title="Fecha", yaxis_title="Casos Diarios")
            st.plotly_chart(fig_fc, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo ajustar el modelo SARIMA para {country_forecast}: {e}")

# ------------------------------------------------------------------------------
# TAB 4: CLUSTERING Y PCA
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Segmentación de Países (K-Means y PCA)")
    
    # Preparación de datos
    c_features = df.groupby('Country_Region').agg({
        'Confirmed': 'sum',
        'Deaths': 'sum',
        'Incident_Rate': 'mean'
    }).dropna()
    c_features = c_features[c_features['Confirmed'] >= 5000].copy()
    c_features['CFR'] = (c_features['Deaths'] / c_features['Confirmed']) * 100
    
    # Tasa de crecimiento aproximada a 7 días
    recent_growth = ts_raw.iloc[:, [1, -8, -1]].groupby('Country/Region').sum()
    recent_growth['Growth_7d'] = ((recent_growth.iloc[:, 1] - recent_growth.iloc[:, 0]) / (recent_growth.iloc[:, 0] + 1)) * 100
    
    merged_cluster_df = c_features.join(recent_growth[['Growth_7d']]).dropna()
    cluster_vars = ['Incident_Rate', 'CFR', 'Growth_7d']
    
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(merged_cluster_df[cluster_vars])
    
    k_val = st.slider("Selecciona el número de clústeres (k)", min_value=2, max_value=6, value=4)
    kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    merged_cluster_df['Cluster'] = kmeans.fit_predict(scaled_matrix)
    
    # PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(scaled_matrix)
    merged_cluster_df['PCA1'] = pca_coords[:, 0]
    merged_cluster_df['PCA2'] = pca_coords[:, 1]
    
    var_pca = pca.explained_variance_ratio_ * 100
    st.markdown(f"**Varianza Total Explicada por PCA:** `{sum(var_pca):.2f}%` (PC1: `{var_pca[0]:.1f}%`, PC2: `{var_pca[1]:.1f}%`)")
    
    fig_pca = px.scatter(
        merged_cluster_df.reset_index(),
        x='PCA1', y='PCA2', color=merged_cluster_df['Cluster'].astype(str),
        hover_name='Country_Region', size='Confirmed',
        title="Proyección PCA Bidimensional por Clúster",
        labels={'color': 'Clúster'}
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    
    st.markdown("##### Perfil Promedio de los Clústeres")
    st.dataframe(merged_cluster_df.groupby('Cluster')[cluster_vars + ['Confirmed']].mean(), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 5: CALIDAD DE DATOS Y EXPORTACIÓN
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("Calidad de Datos y Extracciones")
    
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown("#### Valores Faltantes (Nulos)")
        nulls_df = df.isnull().sum().to_frame(name="Cantidad de Nulos")
        st.dataframe(nulls_df, use_container_width=True)
        
    with col_q2:
        st.markdown("#### Exportar Muestra Aleatoria (Requisito 1.1)")
        sample_df = df.sample(n=50, random_state=42)
        cols_drop_idx = [sample_df.columns[i] for i in [0, 1, 5, 6, 11] if i < len(sample_df.columns)]
        clean_sample = sample_df.drop(columns=cols_drop_idx)
        
        st.dataframe(clean_sample.head(5), use_container_width=True)
        csv_sample = clean_sample.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Muestra Limpia (CSV)",
            data=csv_sample,
            file_name="muestra_limpia_covid.csv",
            mime="text/csv"
        )
        
    st.info("💡 **Nota:** Todas las gráficas del dashboard (Plotly) incluyen un botón integrado en su barra de herramientas superior derecha para exportarlas directamente en formato PNG o SVG.")

# ==============================================================================
# DESCARGA GLOBAL DE DATOS PROCESADOS
# ==============================================================================
st.sidebar.markdown("---")
csv_filtered = filtered_df_selected.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="💾 Descargar Resumen Filtrado (CSV)",
    data=csv_filtered,
    file_name="resumen_covid_filtrado.csv",
    mime="text/csv"
)