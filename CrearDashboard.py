
import streamlit as st 
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard Hospitalario", layout="wide")

st.title("Dashboard de Egresos Hospitalarios") 
st.write("Análisis básico del archivo CSV")

# Cargar datos
archivo = "Listado_egresos_hospitalarios_ene2022_nov2025.csv"


@st.cache_data
def cargar_datos():
    try:
        return pd.read_csv(archivo, encoding="utf-8")
    except:
        return pd.read_csv(archivo, encoding="Latin1")

df = cargar_datos()

# Mostrar información general 
st.subheader("Información general")
col1, col2, col3 = st.columns(3)
col1.metric("Total de registros", len(df))
col2.metric("Total de columnas", df.shape[1])
col3.metric("Valores nulos", df.isnull().sum().sum())


# Vista previa
st.subheader("Vista previa de Los datos") 
st.dataframe(df.head (20), use_container_width=True)

# Selección de columna para filtro
st.sidebar.header("Filtros")
columna_filtro = st.sidebar.selectbox ("Selecciona una columna para filtrar", df.columns)

valores = df[columna_filtro].dropna().astype(str).unique().tolist() 
valores.sort()

valor_seleccionado = st.sidebar.multiselect(
    f"Selecciona valores de {columna_filtro}",
    valores,
    default=valores[:5] if len(valores) > 5 else valores
)

# Aplicar filtro
if valor_seleccionado:
    df_filtrado = df [df[columna_filtro].astype(str).isin(valor_seleccionado)]
else:
    df_filtrado = df.copy()

st.subheader("Datos filtrados")
st.dataframe(df_filtrado, use_container_width=True)

# Selección de columna para gráficos
st.subheader("Gráficos")
columna_grafico = st.selectbox("Selecciona una columna para analizar", df_filtrado.columns)

conteo = df_filtrado [columna_grafico].astype(str).value_counts().reset_index() 
conteo.columns = [columna_grafico, "Cantidad"]

colA, colB = st.columns(2)

with colA:
    st.write("Gráfico de barras") 
    fig_bar = px.bar(
        conteo.head (10),
        x=columna_grafico,
        y="Cantidad",
        text_auto=True
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with colB:
    st.write("Gráfico circular")
    fig_pie = px.pie(
        conteo.head (10),
        names=columna_grafico,
        values="Cantidad"
    )
st.plotly_chart(fig_pie, use_container_width=True)