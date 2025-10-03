# app.py
import streamlit as st
import pandas as pd

from utils.data_view import show_data_overview
from utils.correlations import show_correlations
from utils.clustering import show_clustering
from utils.stats_analysis import show_stats
from utils.type_config import show_type_config

st.set_page_config(page_title="Student Survey Analyzer", layout="wide")
st.title("📊 Student Survey Analyzer")

# --- Загрузка CSV ---
st.sidebar.header("Загрузка данных")
uploaded_file = st.sidebar.file_uploader("Выберите CSV файл", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Файл успешно загружен!")

    # --- Меню анализа ---
    st.sidebar.header("Выберите анализ")
    analysis_type = st.sidebar.radio(
        "Что делать?",
        [
            "Базовый просмотр",
            "Настройка типов данных",
            "Корреляции",
            "Кластеризация",
            "Статистический анализ"
        ]
    )

    if analysis_type == "Базовый просмотр":
        show_data_overview(df)
    elif analysis_type == "Настройка типов данных":
        show_type_config(df)
    elif analysis_type == "Корреляции":
        show_correlations(df)
    elif analysis_type == "Кластеризация":
        show_clustering(df)
    elif analysis_type == "Статистический анализ":
        show_stats(df)
else:
    st.info("⬅ Загрузите CSV через боковое меню, чтобы начать работу.")