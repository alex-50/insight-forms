import streamlit as st
import pandas as pd

st.markdown("""
    <style>
    /* Увеличиваем базовый размер шрифта для всего приложения */
    html, body, [class*="css"]  {
        font-size: 18px !important;  /* Базовый размер шрифта */
        font-family: Arial, sans-serif;
    }
    /* Увеличиваем заголовки */
    h1 {
        font-size: 36px !important;  /* Заголовок (st.title) */
    }
    h2 {
        font-size: 28px !important;  /* Подзаголовок (st.subheader) */
    }
    h3 {
        font-size: 24px !important;  /* Подзаголовки в markdown */
    }
    /* Увеличиваем шрифт в таблицах */
    .stDataFrame table {
        font-size: 18px !important;
    }
    /* Увеличиваем шрифт в боковой панели */
    .css-1d391kg, .css-1v3fvcr {
        font-size: 18px !important;
    }
    /* Увеличиваем шрифт в selectbox, multiselect и radio */
    .stSelectbox, .stMultiSelect, .stRadio {
        font-size: 18px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Survey Analyzer 📊", layout="wide")
st.title("📊 Анализ опросов")

# --- Загрузка CSV ---
st.sidebar.header("Загрузка данных")
uploaded_file = st.sidebar.file_uploader("Выберите CSV файл", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['df'] = df
    st.success("Файл успешно загружен! Выберите страницу анализа в боковом меню.")
else:
    st.info("⬅ Загрузите CSV через боковое меню, чтобы начать работу.")
