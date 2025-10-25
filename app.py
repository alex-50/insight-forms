import streamlit as st
import pandas as pd

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