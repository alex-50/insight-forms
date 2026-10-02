"""
Точка входа приложения Survey Analyzer.

- Настраивает страницу (layout, шрифты).
- Определяет навигацию по страницам через st.navigation + st.Page.
- Обеспечивает загрузку CSV в сайдбаре (только один раз на файл).
- Запускает выбранную страницу через pg.run().
"""
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Survey Analyzer 📊", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-size: 18px !important;
        font-family: Arial, sans-serif;
    }
    h1 { font-size: 36px !important; }
    h2 { font-size: 28px !important; }
    h3 { font-size: 24px !important; }
    .stDataFrame table { font-size: 18px !important; }
    .stSelectbox, .stMultiSelect, .stRadio { font-size: 18px !important; }
    </style>
""", unsafe_allow_html=True)

pg = st.navigation([
    st.Page("type_config.py", title="⚙️ Настройка типов", default=True),
    st.Page("basis_analysis.py", title="📋 Базовый анализ"),
    st.Page("advanced_analysis.py", title="🔬 Продвинутый анализ"),
])

st.sidebar.header("📁 Загрузка данных")
uploaded_file = st.sidebar.file_uploader("Выберите CSV файл", type=["csv"])

if uploaded_file is not None:
    if st.session_state.get('_loaded_file') != uploaded_file.name:
        df = pd.read_csv(uploaded_file)
        for c in df.select_dtypes(include=['object']).columns:
            df[c] = df[c].astype(str)
        st.session_state['df'] = df
        st.session_state['_loaded_file'] = uploaded_file.name
        st.session_state.pop('column_types', None)
        st.session_state.pop('column_aliases', None)
        for k in list(st.session_state.keys()):
            if k.startswith(("alias_", "type_select_", "group_col_", "chart_type_", "sel_")):
                del st.session_state[k]
    st.sidebar.success(f"Файл загружен: {uploaded_file.name}")
elif 'df' not in st.session_state:
    st.sidebar.info("Загрузите CSV, чтобы начать работу.")

pg.run()