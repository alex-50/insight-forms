import streamlit as st
import pandas as pd


def show_type_config(df):
    st.subheader("⚙️ Настройка типов данных")

    # Инициализация session_state для хранения типов столбцов, если ещё не создано
    if 'column_types' not in st.session_state:
        st.session_state.column_types = {}
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                if pd.api.types.is_integer_dtype(dtype) and df[col].nunique() <= 10:
                    param_type = "Категориальный"
                else:
                    param_type = "Количественный"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                param_type = "Временной"
            elif pd.api.types.is_object_dtype(dtype):
                unique_count = df[col].nunique()
                avg_length = df[col].dropna().apply(lambda x: len(str(x))).mean()
                if unique_count <= 10 or avg_length < 20:
                    param_type = "Категориальный"
                else:
                    param_type = "Текстовый"
            else:
                param_type = "Игнорировать"
            st.session_state.column_types[col] = param_type

    # Таблица параметров с выбором типов
    st.write("### Выберите типы данных для столбцов")
    st.markdown("Выберите подходящий тип для каждого столбца. Это повлияет на визуализации и анализ.")
    type_options = ["Количественный", "Категориальный", "Временной", "Текстовый", "Игнорировать"]
    param_types = []
    for col in df.columns:
        selected_type = st.selectbox(
            f"Тип для {col}",
            options=type_options,
            index=type_options.index(st.session_state.column_types[col]),
            key=f"type_select_{col}"
        )
        st.session_state.column_types[col] = selected_type
        param_types.append({
            'Параметр': col,
            'Тип': selected_type
        })
    st.dataframe(pd.DataFrame(param_types))

    st.info("Изменённые типы данных будут использованы в визуализациях на странице 'Базовый просмотр'.")