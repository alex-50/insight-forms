import streamlit as st
import pandas as pd

st.set_page_config(page_title="Конфигурация типов", layout="wide")


def show_type_config(df):
    # Инициализация типов, если ещё не заданы
    if 'column_types' not in st.session_state:
        st.session_state.column_types = {}
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                if pd.api.types.is_integer_dtype(dtype) and df[col].nunique() <= 10:
                    param_type = "Категориальный"
                else:
                    param_type = "Количественный"
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

    # Псевдонимы
    if 'column_aliases' not in st.session_state:
        st.session_state.column_aliases = {col: col for col in df.columns}

    st.write("### 🏷 Переименование столбцов и выбор типа")
    st.markdown("Задайте новое имя и тип данных для каждого столбца (в одной строке).")

    type_options = ["Количественный", "Категориальный", "Текстовый", "Игнорировать"]
    new_names = {}

    # Выводим горизонтально: псевдоним + тип
    for col in df.columns:
        cols = st.columns([3, 2])  # пропорции: 3 для имени, 2 для типа
        with cols[0]:
            alias = st.text_input(
                f"Новое имя для '{col}'",
                value=st.session_state.column_aliases.get(col, col),
                key=f"alias_{col}"
            )
        with cols[1]:
            current_type = st.session_state.column_types.get(col, "Количественный")
            selected_type = st.selectbox(
                f"Тип для '{col}'",
                options=type_options,
                index=type_options.index(current_type),
                key=f"type_select_{col}"
            )

        new_names[col] = alias
        st.session_state.column_types[col] = selected_type

    # Обновляем DataFrame, если имена изменились
    if any(new_names[col] != col for col in df.columns):
        df.rename(columns=new_names, inplace=True)
        st.session_state['df'] = df

        # Обновляем словарь типов под новые имена
        st.session_state.column_types = {
            new_names.get(k, k): v for k, v in st.session_state.column_types.items()
        }

        # Обновляем псевдонимы
        st.session_state.column_aliases = new_names
        st.success("✅ Псевдонимы обновлены")

    # Итоговая таблица
    st.write("### Итоговая конфигурация")
    param_types = [{'Параметр': col, 'Тип': st.session_state.column_types[col]} for col in df.columns]
    st.dataframe(pd.DataFrame(param_types))

    st.info("Изменённые типы и названия будут использоваться на странице 'Базовый просмотр'.")


# Основная логика страницы
st.title("⚙️ Настройка типов данных и псевдонимов")

if 'df' in st.session_state:
    show_type_config(st.session_state['df'])
else:
    st.info("Загрузите данные на главной странице.")
