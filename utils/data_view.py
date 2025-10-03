import streamlit as st
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def show_data_overview(df):
    st.subheader("📋 Базовый просмотр данных")

    # Размер DataFrame
    st.write("### Размер DataFrame:", df.shape)

    # Первые строки и общая инфа
    st.write("### Первые строки таблицы")
    st.dataframe(df.head())

    st.write("### Общая информация")
    st.write(df.describe(include="all"))

    # Информация о типах данных
    st.write("### Информация о типах данных:")
    info_data = []
    for col in df.columns:
        info_data.append({
            'Столбец': col,
            'Тип': str(df[col].dtype),
            'Не-NULL': df[col].count(),
            'Всего': len(df),
            'Уникальных': df[col].nunique()
        })
    st.dataframe(pd.DataFrame(info_data))

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

    # Таблица параметров и их типов (статическая)
    st.write("### Типы параметров")
    param_types = []
    for col in df.columns:
        param_types.append({
            'Параметр': col,
            'Тип': st.session_state.column_types[col]
        })
    st.dataframe(pd.DataFrame(param_types))

    # Определяем столбцы по типам для визуализаций
    numeric_cols = [col for col, col_type in st.session_state.column_types.items() if col_type == "Количественный"]
    categorical_cols = [col for col, col_type in st.session_state.column_types.items() if col_type == "Категориальный"]
    text_cols = [col for col, col_type in st.session_state.column_types.items() if col_type == "Текстовый"]
    time_cols = [col for col, col_type in st.session_state.column_types.items() if col_type == "Временной"]

    st.write("### Визуализация признаков")

    # Настройки от пользователя для всех типов данных
    st.sidebar.markdown("### Настройка базового просмотра")
    selected_numeric = st.sidebar.multiselect(
        "Числовые для гистограмм:", numeric_cols, default=[]
    )
    selected_categorical = st.sidebar.multiselect(
        "Категориальные для графиков:", categorical_cols, default=[]
    )
    selected_text = st.sidebar.multiselect(
        "Текстовые для облаков слов:", text_cols, default=[]
    )
    selected_time = st.sidebar.multiselect(
        "Временные для графиков:", time_cols, default=[]
    )

    # Числовые → гистограммы
    for col in selected_numeric:
        st.markdown(f"**Гистограмма: {col}**")
        fig = px.histogram(df, x=col, nbins=20, title=f"Распределение {col}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # Категориальные → bar или pie
    cat_chart_types = {}
    for col in selected_categorical:
        chart_type = st.sidebar.radio(
            f"Тип графика для {col}",
            options=["bar", "pie"],
            index=0,
            key=f"chart_type_{col}"
        )
        cat_chart_types[col] = chart_type
        st.markdown(f"**{col}**")
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, "count"]
        if chart_type == "bar":
            fig = px.bar(value_counts, y=col, x="count", title=f"Распределение {col}", template="plotly_white")
        else:
            fig = px.pie(value_counts, names=col, values="count", title=f"Распределение {col}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # Текстовые → облака слов
    for col in selected_text:
        st.markdown(f"**Облако слов: {col}**")
        text = " ".join(str(val) for val in df[col].dropna())
        if not text.strip():
            st.warning(f"Нет текста для отображения в столбце {col}.")
            continue
        wordcloud = WordCloud(width=1200, height=800, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig, use_container_width=False)

    # Временные → lineplot или гистограмма
    time_chart_types = {}
    for col in selected_time:
        chart_type = st.sidebar.radio(
            f"Тип графика для {col}",
            options=["histogram", "line"],
            index=0,
            key=f"time_chart_type_{col}"
        )
        time_chart_types[col] = chart_type
        st.markdown(f"**{col}**")
        if chart_type == "histogram":
            fig = px.histogram(df, x=col, title=f"Распределение {col} по времени", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Для линейного графика нужен числовой столбец для агрегации
            if numeric_cols:
                value_col = st.sidebar.selectbox(
                    f"Числовой столбец для агрегации по {col}",
                    options=numeric_cols,
                    key=f"time_value_col_{col}"
                )
                agg_df = df.groupby(col)[value_col].mean().reset_index()
                fig = px.line(agg_df, x=col, y=value_col, title=f"Среднее {value_col} по {col}", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Для линейного графика по {col} нужен хотя бы один числовой столбец.")