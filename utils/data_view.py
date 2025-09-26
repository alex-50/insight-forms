import streamlit as st
import plotly.express as px
import pandas as pd


def show_data_overview(df):
    st.subheader("📋 Базовый просмотр данных")

    st.write("###  Размер DataFrame:", df.shape)

    # Первые строки и общая инфа
    st.write("### Первые строки таблицы")
    st.dataframe(df.head())

    st.write("### Общая информация")
    st.write(df.describe(include="all"))

    # Альтернативный способ показать информацию о данных
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

    # Определяем числовые и категориальные
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.write("### Визуализация признаков")

    # Настройки от пользователя для числовых и категориальных
    st.sidebar.markdown("### Настройка базового просмотра")
    selected_numeric = st.sidebar.multiselect(
        "Числовые для гистограмм:", numeric_cols, default=[]
    )
    selected_categorical = st.sidebar.multiselect(
        "Категориальные для графиков:", categorical_cols, default=[]
    )

    # Для каждого выбранного категориального признака даём выбор типа графика
    cat_chart_types = {}
    for col in selected_categorical:
        chart_type = st.sidebar.radio(
            f"Тип графика для {col}",
            options=["bar", "pie"],
            index=0,
            key=f"chart_type_{col}"
        )
        cat_chart_types[col] = chart_type

    # Числовые → гистограммы
    for col in selected_numeric:
        st.markdown(f"**{col}**")
        fig = px.histogram(df, x=col, nbins=20, title=f"Распределение {col}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # Категориальные → выбранный пользователем тип графика
    for col in selected_categorical:
        st.markdown(f"**{col}**")
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, "count"]

        if cat_chart_types.get(col) == "bar":
            fig = px.bar(value_counts, y=col, x="count", title=f"Распределение {col}", template="plotly_white")
        else:
            fig = px.pie(value_counts, names=col, values="count", title=f"Распределение {col}", template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)
