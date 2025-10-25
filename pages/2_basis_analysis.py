import streamlit as st
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
st.set_page_config(page_title="Базовый анализ", layout="wide")


@st.cache_data
def generate_wordcloud(text):
    return WordCloud(width=1200, height=800, background_color="white").generate(text)


def show_data_overview(df):
    st.subheader("📋 Базовый просмотр данных")

    st.write("### Размер DataFrame:", df.shape)

    st.write("### Первые строки таблицы")
    st.dataframe(df.head())

    st.write("### Общая информация")
    st.write(df.describe(include="all"))

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

    st.write("### Типы параметров")
    param_types = []
    for col in df.columns:
        param_types.append({
            'Параметр': col,
            'Тип': st.session_state.column_types[col]
        })
    st.dataframe(pd.DataFrame(param_types))

    numeric_cols = [col for col, col_type in st.session_state.column_types.items() if col_type == "Количественный"]
    categorical_cols = [col for col, col_type in st.session_state.column_types.items() if
                        col_type == "Категориальный" and df[col].nunique() <= 10]
    text_cols = [col for col, col_type in st.session_state.column_types.items() if col_type == "Текстовый"]

    st.write("### Визуализация признаков")

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

    if selected_numeric:
        st.markdown("#### Количественные данные")
        for col in selected_numeric:
            st.markdown(f"**Гистограмма: {col}**")
            group_col = st.selectbox(
                "Группировать по категориальному столбцу:",
                ["Без группировки"] + categorical_cols,
                index=0,
                key=f"group_col_numeric_{col}"
            )
            group_col = None if group_col == "Без группировки" else group_col
            fig = px.histogram(
                df, x=col, color=group_col, nbins=20,
                title=f"Распределение {col} {f'по {group_col}' if group_col else ''}", template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

    if selected_categorical:
        st.markdown("#### Категориальные данные")
        cat_chart_types = {}
        for col in selected_categorical:
            st.markdown(f"**{col}**")
            group_col = st.selectbox(
                "Группировать по категориальному столбцу:",
                ["Без группировки"] + [c for c in categorical_cols if c != col],
                index=0,
                key=f"group_col_categorical_{col}"
            )
            group_col = None if group_col == "Без группировки" else group_col
            chart_type = st.sidebar.radio(
                f"Тип графика для {col}",
                options=["bar", "pie"],
                index=0,
                key=f"chart_type_{col}"
            )
            cat_chart_types[col] = chart_type
            if chart_type == "bar":
                if group_col:
                    value_counts = df.groupby([group_col, col]).size().reset_index(name="count")
                    fig = px.bar(
                        value_counts, x=col, y="count", color=group_col,
                        title=f"Распределение {col} по {group_col}", template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = [col, "count"]
                    fig = px.bar(
                        value_counts, y=col, x="count", title=f"Распределение {col}", template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                if group_col:
                    for group in df[group_col].dropna().unique():
                        group_df = df[df[group_col] == group]
                        value_counts = group_df[col].value_counts().reset_index()
                        value_counts.columns = [col, "count"]
                        fig = px.pie(
                            value_counts, names=col, values="count",
                            title=f"Распределение {col} для {group_col}={group}", template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = [col, "count"]
                    fig = px.pie(
                        value_counts, names=col, values="count", title=f"Распределение {col}", template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    if selected_text:
        st.markdown("#### Текстовые данные")
        for col in selected_text:
            st.markdown(f"**Облако слов: {col}**")
            group_col = st.selectbox(
                "Группировать по категориальному столбцу:",
                ["Без группировки"] + categorical_cols,
                index=0,
                key=f"group_col_text_{col}"
            )
            group_col = None if group_col == "Без группировки" else group_col
            if group_col:
                for group in df[group_col].dropna().unique():
                    st.markdown(f"**Облако слов: {col} для {group_col}={group}**")
                    group_text = " ".join(str(val) for val in df[df[group_col] == group][col].dropna())
                    if not group_text.strip():
                        st.warning(f"Нет текста для {col} в группе {group}.")
                        continue
                    wordcloud = generate_wordcloud(group_text)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig, use_container_width=False)
            else:
                text = " ".join(str(val) for val in df[col].dropna())
                if not text.strip():
                    st.warning(f"Нет текста для отображения в столбце {col}.")
                    continue
                wordcloud = generate_wordcloud(text)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig, use_container_width=False)


st.title("🤓 Базовый анализ")
if 'df' in st.session_state:
    show_data_overview(st.session_state['df'])
else:
    st.info("Загрузите данные на главной странице.")