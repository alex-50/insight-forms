"""
Страница «Базовый анализ».

Отображает:
- размер, первые строки, описательную статистику (без игнорируемых столбцов);
- таблицу типов параметров;
- визуализации: гистограммы, bar/pie, wordcloud — по выбору пользователя в сайдбаре.

Все списки столбцов строятся из st.session_state.column_types, поэтому
столбцы с типом «Игнорировать» нигде не участвуют.
"""
import streamlit as st
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from plot_styles import apply_bar_style, apply_pie_style
from type_config import init_column_types


@st.cache_data
def generate_wordcloud(text: str) -> WordCloud:
    """
    Кэшированная генерация облака слов.

    Args:
        text: объединённый текст для облака.

    Returns:
        Объект WordCloud с построенной картой слов.
    """
    return WordCloud(width=1200, height=800, background_color="white").generate(text)


def show_data_overview(df: pd.DataFrame) -> None:
    """
    Отображает страницу базового анализа и визуализации.

    Args:
        df: DataFrame из st.session_state['df'].
    """
    st.subheader("📋 Базовый просмотр данных")

    st.write("### Размер DataFrame:", df.shape)

    st.write("### Первые строки таблицы")
    st.dataframe(df.head())

    init_column_types(df)

    active_cols = [
        c for c in df.columns
        if st.session_state.column_types.get(c) != "Игнорировать"
    ]

    st.write("### Общая информация")
    if active_cols:
        st.write(df[active_cols].describe(include="all"))
    else:
        st.info("Все столбцы помечены как «Игнорировать».")

    st.write("### Информация о типах данных:")
    info_data = []
    for col in active_cols:
        info_data.append({
            'Столбец': col,
            'Тип': str(df[col].dtype),
            'Не-NULL': df[col].count(),
            'Всего': len(df),
            'Уникальных': df[col].nunique()
        })
    st.dataframe(pd.DataFrame(info_data))

    st.write("### Типы параметров")
    param_types = [
        {'Параметр': col, 'Тип': st.session_state.column_types.get(col, "—")}
        for col in df.columns
    ]
    st.dataframe(pd.DataFrame(param_types))

    numeric_cols = [
        c for c, t in st.session_state.column_types.items()
        if t == "Количественный" and c in df.columns
    ]
    categorical_cols = [
        c for c, t in st.session_state.column_types.items()
        if t == "Категориальный" and c in df.columns and df[c].nunique() <= 10
    ]
    text_cols = [
        c for c, t in st.session_state.column_types.items()
        if t == "Текстовый" and c in df.columns
    ]

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

    # --- Числовые ---
    if selected_numeric:
        st.markdown("#### Количественные данные")
        for col in selected_numeric:
            if col not in df.columns:
                continue
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
                title=f"Распределение {col} {f'по {group_col}' if group_col else ''}",
                template="plotly_white",
                text_auto=True
            )
            fig = apply_bar_style(fig)
            st.plotly_chart(fig, use_container_width=True)

    # --- Категориальные ---
    if selected_categorical:
        st.markdown("#### Категориальные данные")

        for col in selected_categorical:
            if col not in df.columns:
                continue
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

            if chart_type == "bar":
                if group_col:
                    value_counts = df.groupby([group_col, col]).size().reset_index(name="count")
                    fig = px.bar(
                        value_counts, x=col, y="count", color=group_col,
                        title=f"Распределение {col} по {group_col}",
                        template="plotly_white",
                        text_auto=True
                    )
                else:
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = [col, "count"]
                    fig = px.bar(
                        value_counts, y=col, x="count",
                        title=f"Распределение {col}",
                        template="plotly_white",
                        text_auto=True
                    )
                fig = apply_bar_style(fig)
                st.plotly_chart(fig, use_container_width=True)

            else:  # pie
                if group_col:
                    for group in df[group_col].dropna().unique():
                        st.markdown(f"**{col} для {group_col} = {group}**")
                        grp = df[df[group_col] == group]
                        value_counts = grp[col].value_counts().reset_index()
                        value_counts.columns = [col, "count"]
                        fig = px.pie(
                            value_counts, names=col, values="count",
                            title=f"Распределение {col} ({group})",
                            template="plotly_white"
                        )
                        fig = apply_pie_style(fig)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = [col, "count"]
                    fig = px.pie(
                        value_counts, names=col, values="count",
                        title=f"Распределение {col}",
                        template="plotly_white"
                    )
                    fig = apply_pie_style(fig)
                    st.plotly_chart(fig, use_container_width=True)

    # --- Текстовые ---
    if selected_text:
        st.markdown("#### Текстовые данные")

        for col in selected_text:
            if col not in df.columns:
                continue
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
                    st.markdown(f"**{col}, {group_col}={group}**")
                    group_text = " ".join(str(v) for v in df[df[group_col] == group][col].dropna())
                    if not group_text.strip():
                        st.warning(f"Нет текста для {col} в группе {group}.")
                        continue
                    wc = generate_wordcloud(group_text)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
            else:
                text = " ".join(str(v) for v in df[col].dropna())
                if not text.strip():
                    st.warning(f"Нет текста для отображения в столбце {col}.")
                    continue
                wc = generate_wordcloud(text)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)


st.title("🤓 Базовый анализ")

if 'df' in st.session_state:
    show_data_overview(st.session_state['df'])
else:
    st.info("Загрузите данные на главной странице.")