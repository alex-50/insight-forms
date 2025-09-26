import streamlit as st
import plotly.express as px


def show_correlations(df):
    st.subheader("🔗 Корреляционный анализ")

    # Проверка на пустой DataFrame
    if df.empty:
        st.error("DataFrame пустой!")
        return

    st.write("Размер DataFrame:", df.shape)
    st.write("Первые 5 строк:")
    st.write(df.head())

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Отладочная информация
    st.write(f"📊 Всего числовых столбцов: {len(numeric_cols)}")
    if numeric_cols:
        st.write("🔍 Числовые столбцы:", numeric_cols)

    if len(numeric_cols) < 2:
        st.warning("Нужно хотя бы 2 числовых столбца для анализа корреляций")
        return

    st.sidebar.markdown("### Настройка корреляций")
    selected_cols = st.sidebar.multiselect(
        "Выберите столбцы для анализа корреляций:",
        numeric_cols,
        default=numeric_cols[:min(10, len(numeric_cols))]  # Ограничение по умолчанию
    )

    if len(selected_cols) < 2:
        st.warning("Выберите хотя бы два числовых столбца.")
        return

    # Работаем только с выбранными столбцами
    analysis_df = df[selected_cols].dropna()

    if len(analysis_df) == 0:
        st.error("После удаления пропущенных значений данных не осталось!")
        return

    st.write(f"📈 Анализируем {len(analysis_df)} строк с {len(selected_cols)} столбцами")

    corr_matrix = analysis_df.corr()

    # Тепловая карта
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title="Тепловая карта корреляций",
    )
    fig_heatmap.update_layout(width=800, height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Настройки порогов
    pos_threshold = st.sidebar.slider("Порог положительной связи", 0.0, 1.0, 0.6, 0.05)
    neg_threshold = st.sidebar.slider("Порог отрицательной связи", -1.0, 0.0, -0.6, 0.05)

    # Поиск сильных связей
    strong_pos = []
    strong_neg = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if val >= pos_threshold:
                strong_pos.append((corr_matrix.columns[i], corr_matrix.columns[j], val))
            elif val <= neg_threshold:
                strong_neg.append((corr_matrix.columns[i], corr_matrix.columns[j], val))

    # Функция для создания scatter plot
    def create_scatter_plot(x_col, y_col, corr_val, correlation_type):
        # Проверка данных
        plot_data = analysis_df[[x_col, y_col]].dropna()

        if len(plot_data) < 2:
            st.warning(f"Недостаточно данных для {x_col} vs {y_col}")
            return None

        fig = px.scatter(
            plot_data,
            x=x_col,
            y=y_col,
            title=f"{correlation_type}: {x_col} vs {y_col} (r={corr_val:.2f})",
            trendline="ols",
            template="plotly_white"
        )
        return fig

    # Отображение положительных связей
    if strong_pos:
        st.markdown(f"### ✅ Сильные положительные связи (r ≥ {pos_threshold})")
        for x_col, y_col, corr_val in strong_pos:
            st.write(f"**{x_col}** ↔ **{y_col}** (r = {corr_val:.3f})")
            fig = create_scatter_plot(x_col, y_col, corr_val, "Положительная связь")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
    else:
        st.info(f"Нет сильных положительных связей (r ≥ {pos_threshold})")

    # Отображение отрицательных связей
    if strong_neg:
        st.markdown(f"### 🔻 Сильные отрицательные связи (r ≤ {neg_threshold})")
        for x_col, y_col, corr_val in strong_neg:
            st.write(f"**{x_col}** ↔ **{y_col}** (r = {corr_val:.3f})")
            fig = create_scatter_plot(x_col, y_col, corr_val, "Отрицательная связь")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
    else:
        st.info(f"Нет сильных отрицательных связей (r ≤ {neg_threshold})")
