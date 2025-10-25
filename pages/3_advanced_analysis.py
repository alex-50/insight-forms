import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score  # Добавлен импорт для silhouette_score
import pandas as pd
import numpy as np
import scipy.stats as stats

st.set_page_config(page_title="Продвинутый анализ 🔬", layout="wide")

def show_correlations(df):
    st.subheader("🔗 Корреляционный анализ")

    if df.empty:
        st.error("DataFrame пустой!")
        return

    st.write("Размер DataFrame:", df.shape)
    st.write("Первые 5 строк:")
    st.write(df.head())

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

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

    analysis_df = df[selected_cols].dropna()

    if len(analysis_df) == 0:
        st.error("После удаления пропущенных значений данных не осталось!")
        return

    st.write(f"📈 Анализируем {len(analysis_df)} строк с {len(selected_cols)} столбцами")

    corr_matrix = analysis_df.corr()

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

    pos_threshold = st.sidebar.slider("Порог положительной связи", 0.0, 1.0, 0.6, 0.05)
    neg_threshold = st.sidebar.slider("Порог отрицательной связи", -1.0, 0.0, -0.6, 0.05)

    strong_pos = []
    strong_neg = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if val >= pos_threshold:
                strong_pos.append((corr_matrix.columns[i], corr_matrix.columns[j], val))
            elif val <= neg_threshold:
                strong_neg.append((corr_matrix.columns[i], corr_matrix.columns[j], val))

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


def show_clustering(df):
    st.subheader("🌀 Кластеризация (KMeans)")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Для кластеризации нужно минимум 2 числовых столбца.")
        return

    selected_cols = st.multiselect(
        "Выберите параметры для кластеризации:",
        numeric_cols,
        default=numeric_cols[:2]
    )

    if len(selected_cols) < 2:
        st.warning("Выберите хотя бы 2 параметра.")
        return

    X = df[selected_cols].dropna()
    if len(X) < 2:
        st.warning("Недостаточно данных для кластеризации (нужно минимум 2 строки).")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if 'n_clusters' not in st.session_state:
        st.session_state['n_clusters'] = 3

    if st.button("Определить оптимальное количество кластеров"):
        max_k = min(10, len(X) - 1)  # Максимум k - не больше, чем строк минус 1
        if max_k < 2:
            st.warning("Недостаточно данных для определения оптимального k.")
            return

        silhouette_scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)

        optimal_k = range(2, max_k + 1)[np.argmax(silhouette_scores)]
        st.session_state['n_clusters'] = optimal_k
        st.success(f"Оптимальное количество кластеров (по Silhouette score): {optimal_k}")

    n_clusters = st.slider(
        "Количество кластеров (k):",
        min_value=2,
        max_value=10,
        value=st.session_state['n_clusters']
    )

    st.session_state['n_clusters'] = n_clusters

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df_result = X.copy()
    df_result["Cluster"] = clusters

    st.write("### Центроиды кластеров")
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=selected_cols)
    centroids["Cluster"] = range(n_clusters)
    st.dataframe(centroids)

    if len(selected_cols) == 2:
        fig = px.scatter(
            df_result, x=selected_cols[0], y=selected_cols[1],
            color="Cluster", title="Кластеры студентов",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.scatter_matrix(
            df_result,
            dimensions=selected_cols,
            color="Cluster",
            title="Кластеры студентов (scatter matrix)"
        )
        st.plotly_chart(fig, use_container_width=True)


def _t_test(df: pd.DataFrame, col_group: str, col_value: str):
    groups = df[col_group].dropna().unique()
    if len(groups) != 2:
        return None, "Для t-test нужно строго 2 группы (2 уникальных значения в колонке с группами)."

    g1, g2 = groups[0], groups[1]
    s1 = df[df[col_group] == g1][col_value].dropna()
    s2 = df[df[col_group] == g2][col_value].dropna()
    if len(s1) < 2 or len(s2) < 2:
        return None, "Недостаточно наблюдений в одной из групп (нужно >=2)."

    stat, pval = stats.ttest_ind(s1, s2, equal_var=False)
    return {"statistic": stat, "pvalue": pval, "group_names": (g1, g2), "n1": len(s1), "n2": len(s2)}, None


def _chi2(df: pd.DataFrame, col1: str, col2: str):
    contingency = pd.crosstab(df[col1], df[col2])
    if contingency.size == 0:
        return None, "Контингентная таблица пустая."
    chi2, pval, dof, expected = stats.chi2_contingency(contingency)
    return {"chi2": chi2, "pvalue": pval, "dof": dof, "expected": expected, "contingency": contingency}, None


def _anova(df: pd.DataFrame, col_group: str, col_value: str):
    unique_groups = df[col_group].dropna().unique()
    groups = [df[df[col_group] == g][col_value].dropna() for g in unique_groups]
    if len(groups) < 2:
        return None, "Для ANOVA нужно минимум 2 группы."
    if any(len(g) < 2 for g in groups):
        return None, "Недостаточно наблюдений в одной из групп (нужно >=2)."
    stat, pval = stats.f_oneway(*groups)
    return {"F": stat, "pvalue": pval, "groups": unique_groups}, None


def show_stats(df: pd.DataFrame):
    st.subheader("📊 Статистический анализ (t-test, χ², ANOVA)")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # категориальные = все нечисловые + целые с небольшим числом уникальных значений
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    int_like = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() <= 10]
    for c in int_like:
        if c not in cat_cols:
            cat_cols.append(c)

    test = st.selectbox("Выберите тест:", ["t-test", "χ² (хи-квадрат)", "ANOVA"])

    if test == "t-test":
        group_candidates = [c for c in df.columns if df[c].nunique() == 2]
        if not group_candidates:
            st.warning("Нет колонок с ровно 2 уникальными значениями для групп (нужна бинарная группа).")
            return
        col_group = st.selectbox("Колонка с группами (2 уникума):", group_candidates)
        if not num_cols:
            st.warning("В таблице нет числовых колонок для сравнения.")
            return
        col_value = st.selectbox("Числовая колонка (значение):", num_cols)
        if st.button("Выполнить t-test"):
            res, err = _t_test(df, col_group, col_value)
            if err:
                st.error(err)
            else:
                st.write(f"t = {res['statistic']:.4f}, p = {res['pvalue']:.4f}")
                st.write("Интерпретация:", "✅ различие статистически значимо (p < 0.05)" if res[
                                                                                                'pvalue'] < 0.05 else "ℹ️ различие не значимо (p >= 0.05)")
                st.write(f"Размеры групп: {res['group_names'][0]}: {res['n1']}, {res['group_names'][1]}: {res['n2']}")

                fig = px.box(df, x=col_group, y=col_value, points="all", title=f"{col_value} по {col_group}")
                st.plotly_chart(fig, use_container_width=True)

    elif test == "χ² (хи-квадрат)":
        if len(cat_cols) < 2:
            st.warning("Недостаточно категориальных колонок для χ² (нужно минимум 2).")
            return
        col1 = st.selectbox("Категориальная колонка 1:", cat_cols, index=0)
        col2 = st.selectbox("Категориальная колонка 2:", [c for c in cat_cols if c != col1], index=0)
        if st.button("Выполнить χ²"):
            res, err = _chi2(df, col1, col2)
            if err:
                st.error(err)
            else:
                st.write(f"χ² = {res['chi2']:.4f}, p = {res['pvalue']:.4f}, dof = {res['dof']}")
                st.write("Интерпретация:",
                         "✅ зависимость между переменными" if res['pvalue'] < 0.05 else "ℹ️ зависимость не обнаружена")
                st.markdown("**Контингентная таблица:**")
                st.dataframe(res["contingency"])

    elif test == "ANOVA":
        group_candidates = [c for c in df.columns if df[c].nunique() >= 2]
        if not group_candidates or not num_cols:
            st.warning("Недостаточно данных (нужны группирующая колонка и числовая колонка).")
            return
        col_group = st.selectbox("Группирующая колонка:", group_candidates)
        col_value = st.selectbox("Числовая колонка:", num_cols)
        if st.button("Выполнить ANOVA"):
            res, err = _anova(df, col_group, col_value)
            if err:
                st.error(err)
            else:
                st.write(f"F = {res['F']:.4f}, p = {res['pvalue']:.4f}")
                st.write("Интерпретация:", "✅ найдены различия между группами (p < 0.05)" if res[
                                                                                                 'pvalue'] < 0.05 else "ℹ️ различия не выявлены")
                fig = px.box(df, x=col_group, y=col_value, points="all", title=f"{col_value} по {col_group}")
                st.plotly_chart(fig, use_container_width=True)


st.title("🔬 Продвинутый анализ")
if 'df' in st.session_state:
    df = st.session_state['df']
    analysis_type = st.sidebar.radio(
        "Выберите тип продвинутого анализа:",
        ["Корреляции 📉", "Кластеризация 🆎", "Статистический анализ 🧮"]
    )
    if analysis_type == "Корреляции 📉":
        show_correlations(df)
    elif analysis_type == "Кластеризация 🆎":
        show_clustering(df)
    elif analysis_type == "Статистический анализ 🧮":
        show_stats(df)
else:
    st.info("Загрузите данные на главной странице.")
