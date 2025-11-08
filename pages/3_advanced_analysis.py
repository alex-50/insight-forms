import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import scipy.stats as stats
from streamlit_ace import st_ace
import sqlite3

st.set_page_config(page_title="Продвинутый анализ 🔬", layout="wide")


# ==================== Стили графиков ====================
def apply_plot_style(fig):
    fig.update_layout(
        font=dict(size=16, family="Consolas", color="#404040"),
        title_font=dict(size=22, family="Consolas", color="#404040"),
        legend=dict(font=dict(size=14, family="Consolas", color="#404040")),
        xaxis=dict(title_font=dict(size=18, family="Consolas", color="#404040"),
                   tickfont=dict(size=14, family="Consolas", color="#404040")),
        yaxis=dict(title_font=dict(size=18, family="Consolas", color="#404040"),
                   tickfont=dict(size=14, family="Consolas", color="#404040")),
        plot_bgcolor="rgba(255,255,255,1)",
        paper_bgcolor="rgba(255,255,255,0)"
    )
    return fig


# ==================== Корреляции ====================
def show_correlations(df):
    st.subheader("🔗 Корреляционный анализ")
    if df.empty:
        st.error("DataFrame пустой!")
        return
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Нужно хотя бы 2 числовых столбца")
        return
    selected_cols = st.sidebar.multiselect(
        "Выберите столбцы для анализа корреляций:",
        numeric_cols,
        default=numeric_cols[:min(10, len(numeric_cols))]
    )
    if len(selected_cols) < 2:
        st.warning("Выберите хотя бы два столбца.")
        return
    analysis_df = df[selected_cols].dropna()
    corr_matrix = analysis_df.corr()
    fig_heatmap = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                            title="Тепловая карта корреляций")
    fig_heatmap.update_layout(width=800, height=600)
    fig_heatmap = apply_plot_style(fig_heatmap)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    pos_threshold = st.sidebar.slider("Порог положительной связи", 0.0, 1.0, 0.6, 0.05)
    neg_threshold = st.sidebar.slider("Порог отрицательной связи", -1.0, 0.0, -0.6, 0.05)

    strong_pos, strong_neg = [], []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if val >= pos_threshold:
                strong_pos.append((corr_matrix.columns[i], corr_matrix.columns[j], val))
            elif val <= neg_threshold:
                strong_neg.append((corr_matrix.columns[i], corr_matrix.columns[j], val))

    def create_scatter_plot(x_col, y_col, corr_val, correlation_type):
        plot_data = analysis_df[[x_col, y_col]].dropna()
        if len(plot_data) < 2:
            st.warning(f"Недостаточно данных для {x_col} vs {y_col}")
            return None
        fig = px.scatter(plot_data, x=x_col, y=y_col,
                         title=f"{correlation_type}: {x_col} vs {y_col} (r={corr_val:.2f})", trendline="ols")
        fig = apply_plot_style(fig)
        return fig

    if strong_pos:
        st.markdown(f"### ✅ Сильные положительные связи (r ≥ {pos_threshold})")
        for x_col, y_col, corr_val in strong_pos:
            st.write(f"**{x_col}** ↔ **{y_col}** (r = {corr_val:.3f})")
            fig = create_scatter_plot(x_col, y_col, corr_val, "Положительная связь")
            if fig: st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
    else:
        st.info(f"Нет сильных положительных связей (r ≥ {pos_threshold})")

    if strong_neg:
        st.markdown(f"### 🔻 Сильные отрицательные связи (r ≤ {neg_threshold})")
        for x_col, y_col, corr_val in strong_neg:
            st.write(f"**{x_col}** ↔ **{y_col}** (r = {corr_val:.3f})")
            fig = create_scatter_plot(x_col, y_col, corr_val, "Отрицательная связь")
            if fig: st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
    else:
        st.info(f"Нет сильных отрицательных связей (r ≤ {neg_threshold})")


# ==================== Кластеризация ====================
def show_clustering(df):
    st.subheader("🌀 Кластеризация (KMeans)")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) < 2: st.warning("Нужно минимум 2 числовых столбца"); return
    selected_cols = st.multiselect("Выберите параметры для кластеризации:", numeric_cols, default=numeric_cols[:2])
    if len(selected_cols) < 2: st.warning("Выберите хотя бы 2 столбца"); return
    X = df[selected_cols].dropna()
    if len(X) < 2: st.warning("Недостаточно данных"); return
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if 'n_clusters' not in st.session_state: st.session_state['n_clusters'] = 3
    if st.button("Определить оптимальное количество кластеров"):
        max_k = min(10, len(X) - 1)
        if max_k < 2: st.warning("Недостаточно данных для определения k"); return
        silhouette_scores = [
            silhouette_score(X_scaled, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)) for k in
            range(2, max_k + 1)]
        optimal_k = range(2, max_k + 1)[np.argmax(silhouette_scores)]
        st.session_state['n_clusters'] = optimal_k
        st.success(f"Оптимальное k: {optimal_k}")

    n_clusters = st.slider("Количество кластеров (k):", min_value=2, max_value=10, value=st.session_state['n_clusters'])
    st.session_state['n_clusters'] = n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_result = X.copy()
    df_result["Cluster"] = clusters

    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=selected_cols)
    centroids["Cluster"] = range(n_clusters)
    st.write("### Центроиды кластеров")
    st.dataframe(centroids)

    if len(selected_cols) == 2:
        fig = px.scatter(df_result, x=selected_cols[0], y=selected_cols[1], color="Cluster",
                         color_continuous_scale="Viridis", title="Кластеры")
    else:
        fig = px.scatter_matrix(df_result, dimensions=selected_cols, color="Cluster", title="Кластеры (scatter matrix)")
    fig = apply_plot_style(fig)
    st.plotly_chart(fig, use_container_width=True)


# ==================== Статистический анализ ====================
def _t_test(df, col_group, col_value):
    g1, g2 = df[col_group].dropna().unique()
    s1, s2 = df[df[col_group] == g1][col_value].dropna(), df[df[col_group] == g2][col_value].dropna()
    stat, pval = stats.ttest_ind(s1, s2, equal_var=False)
    return {"statistic": stat, "pvalue": pval, "group_names": (g1, g2), "n1": len(s1), "n2": len(s2)}


def _chi2(df, col1, col2):
    contingency = pd.crosstab(df[col1], df[col2])
    chi2, pval, dof, expected = stats.chi2_contingency(contingency)
    return {"chi2": chi2, "pvalue": pval, "dof": dof, "expected": expected, "contingency": contingency}


def _anova(df, col_group, col_value):
    groups = [df[df[col_group] == g][col_value].dropna() for g in df[col_group].dropna().unique()]
    stat, pval = stats.f_oneway(*groups)
    return {"F": stat, "pvalue": pval}


def show_stats(df):
    st.subheader("📊 Статистический анализ (t-test, χ², ANOVA)")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    int_like = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() <= 10]
    for c in int_like:
        if c not in cat_cols: cat_cols.append(c)

    test = st.selectbox("Выберите тест:", ["t-test", "χ² (хи-квадрат)", "ANOVA"])
    if test == "t-test":
        group_candidates = [c for c in df.columns if df[c].nunique() == 2]
        if not group_candidates or not num_cols: st.warning("Недостаточно данных для t-test"); return
        col_group = st.selectbox("Колонка с группами:", group_candidates)
        col_value = st.selectbox("Числовая колонка:", num_cols)
        if st.button("Выполнить t-test"):
            res = _t_test(df, col_group, col_value)
            st.write(f"t={res['statistic']:.4f}, p={res['pvalue']:.4f}")
            fig = px.box(df, x=col_group, y=col_value, points="all", title=f"{col_value} по {col_group}")
            fig = apply_plot_style(fig)
            st.plotly_chart(fig, use_container_width=True)
    elif test == "χ² (хи-квадрат)":
        if len(cat_cols) < 2: st.warning("Недостаточно категориальных колонок"); return
        col1 = st.selectbox("Категориальная колонка 1:", cat_cols)
        col2 = st.selectbox("Категориальная колонка 2:", [c for c in cat_cols if c != col1])
        if st.button("Выполнить χ²"):
            res = _chi2(df, col1, col2)
            st.write(f"χ²={res['chi2']:.4f}, p={res['pvalue']:.4f}, dof={res['dof']}")
            st.dataframe(res["contingency"])
    elif test == "ANOVA":
        group_candidates = [c for c in df.columns if df[c].nunique() >= 2]
        if not group_candidates or not num_cols: st.warning("Недостаточно данных для ANOVA"); return
        col_group = st.selectbox("Группирующая колонка:", group_candidates)
        col_value = st.selectbox("Числовая колонка:", num_cols)
        if st.button("Выполнить ANOVA"):
            res = _anova(df, col_group, col_value)
            st.write(f"F={res['F']:.4f}, p={res['pvalue']:.4f}")
            fig = px.box(df, x=col_group, y=col_value, points="all", title=f"{col_value} по {col_group}")
            fig = apply_plot_style(fig)
            st.plotly_chart(fig, use_container_width=True)


# ==================== SQL ====================
def show_sql(df):
    st.subheader("💻 SQL-запросы к данным")
    st.markdown("""
        Здесь вы можете выполнять SQL-запросы к загруженной таблице.
        - Таблица называется `data`
        - Используется SQLite диалект
        - Нажмите **Ctrl+Enter** или кнопку **APPLY** для выполнения запроса
    """)
    conn = sqlite3.connect(":memory:")
    df.to_sql("data", conn, index=False, if_exists="replace")
    sql_query = st_ace(value="SELECT * FROM data LIMIT 10;", language="sql", theme="chrome", height=250)
    if sql_query.strip():
        try:
            res = pd.read_sql_query(sql_query, conn)
            st.dataframe(res)
        except Exception as e:
            st.error(f"Ошибка SQL: {e}")


# ==================== Основная логика ====================
st.title("🔬 Продвинутый анализ")
if 'df' in st.session_state:
    df = st.session_state['df']
    analysis_type = st.sidebar.radio(
        "Выберите тип анализа:",
        ["Корреляции 📉", "Кластеризация 🆎", "Статистический анализ 🧮", "SQL-запросы 🛠"]
    )
    if analysis_type == "Корреляции 📉":
        show_correlations(df)
    elif analysis_type == "Кластеризация 🆎":
        show_clustering(df)
    elif analysis_type == "Статистический анализ 🧮":
        show_stats(df)
    elif analysis_type == "SQL-запросы 🛠":
        show_sql(df)
else:
    st.info("Загрузите данные на главной странице.")
