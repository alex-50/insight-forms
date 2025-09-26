import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd


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

    n_clusters = st.slider("Количество кластеров (k):", 2, 10, 3)

    X = df[selected_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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
