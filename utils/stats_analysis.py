import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px


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

    # Подготовка списков столбцов
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # категориальные — все нечисловые + целые с небольшим числом уникальных значений
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    int_like = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() <= 10]
    for c in int_like:
        if c not in cat_cols:
            cat_cols.append(c)

    test = st.selectbox("Выберите тест:", ["t-test", "χ² (хи-квадрат)", "ANOVA"])

    if test == "t-test":
        # показать только колонки с ровно 2 уникальными значениями для групп
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
                # boxplot
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
        # группирующая колонка — любая с >=2 уникумов
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
