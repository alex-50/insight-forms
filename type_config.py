"""
Модуль конфигурации столбцов: автоопределение типов, переименование, псевдонимы.

Содержит:
- detect_type: автоопределение типа столбца по dtype и значениям.
- init_column_types: синхронизация st.session_state с актуальными столбцами df.
- show_type_config: UI для переименования столбцов и выбора типа.
"""
import streamlit as st
import pandas as pd


def detect_type(series: pd.Series) -> str:
    """
    Определяет тип параметра по столбцу.

    Логика:
    - numeric + integer + <=10 уникальных → "Категориальный"
    - numeric (иначе) → "Количественный"
    - object + (<=10 уникальных ИЛИ средняя длина < 20) → "Категориальный"
    - object (иначе) → "Текстовый"
    - всё остальное (datetime, bool, category и т.п.) → "Игнорировать"

    Args:
        series: столбец pandas для анализа.

    Returns:
        Один из: "Количественный", "Категориальный", "Текстовый", "Игнорировать".
    """
    dtype = series.dtype
    if pd.api.types.is_numeric_dtype(dtype):
        if pd.api.types.is_integer_dtype(dtype) and series.nunique() <= 10:
            return "Категориальный"
        return "Количественный"
    if pd.api.types.is_object_dtype(dtype):
        if series.nunique() <= 10 or series.dropna().apply(lambda x: len(str(x))).mean() < 20:
            return "Категориальный"
        return "Текстовый"
    return "Игнорировать"


def init_column_types(df: pd.DataFrame) -> None:
    """
    Синхронизирует st.session_state.column_types и column_aliases с df.

    - Удаляет ключи, которых больше нет в df.columns.
    - Добавляет отсутствующие столбцы с автоопределённым типом и псевдонимом = имени.

    Args:
        df: актуальный DataFrame из st.session_state['df'].
    """
    if 'column_types' not in st.session_state:
        st.session_state.column_types = {}
    if 'column_aliases' not in st.session_state:
        st.session_state.column_aliases = {}

    cols = list(df.columns)
    for k in list(st.session_state.column_types.keys()):
        if k not in cols:
            del st.session_state.column_types[k]
    for k in list(st.session_state.column_aliases.keys()):
        if k not in cols:
            del st.session_state.column_aliases[k]
    for col in cols:
        if col not in st.session_state.column_types:
            st.session_state.column_types[col] = detect_type(df[col])
        if col not in st.session_state.column_aliases:
            st.session_state.column_aliases[col] = col


def show_type_config(df: pd.DataFrame) -> None:
    """
    Отображает UI настройки типов и переименования столбцов.

    Что делает:
    - Показывает text_input для нового имени и selectbox для типа каждого столбца.
    - По кнопке «Применить» валидирует (дубли/пустые имена) и применяет:
      * переименование df,
      * пересборку column_types и column_aliases,
      * очистку виджет-ключей, привязанных к столбцам.
    - Отображает итоговую таблицу типов.

    Args:
        df: DataFrame из st.session_state['df'].
    """
    init_column_types(df)

    st.write("### 🏷 Переименование столбцов и выбор типа")
    st.caption("Введите новые имена, при необходимости поменяйте тип и нажмите «Применить».")

    type_options = ["Количественный", "Категориальный", "Текстовый", "Игнорировать"]

    h1, h2, h3 = st.columns([3, 2, 1])
    h1.markdown("**Текущее имя → Новое имя**")
    h2.markdown("**Тип**")

    new_names, new_types = {}, {}

    for col in df.columns:
        c1, c2, c3 = st.columns([3, 2, 1])
        with c1:
            alias = st.text_input(
                label=f"Новое имя для `{col}`",
                value=col,
                key=f"alias_{col}",
                label_visibility="collapsed",
            )
        with c2:
            cur = st.session_state.column_types.get(col, "Количественный")
            t = st.selectbox(
                label=f"Тип для `{col}`",
                options=type_options,
                index=type_options.index(cur) if cur in type_options else 0,
                key=f"type_select_{col}",
                label_visibility="collapsed",
            )

        new_names[col] = alias
        new_types[col] = t

    st.divider()

    if st.button("✅ Применить изменения", type="primary"):
        values = list(new_names.values())
        errors = []

        if len(set(values)) != len(values):
            dups = sorted({v for v in values if values.count(v) > 1})
            errors.append(f"❌ Дубли имён: {', '.join(dups)}")
        if any(not v.strip() for v in values):
            errors.append("❌ Есть пустые имена")

        if errors:
            for e in errors:
                st.error(e)
        else:
            renamed_df = df.rename(columns=new_names)
            st.session_state['df'] = renamed_df

            st.session_state.column_types = {
                new_names[old]: new_types[old] for old in new_names
            }
            st.session_state.column_aliases = {
                new_names[old]: new_names[old] for old in new_names
            }

            for k in list(st.session_state.keys()):
                if k.startswith(("alias_", "type_select_", "group_col_", "chart_type_", "sel_")):
                    del st.session_state[k]

            st.success("✅ Изменения сохранены")

    st.write("### Итоговая конфигурация")
    df_cur = st.session_state.get('df', df)
    st.dataframe(pd.DataFrame([
        {'Параметр': c, 'Тип': st.session_state.column_types.get(c, '—')}
        for c in df_cur.columns
    ]))


st.title("⚙️ Настройка типов данных и псевдонимов")

if 'df' in st.session_state:
    show_type_config(st.session_state['df'])
else:
    st.info("Загрузите данные на главной странице.")
