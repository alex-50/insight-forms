import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def show_wordcloud(df):
    st.subheader("☁️ Облако слов")

    text_cols = df.select_dtypes(include=['object']).columns.tolist()

    if not text_cols:
        st.warning("Текстовых столбцов не найдено.")
        return

    selected_col = st.selectbox("Выберите столбец для анализа:", text_cols)

    text = " ".join(str(val) for val in df[selected_col].dropna())
    if not text.strip():
        st.warning("Нет текста для отображения.")
        return

    # Генерация облака
    wordcloud = WordCloud(width=1200, height=800, background_color="white").generate(text)

    # Отрисовка
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig, use_container_width=False)
