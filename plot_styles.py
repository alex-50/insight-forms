"""
Единые стили графиков Plotly для всего приложения.

Разделение по типам графиков нужно потому, что у разных трейсов
разные наборы допустимых свойств. Например, heatmap не поддерживает
`textposition`, а bar/pie — поддерживают.
"""
from plotly.graph_objects import Figure


def apply_plot_style(fig: Figure) -> Figure:
    """
    Базовый стиль layout: шрифты, цвета, отступы.

    Применяется ко всем графикам без исключения.
    Не трогает свойства трейсов (безопасно для heatmap, scatter, box).

    Args:
        fig: объект Figure из Plotly.

    Returns:
        Тот же fig с обновлённым layout.
    """
    fig.update_layout(
        font=dict(size=16, family="Consolas", color="#404040"),
        title_font=dict(size=22, family="Consolas", color="#404040"),
        legend=dict(font=dict(size=14, family="Consolas", color="#404040")),
        xaxis=dict(
            title_font=dict(size=18, family="Consolas", color="#404040"),
            tickfont=dict(size=14, family="Consolas", color="#404040"),
        ),
        yaxis=dict(
            title_font=dict(size=18, family="Consolas", color="#404040"),
            tickfont=dict(size=14, family="Consolas", color="#404040"),
        ),
        bargap=0.15,
        plot_bgcolor="rgba(255,255,255,1)",
        paper_bgcolor="rgba(255,255,255,0)",
    )
    return fig


def apply_bar_style(fig: Figure) -> Figure:
    """
    Стиль для bar/histogram: базовый layout + подписи значений.

    Args:
        fig: объект Figure (px.bar / px.histogram).

    Returns:
        Тот же fig со стилем.
    """
    fig = apply_plot_style(fig)
    fig.update_traces(
        textfont=dict(size=16, color="whitesmoke", family="Consolas"),
        textposition="auto",
        marker_line_width=1.2,
    )
    return fig


def apply_pie_style(fig: Figure) -> Figure:
    """
    Стиль для круговых диаграмм: проценты + метки + значения.

    Args:
        fig: объект Figure (px.pie).

    Returns:
        Тот же fig со стилем.
    """
    fig = apply_plot_style(fig)
    fig.update_traces(
        textinfo="percent+label+value",
        textfont=dict(size=14, color="#404040", family="Consolas"),
        marker=dict(line=dict(width=1)),
    )
    return fig