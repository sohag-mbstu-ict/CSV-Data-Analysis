"""Create polished charts from CSV query results."""

import re
import time
from pathlib import Path

import pandas as pd


def wants_bar_chart(question):
    """Return whether the question contains an explicit visualization intent."""
    intent = re.compile(
        r"\b(?:"
        r"show|shows|showing|display|displays|displaying|"
        r"visuali[sz](?:e|es|ed|ing|ation|ations)|"
        r"visualisation|visualisation|visual|graph|graphs|graphical|"
        r"chart|charts|plot|plots|plotting|draw|draws|drawing|"
        r"render|renders|rendering|illustrate|illustrates|illustrating|"
        r"depict|depicts|depicting|"
        r"bar\s+chart|line\s+chart|pie\s+chart|scatter\s+plot|"
        r"histogram|heatmap|visual\s+representation|visual\s+breakdown|"
        r"see\s+(?:a\s+)?(?:visual|chart|graph|plot)"
        r")\b",
        re.I,
    )
    return bool(intent.search(question))


# Measures that represent money going out. When a query returns them as negative
# numbers, the chart draws their size as a positive bar and says so in the footer.
OUTFLOW_WORDS = {"loss", "losses", "cost", "costs", "expense", "expenses", "refund", "refunds", "deficit", "spend"}
GAIN_WORDS = {"profit", "profits", "margin", "gain", "gains", "income"}
TIME_WORDS = {"year", "month", "quarter", "week", "day", "date", "period"}
ID_WORDS = {"id", "index", "rank"}

INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"
NEGATIVE = "#C2410C"
OUTFLOW = "#DC6B4F"
GAIN = "#2F9E6E"
PALETTE = ["#2F5FB3", "#7C8AA5", "#D9A23F", "#7A5BA8", "#3A9BB5"]

# Conversational lead-ins and "in a graph" endings that should not appear in a chart title.
FILLER = re.compile(
    r"^\s*(?:(?:please|can you|could you|show me|show|give me|create|make|draw|plot|display|"
    r"visuali[sz]e|graph|chart|a|an|the)\s+)+"
    r"|\s+(?:in|as|with|using|on)\s+(?:a|an|the)?\s*(?:bar\s+|line\s+|pie\s+)?"
    r"(?:graph|chart|plot|visual\w*)\s*$",
    re.I,
)


def _words(value):
    return set(re.findall(r"[a-z0-9]+", str(value).lower()))


def _display_name(column):
    name = re.sub(r"^(total|sum|avg|average|mean|min|max|count)_?", "", str(column), flags=re.I)
    return re.sub(r"[_\s]+", " ", name).strip().title() or str(column)


def _compact(value):
    """Format 6894959 as 6.89M and 1250 as 1.25K, keeping small numbers readable."""
    magnitude = abs(value)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= limit:
            return f"{value / limit:,.2f}".rstrip("0").rstrip(".") + suffix
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _chart_title(question):
    title = question.strip().rstrip("?.! ")
    previous = None
    while previous != title:
        previous, title = title, FILLER.sub("", title).strip()
    title = title or question.strip().rstrip("?.! ")
    return title[0].upper() + title[1:]


def _metric_color(column, index):
    words = _words(column)
    if words & OUTFLOW_WORDS:
        return OUTFLOW
    if words & GAIN_WORDS:
        return GAIN
    return PALETTE[index % len(PALETTE)]


def save_bar_chart(df, question):
    """Render a clean horizontal bar chart from query results and save it under charts/."""
    if df.empty:
        return None, "The query returned no rows to chart."

    numeric_columns = [
        column for column in df.select_dtypes(include="number").columns
        if not pd.api.types.is_bool_dtype(df[column])
    ]
    if not numeric_columns:
        return None, "The query result has no numeric columns to chart."

    question_words = _words(question) - {
        "a", "an", "and", "bar", "chart", "compare", "graph", "in", "me", "of", "plot", "show",
        "the", "total", "visualize", "visualise",
    }
    matching_metrics = [column for column in numeric_columns if question_words & _words(column)]
    metric_columns = matching_metrics or [
        column for column in numeric_columns
        if not _words(column) & (TIME_WORDS | ID_WORDS)
    ][:1] or [numeric_columns[0]]

    dimensions = [column for column in df.columns if column not in metric_columns]
    categorical_dimensions = [
        column for column in dimensions
        if not pd.api.types.is_numeric_dtype(df[column]) or _words(column) & TIME_WORDS
    ]
    category_column = next(
        (column for column in categorical_dimensions if question_words & _words(column)),
        None,
    ) or next(
        (column for column in categorical_dimensions if 1 < df[column].nunique(dropna=True) <= 100),
        None,
    )

    plot_df = df.copy()
    for column in metric_columns:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=metric_columns)
    if plot_df.empty:
        return None, "The selected chart measures contain no numeric values."
    if category_column is None:
        plot_df = plot_df.head(1)

    # A loss of -1.3M is a 1.3M loss: draw its size instead of a bar pointing left.
    notes = []
    for column in metric_columns:
        values = plot_df[column]
        if _words(column) & OUTFLOW_WORDS and (values <= 0).all() and (values < 0).any():
            plot_df[column] = values.abs()
            notes.append(f"{_display_name(column)} shown as a positive amount.")

    max_categories = 18
    if category_column and len(plot_df) > max_categories:
        if _words(category_column) & TIME_WORDS:
            plot_df = plot_df.head(max_categories)
        else:
            plot_df = plot_df.assign(
                _chart_magnitude=plot_df[metric_columns].abs().sum(axis=1)
            ).nlargest(max_categories, "_chart_magnitude").drop(columns="_chart_magnitude")
        notes.append(f"{max_categories} of {len(df)} rows shown.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    if category_column is None:
        # One row of totals: each measure becomes its own labelled bar, so no legend is needed.
        labels = [_display_name(column) for column in metric_columns]
        series = [(
            None,
            [float(plot_df[column].iloc[0]) for column in metric_columns],
            [_metric_color(column, index) for index, column in enumerate(metric_columns)],
        )]
    else:
        labels = plot_df[category_column].fillna("(missing)").astype(str).tolist()
        series = []
        for index, column in enumerate(metric_columns):
            values = plot_df[column].astype(float).tolist()
            base = _metric_color(column, index)
            if len(metric_columns) == 1:
                # Single measure: a genuinely negative category (e.g. a losing region) is highlighted.
                colors = [NEGATIVE if value < 0 else base for value in values]
            else:
                colors = [base] * len(values)
            series.append((_display_name(column), values, colors))

    group_count = len(series)
    bar_count = len(labels) * group_count
    fig_height = max(3.0, min(11.0, 1.9 + 0.42 * bar_count + 0.25 * len(labels)))
    fig, ax = plt.subplots(figsize=(10, fig_height), layout="constrained")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    positions = list(range(len(labels)))
    slot = 0.8 / group_count
    all_values = []
    for index, (name, values, colors) in enumerate(series):
        offset = (index - (group_count - 1) / 2) * slot
        bars = ax.barh(
            [position + offset for position in positions], values,
            height=slot * 0.82, color=colors, label=name, zorder=3,
        )
        ax.bar_label(bars, labels=[_compact(value) for value in values],
                     padding=6, fontsize=9.5, color=INK)
        all_values.extend(values)

    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, pad=8, labelsize=10.5, labelcolor=INK)
    ax.tick_params(axis="x", length=0, labelsize=9, labelcolor=MUTED)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _compact(value)))
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Leave room for the value labels at the end of each bar.
    low, high = min(all_values + [0]), max(all_values + [0])
    span = (high - low) or 1
    ax.set_xlim(low - span * 0.14 if low < 0 else 0, high + span * 0.14 if high > 0 else 0)
    ax.axvline(0, color="#9CA3AF", linewidth=1, zorder=4)

    if category_column is not None:
        ax.set_ylabel(_display_name(category_column), color=MUTED, labelpad=10)
    if len(metric_columns) == 1:
        ax.set_xlabel(_display_name(metric_columns[0]), color=MUTED, labelpad=8)

    fig.suptitle(_chart_title(question), x=0.01, ha="left", fontsize=15,
                 fontweight="bold", color=INK)
    if group_count > 1:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncols=min(group_count, 4),
                  frameon=False, fontsize=9.5, handlelength=1.2, handleheight=1.0,
                  labelcolor=INK, borderaxespad=0)

    footer = "  ·  ".join(["Source: query results"] + notes)
    fig.text(0.01, -0.02, footer, fontsize=8, color=MUTED, ha="left", va="top")

    chart_dir = Path.cwd() / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")[:48] or "bar_chart"
    chart_path = chart_dir / f"{slug}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(chart_path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return chart_path, None
