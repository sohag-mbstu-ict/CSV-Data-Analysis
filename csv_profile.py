"""CSV inspection and dataset profile generation."""
import re

import pandas as pd


def is_dataset_description_request(question):
    """Identify questions asking for the dataset's schema or overall profile."""
    text = question.lower()
    if re.search(r"\b(schema|metadata|overview|profile)\b", text):
        return True
    asks_to_describe = re.search(
        r"\b(describe|summari[sz]e|summary|explain|tell\s+me\s+about|what\s+is\s+in|"
        r"what\s+does\s+.+\s+contain|what\s+are\s+the\s+|which\s+|list|show)\b",
        text,
    )
    mentions_dataset = re.search(
        r"\b(csv|dataset|data\s+set|data|file|table|columns?|fields?)\b",
        text,
    )
    return bool(asks_to_describe and mentions_dataset)


def build_profile(csv_path):
    """Describe the dataset without sending raw rows to the language model."""
    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        raise ValueError("The CSV file has no data rows.")

    duplicates = int(df.duplicated().sum())
    lines = [
        f"Table `data`: {len(df) - duplicates} rows (exact duplicate rows removed). Use this table by default.",
        f"Table `data_raw`: {len(df)} rows exactly as uploaded ({duplicates} duplicate rows). "
        "Use it only for questions about duplicates.",
        "Columns:",]
    
    missing = {}
    for column in df.columns:
        series = df[column]
        missing_count = int(series.isna().sum())
        missing[column] = missing_count
        info = f'- "{column}": missing {missing_count}, {series.nunique()} distinct'
        if pd.api.types.is_numeric_dtype(series):
            info = f'- "{column}" (number): missing {missing_count}, range {series.min()} to {series.max()}'
        else:
            text = series.dropna().astype(str)
            looks_like_date = (
                len(text)
                and text.head(200).str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.9
            )
            if looks_like_date:
                info = f'- "{column}" (date): missing {missing_count}, from {text.min()} to {text.max()}'
            elif series.nunique() <= 30:
                info += f", values: {sorted(text.unique().tolist())}"
            else:
                info += f", examples: {text.unique()[:3].tolist()}"
        lines.append(info)

    return "\n".join(lines), missing, duplicates
