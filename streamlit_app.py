import io
import re

import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import LineString
from shapely.strtree import STRtree


st.set_page_config(page_title="VA-linje matcher", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def normalize_value(value, mode="trim_upper"):
    if value is None or pd.isna(value):
        return None

    if mode == "raw":
        return str(value)

    text = str(value).strip()

    if mode == "trim_upper":
        return text.upper()

    if mode == "digits_only":
        digits = re.findall(r"\d+", text)
        return "".join(digits) if digits else None

    return text


def detect_col(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def forward_fill(df, column):
    if column in df.columns:
        df[column] = df[column].ffill()
    return df


def build_lines(df, gid_col, order_col, x_col, y_col):
    data = df.copy()
    data[order_col] = pd.to_numeric(data[order_col], errors="coerce")
    data = data.sort_values([gid_col, order_col], kind="mergesort")

    group_rows = {}
    lines = []

    for gid, group in data.groupby(gid_col, dropna=False, sort=False):
        indexes = group.index.tolist()
        group_rows[gid] = indexes

        xs = pd.to_numeric(group[x_col], errors="coerce").to_numpy()
        ys = pd.to_numeric(group[y_col], errors="coerce").to_numpy()
        valid = ~np.isnan(xs) & ~np.isnan(ys)
        xs, ys = xs[valid], ys[valid]

        if len(xs) >= 2:
            geometry = LineString(list(zip(xs.tolist(), ys.tolist())))
            lines.append((gid, geometry, int(len(xs))))

    return lines, group_rows


def first_real_value(series):
    """Return the first non-empty value in a group."""
    values = series.dropna()

    if values.empty:
        return np.nan

    for value in values:
        if isinstance(value, str) and not value.strip():
            continue
        return value

    return np.nan


def group_rep(df, gid_col):
    """Create one representative row per line/group.

    Values are taken from the first non-empty cell in each group. This is
    important because Excel/Gemini exports may store an attribute on a later
    point rather than on the first point of the line.
    """
    rows = []
