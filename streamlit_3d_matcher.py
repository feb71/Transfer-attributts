import io
import re

import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import LineString
from shapely.strtree import STRtree


st.set_page_config(page_title="VA-linje 3D-matcher", layout="wide")


def detect_col(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def normalize_value(value, mode="trim_upper"):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if mode == "raw":
        return str(value)
    if mode == "digits_only":
        digits = re.findall(r"\d+", text)
        return "".join(digits) if digits else None
    return text.upper()


def first_value(series):
    for value in series.dropna():
        if not isinstance(value, str) or value.strip():
            return value
    return np.nan


def group_rep(df, gid_col):
    rows = []
    for gid, group in df.groupby(gid_col, dropna=False, sort=False):
        row = {gid_col: gid}
        for column in df.columns:
            if column != gid_col:
                row[column] = first_value(group[column])
        rows.append(row)
    return pd.DataFrame(rows).set_index(gid_col)


def build_line_data(df, gid_col, order_col, x_col, y_col, z_col):
    data = df.copy()
    data[order_col] = pd.to_numeric(data[order_col], errors="coerce")
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data[z_col] = pd.to_numeric(data[z_col], errors="coerce")
    data = data.sort_values([gid_col, order_col], kind="mergesort")

    groups = {}
    lines = []

    for gid, group in data.groupby(gid_col, dropna=False, sort=False):
        indexes = group.index.tolist()
        points = group[[x_col, y_col, z_col]].dropna().to_numpy(dtype=float)

        if len(points) < 2:
            continue

        xy = points[:, :2]
        xy_line = LineString(xy.tolist())

        # Fjern eventuelle like nabopunkter som gir null segmentlengde.
        keep = [0]
        for index in range(1, len(points)):
            if np.linalg.norm(points[index, :2] - points[keep[-1], :2]) > 1e-9:
                keep.append(index)
        points = points[keep]

        if len(points) < 2:
            continue

        xy_line = LineString(points[:, :2].tolist())
        lines.append({"gid": gid, "points": points, "xy": xy_line, "indexes": indexes})
        groups[gid] = indexes

    return lines, groups


def interpolate_3d(points, fractions):
    """Interpolate X/Y/Z at normalized positions along a 3D point sequence."""
    xy = points[:, :2]
    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = cumulative[-1]

    if total <= 0:
        return np.repeat(points[:1], len(fractions), axis=0)

    result = []
    for fraction in fractions:
        target = float(np.clip(fraction, 0.0, 1.0)) * total
        segment = int(np.searchsorted(cumulative, target, side="right") - 1)
        segment = max(0, min(segment, len(points) - 2))
        length = cumulative[segment + 1] - cumulative[segment]
        ratio = 0.0 if length <= 0 else (target - cumulative[segment]) / length
        result.append(points[segment] + ratio * (points[segment + 1] - points[segment]))

    return np.asarray(result)


def match_attributes(measured, theoretical, rules, ignore_missing):
    count = 0
    for rule in rules:
        mv = normalize_value(measured.get(rule["meas_col"]), rule.get("mode", "trim_upper"))
        tv = normalize_value(theoretical.get(rule["theo_col"]), rule.get("mode", "trim_upper"))
        if ignore_missing and mv is None:
            continue
        if mv is not None and tv is not None and mv == tv:
            count += 1
    return count


def orient_theoretical(measured_points, theoretical_points):
    normal = interpolate_3d(theoretical_points, [0.0, 1.0])
    measured = interpolate_3d(measured_points, [0.0, 1.0])
    forward = np.linalg.norm(measured[0, :2] - normal[0, :2]) + np.linalg.norm(measured[1, :2] - normal[1, :2])
    reverse = np.linalg.norm(measured[0, :2] - normal[1, :2]) + np.linalg.norm(measured[1, :2] - normal[0, :2])
    return theoretical_points if forward <= reverse else theoretical_points[::-1]


def restore_original_values(output, original):
    for column in original.columns:
        values = original[column]
        keep = values.map(
            lambda value: not (
                value is None
                or pd.isna(value)
                or (isinstance(value, str) and not value.strip())
            )
        )
        output.loc[keep, column] = values.loc[keep]
    return output


st.title("VA-linje 3D-matcher: teoretisk → innmålt")
st.caption("Separat eksperimentell versjon. Den eksisterende 2D-appen endres ikke.")

left, right = st.columns(2)
with left:
    theo_file = st.file_uploader("Teoretisk datasett (Excel)", type=["xlsx"], key="theo3d")
with right:
    meas_file = st.file_uploader("Innmålt datasett (Excel)", type=["xlsx"], key="meas3d")

with st.expander("3D-matchinnstillinger", expanded=True):
    horizontal_limit = st.number_input("Maks horisontal avstand (m)", min_value=0.0, value=1.0, step=0.01)
    vertical_limit = st.number_input("Maks høydeforskjell (m)", min_value=0.0, value=0.5, step=0.05)
    max_3d_limit = st.number_input("Maks samlet 3D-avstand (m)", min_value=0.0, value=1.0, step=0.05)
    sample_count = st.number_input("Antall interpolerte kontrollpunkter per linje", min_value=2, value=11, step=1)
    min_valid_samples = st.slider("Minimum andel godkjente kontrollpunkter", 0.0, 1.0, 0.7, 0.05)
    min_attr_matches = st.slider("Minimum matchende attributter", 0, 10, 1)
    ignore_missing = st.checkbox("Ignorer tomt matchfelt i innmålt", value=True)

config_ready = False
match_rules = []
transfer_cols = []

if theo_file is not None and meas_file is not None:
    theo_df = pd.read_excel(theo_file)
    meas_df = pd.read_excel(meas_file)

    id_col = detect_col(theo_df, ["Id", "ID", "id"])
    order_col = detect_col(theo_df, ["Nr.", "Nr", "NR", "nr", "PunktNr", "punktnr"])
    x_col = detect_col(theo_df, ["Øst", "Ost", "X", "E", "East"])
    y_col = detect_col(theo_df, ["Nord", "Y", "N", "North"])
    z_col = detect_col(theo_df, ["Høyde", "Hoyde", "Z", "Kote"])

    id_col_m = detect_col(meas_df, ["Id", "ID", "id"])
    order_col_m = detect_col(meas_df, ["Nr.", "Nr", "NR", "nr", "PunktNr", "punktnr"])
    x_col_m = detect_col(meas_df, ["Øst", "Ost", "X", "E", "East"])
    y_col_m = detect_col(meas_df, ["Nord", "Y", "N", "North"])
    z_col_m = detect_col(meas_df, ["Høyde", "Hoyde", "Z", "Kote"])

    required = [id_col, order_col, x_col, y_col, z_col, id_col_m, order_col_m, x_col_m, y_col_m, z_col_m]
    if any(column is None for column in required):
        st.error("Fant ikke Id, Nr., Øst, Nord og Høyde i begge filer.")
        st.stop()

    st.subheader("1) Match-attributter")
    rules_default = pd.DataFrame([{"meas_col": meas_df.columns[0], "theo_col": theo_df.columns[0], "mode": "trim_upper"}])
    for candidate in ["S_FCODE", "Dimensjon", "Material", "Nett_type"]:
        if candidate in meas_df.columns and candidate in theo_df.columns:
            rules_default.loc[0, "meas_col"] = candidate
            rules_default.loc[0, "theo_col"] = candidate
            break

    rules_df = st.data_editor(
        rules_default,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "meas_col": st.column_config.SelectboxColumn("Innmålt kolonne", options=list(meas_df.columns), required=True),
            "theo_col": st.column_config.SelectboxColumn("Teoretisk kolonne", options=list(theo_df.columns), required=True),
            "mode": st.column_config.SelectboxColumn("Normalisering", options=["trim_upper", "digits_only", "raw"], required=True),
        },
        key="rules3d",
    )
    match_rules = rules_df.dropna(subset=["meas_col", "theo_col"]).to_dict(orient="records")

    st.subheader("2) Attributter som skal overføres")
    preview = meas_df.columns.tolist()
    excluded = {id_col, order_col, x_col, y_col, z_col, id_col_m, order_col_m, x_col_m, y_col_m, z_col_m}
    transfer_options = sorted([column for column in theo_df.columns if column not in excluded])
    transfer_cols = st.multiselect("Velg teoretiske attributter", transfer_options, default=transfer_options, key="transfer3d")
    config_ready = bool(match_rules) and bool(transfer_cols)
else:
    st.info("Last opp begge Excel-filene.")


if st.button("Kjør 3D-matching", type="primary", disabled=not config_ready):
    theo_df = pd.read_excel(theo_file)
    meas_df = pd.read_excel(meas_file)

    # Bare linje-ID skal fylles ned. Andre attributter skal ikke fylles
    # ned på tvers av punkter, fordi det kan endre de opprinnelige verdiene.
    theo_df[id_col] = theo_df[id_col].ffill()
    meas_df[id_col_m] = meas_df[id_col_m].ffill()

    gid = "__gid__"
    theo_df[gid] = theo_df[id_col]
    meas_df[gid] = meas_df[id_col_m]

    theo_lines, _ = build_line_data(theo_df, gid, order_col, x_col, y_col, z_col)
    meas_lines, measured_groups = build_line_data(meas_df, gid, order_col_m, x_col_m, y_col_m, z_col_m)
    theo_rep = group_rep(theo_df, gid)
    meas_rep = group_rep(meas_df, gid)

    geoms = [line["xy"] for line in theo_lines]
    tree = STRtree(geoms)
    output = meas_df.copy(deep=True)
    original = meas_df.copy(deep=True)

    for column in transfer_cols:
        if column not in output.columns:
            output[column] = pd.Series(np.nan, index=output.index, dtype="object")
        else:
            output[column] = output[column].astype("object")

    rows = []
    fractions = np.linspace(0.0, 1.0, int(sample_count))

    for measured in meas_lines:
        measured_gid = measured["gid"]
        measured_rep = meas_rep.loc[measured_gid].to_dict()
        candidate_ids = tree.query(measured["xy"].buffer(horizontal_limit)).tolist()
        if not candidate_ids:
            candidate_ids = list(range(len(theo_lines)))

        best = None
        attribute_rejections = 0
        geometry_rejections = 0
        best_valid_fraction = 0.0
        best_max_horizontal = None
        best_max_vertical = None
        best_max_3d = None
        measured_samples = interpolate_3d(measured["points"], fractions)

        for candidate_index in candidate_ids:
            theoretical = theo_lines[int(candidate_index)]
            theoretical_rep = theo_rep.loc[theoretical["gid"]].to_dict()
            attrs = match_attributes(measured_rep, theoretical_rep, match_rules, ignore_missing)
            if attrs < min_attr_matches:
                attribute_rejections += 1
                continue

            theoretical_points = orient_theoretical(measured["points"], theoretical["points"])
            theoretical_samples = interpolate_3d(theoretical_points, fractions)
            horizontal = np.linalg.norm(measured_samples[:, :2] - theoretical_samples[:, :2], axis=1)
            vertical = np.abs(measured_samples[:, 2] - theoretical_samples[:, 2])
            distance_3d = np.sqrt(horizontal**2 + vertical**2)
            valid = (horizontal <= horizontal_limit) & (vertical <= vertical_limit) & (distance_3d <= max_3d_limit)
            score = (int(valid.sum()), float(np.nanmax(distance_3d)), float(np.nanmean(distance_3d)))
            valid_fraction = score[0] / len(fractions)
            best_valid_fraction = max(best_valid_fraction, valid_fraction)

            if best_max_horizontal is None or float(np.max(horizontal)) < best_max_horizontal:
                best_max_horizontal = float(np.max(horizontal))
            if best_max_vertical is None or float(np.max(vertical)) < best_max_vertical:
                best_max_vertical = float(np.max(vertical))
            if best_max_3d is None or float(np.max(distance_3d)) < best_max_3d:
                best_max_3d = float(np.max(distance_3d))

            if valid_fraction < min_valid_samples:
                geometry_rejections += 1
                continue
            if best is None or score > best["score"]:
                best = {"line": theoretical, "score": score, "horizontal": horizontal, "vertical": vertical, "distance_3d": distance_3d, "attrs": attrs}

        indexes = measured_groups.get(measured_gid, [])
        if best is None:
            if not candidate_ids:
                reason = "Ingen teoretiske linjer tilgjengelig"
            elif attribute_rejections == len(candidate_ids):
                reason = "Match-attributtene stemmer ikke"
            elif geometry_rejections > 0:
                reason = (
                    "Geometrien overskrider 3D-grensene "
                    f"(beste godkjente andel: {best_valid_fraction:.0%})"
                )
            else:
                reason = "Ingen kandidat oppfylte alle matchkrav"

            rows.append({
                "meas_id": measured_gid,
                "status": "no_match",
                "reason": reason,
                "theo_id": None,
                "best_valid_fraction": best_valid_fraction,
                "best_max_horizontal": best_max_horizontal,
                "best_max_vertical": best_max_vertical,
                "best_max_3d": best_max_3d,
            })
            continue

        theoretical_row = theo_rep.loc[best["line"]["gid"]].to_dict()
        for column in transfer_cols:
            value = theoretical_row.get(column, np.nan)
            if value is None or pd.isna(value) or (isinstance(value, str) and not value.strip()):
                continue
            current = output.loc[indexes, column]
            blank = current.map(lambda item: item is None or pd.isna(item) or (isinstance(item, str) and not item.strip()))
            output.loc[indexes, column] = current.mask(blank, value)

        rows.append({
            "meas_id": measured_gid,
            "status": "matched",
            "reason": "OK",
            "theo_id": best["line"]["gid"],
            "attr_matches": best["attrs"],
            "godkjente_punkter": best["score"][0],
            "maks_horisontal": float(np.max(best["horizontal"])),
            "maks_høydeavvik": float(np.max(best["vertical"])),
            "maks_3d_avstand": float(np.max(best["distance_3d"])),
        })

    output = restore_original_values(output, original)
    output = output.drop(columns=[gid], errors="ignore")
    report = pd.DataFrame(rows)

    st.subheader("Resultat")
    st.metric("Matchet", int((report["status"] == "matched").sum()))
    st.metric("Ikke matchet", int((report["status"] != "matched").sum()))
    st.dataframe(report, use_container_width=True)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        output.to_excel(writer, index=False, sheet_name="innmalt_3d_attrib")
        report.to_excel(writer, index=False, sheet_name="match_report_3d")

    st.download_button(
        "Last ned 3D-resultat (Excel)",
        data=buffer.getvalue(),
        file_name="innmalt_teoretisk_3d_match.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
