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

    for gid, group in df.groupby(gid_col, dropna=False, sort=False):
        row = {gid_col: gid}

        for column in df.columns:
            if column == gid_col:
                continue
            row[column] = first_real_value(group[column])

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=df.columns).set_index(gid_col)

    return pd.DataFrame(rows).set_index(gid_col)


def restore_gemini_id_pattern(df, id_col, gid_col, order_col):
    out = df.copy()
    temp = out[[gid_col, order_col]].copy()
    temp[order_col] = pd.to_numeric(temp[order_col], errors="coerce")

    first_indexes = (
        temp.sort_values([gid_col, order_col], kind="mergesort")
        .groupby(gid_col, dropna=False, sort=False)
        .head(1)
        .index
    )

    out.loc[~out.index.isin(first_indexes), id_col] = np.nan
    return out


def strtree_candidates_as_indices(tree, query_geometry, geometries):
    result = tree.query(query_geometry)

    # Shapely 2 normally returns integer geometry indexes.
    if isinstance(result, np.ndarray) and np.issubdtype(result.dtype, np.integer):
        return [int(index) for index in result.tolist()]

    if isinstance(result, (list, tuple)):
        if not result:
            return []
        if isinstance(result[0], (int, np.integer)):
            return [int(index) for index in result]

    # Compatibility fallback for Shapely versions returning geometries.
    geometry_ids = {id(geometry): index for index, geometry in enumerate(geometries)}
    return [geometry_ids[id(geometry)] for geometry in result if id(geometry) in geometry_ids]


def count_matching_pairs(measured_attrs, theoretical_attrs, rules, ignore_missing_in_measured=True):
    count = 0

    for rule in rules:
        measured_column = rule["meas_col"]
        theoretical_column = rule["theo_col"]
        mode = rule.get("mode", "trim_upper")

        measured_value = normalize_value(measured_attrs.get(measured_column), mode)
        theoretical_value = normalize_value(theoretical_attrs.get(theoretical_column), mode)

        if ignore_missing_in_measured and measured_value is None:
            continue

        if measured_value is not None and theoretical_value is not None:
            if measured_value == theoretical_value:
                count += 1

    return count


def snitt_hits(measured_line, theoretical_line, max_dist, n_points):
    if n_points <= 0:
        return 0

    length = float(measured_line.length)
    if length <= 0:
        return 0

    hits = 0
    for index in range(n_points):
        fraction = 0.0 if n_points == 1 else index / (n_points - 1)
        point = measured_line.interpolate(fraction * length)
        if point.distance(theoretical_line) <= max_dist:
            hits += 1

    return hits


def detect_base_columns(theo_df, meas_df):
    id_col = detect_col(theo_df, ["Id", "ID", "id"])
    order_col = detect_col(theo_df, ["Nr.", "Nr", "NR", "nr", "PunktNr", "punktnr"])
    x_col = detect_col(theo_df, ["Øst", "Ost", "X", "E", "East"])
    y_col = detect_col(theo_df, ["Nord", "Y", "N", "North"])

    id_col_m = detect_col(meas_df, ["Id", "ID", "id"])
    order_col_m = detect_col(meas_df, ["Nr.", "Nr", "NR", "nr", "PunktNr", "punktnr"])
    x_col_m = detect_col(meas_df, ["Øst", "Ost", "X", "E", "East"])
    y_col_m = detect_col(meas_df, ["Nord", "Y", "N", "North"])

    return id_col, order_col, x_col, y_col, id_col_m, order_col_m, x_col_m, y_col_m


# -----------------------------
# UI
# -----------------------------
st.title("VA-linje matcher: teoretisk → innmålt")

col_a, col_b = st.columns(2)
with col_a:
    theo_file = st.file_uploader("Teoretisk datasett (Excel)", type=["xlsx"], key="theo")
with col_b:
    meas_file = st.file_uploader("Innmålt datasett (Excel)", type=["xlsx"], key="meas")

with st.expander("Match-regler", expanded=True):
    max_dist = st.number_input("Bufferavstand (meter) (2D)", min_value=0.0, value=1.0, step=0.1)
    use_snitt = st.checkbox("Bruk snitt-kontroll langs linja", value=True)
    snitt_n = st.number_input("Antall snittpunkter", min_value=1, value=7, step=1, disabled=not use_snitt)
    snitt_min_hits = st.number_input(
        "Minimum treff (snittpunkter innenfor buffer)",
        min_value=1,
        value=4,
        step=1,
        disabled=not use_snitt,
    )
    min_attr_matches = st.slider(
        "Min. antall matchende attributter",
        min_value=0,
        max_value=10,
        value=1,
        step=1,
    )
    ignore_missing = st.checkbox(
        "Ignorer match-felt som er tomme i innmålt (wildcard)",
        value=True,
    )
    output_id_first_only = st.checkbox(
        "Skriv ID kun på første punkt per linje i output (Gemini-format)",
        value=True,
    )

with st.expander("Overføring", expanded=True):
    st.write("Innmålt geometri/punktdata blir aldri endret.")
    protected_text = st.text_input(
        "Ekstra beskyttede kolonner (kommaseparert)",
        value="",
    )
    extra_protected = [column.strip() for column in protected_text.split(",") if column.strip()]

config_ready = False
match_rules = []
transfer_cols = []
theo_df = None
meas_df = None

if theo_file is not None and meas_file is not None:
    theo_df = pd.read_excel(theo_file)
    meas_df = pd.read_excel(meas_file)

    columns = detect_base_columns(theo_df, meas_df)
    id_col, order_col, x_col, y_col, id_col_m, order_col_m, x_col_m, y_col_m = columns

    if any(column is None for column in columns):
        st.error("Fant ikke obligatoriske kolonner (Id/Nr/Øst/Nord) i en av filene.")
        st.stop()

    st.write(
        "Oppdaget kolonner:",
        {
            "teoretisk": {"Id": id_col, "Nr": order_col, "X": x_col, "Y": y_col},
            "innmålt": {"Id": id_col_m, "Nr": order_col_m, "X": x_col_m, "Y": y_col_m},
        },
    )

    with st.expander("Vis kolonner i filene (debug)", expanded=False):
        st.write("Teoretisk kolonner:", list(theo_df.columns))
        st.write("Innmålt kolonner:", list(meas_df.columns))

    st.subheader("1) Velg match-attributter")
    st.caption("Velg hvilke kolonner som skal sammenlignes.")

    meas_cols = list(meas_df.columns)
    theo_cols = list(theo_df.columns)

    default_rules = pd.DataFrame(
        [{"meas_col": meas_cols[0], "theo_col": theo_cols[0], "mode": "trim_upper"}]
    )

    preferred_names = [
        "VEAS_VA.Type og Dimensjon",
        "VEAS_Felles.Dimensjon",
        "VEAS_VA.Dimensjon (mm)",
        "Dimensjon",
        "Type",
    ]

    for candidate in preferred_names:
        if candidate in meas_cols:
            default_rules.loc[0, "meas_col"] = candidate
            break

    for candidate in preferred_names:
        if candidate in theo_cols:
            default_rules.loc[0, "theo_col"] = candidate
            break

    rules_df = st.data_editor(
        default_rules,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "meas_col": st.column_config.SelectboxColumn(
                "Innmålt kolonne",
                options=meas_cols,
                required=True,
            ),
            "theo_col": st.column_config.SelectboxColumn(
                "Teoretisk kolonne",
                options=theo_cols,
                required=True,
            ),
            "mode": st.column_config.SelectboxColumn(
                "Normalisering",
                options=["trim_upper", "digits_only", "raw"],
                required=True,
            ),
        },
        key="rules_editor_v11",
    )

    match_rules = rules_df.dropna(subset=["meas_col", "theo_col"]).to_dict(orient="records")

    st.subheader("2) Velg hvilke attributter som skal overføres")

    preview_theo = forward_fill(theo_df.copy(), id_col)
    preview_theo["__gid__"] = preview_theo[id_col]
    preview_rep = group_rep(preview_theo, "__gid__")

    protected_cols = {
        "Id",
        "Nr.",
        "Øst",
        "Nord",
        "Høyde",
        "Profilnr",
        "Lengde",
        "Lengde 3D",
        "Z",
        "Kote",
        "NN2000",
        "Høyde (m)",
        "Kote (m)",
    }.union(extra_protected)

    exclude = {
        "__gid__",
        id_col,
        order_col,
        x_col,
        y_col,
        id_col_m,
        order_col_m,
        x_col_m,
        y_col_m,
    }.union(protected_cols)

    candidate_transfer_cols = sorted(
        column for column in preview_rep.columns if column not in exclude
    )

    transfer_mode = st.radio(
        "Overføring",
        options=["Alle", "Utvalg"],
        horizontal=True,
        key="transfer_mode_v11",
    )

    if transfer_mode == "Utvalg":
        transfer_cols = st.multiselect(
            "Velg kolonner fra teoretisk som skal kopieres til innmålt",
            options=candidate_transfer_cols,
            default=[
                column
                for column in candidate_transfer_cols
                if column.lower().startswith("veas")
            ][:20],
            key="transfer_cols_v11",
        )
    else:
        transfer_cols = candidate_transfer_cols

    st.caption(f"Antall kolonner som overføres: {len(transfer_cols)}")

    config_ready = len(match_rules) > 0 and len(transfer_cols) > 0
else:
    st.info("Last opp begge Excel-filene for å få valg av match-attributter og overføringskolonner.")


run_btn = st.button("Kjør matching", type="primary", disabled=not config_ready)

if run_btn:
    theo_df = pd.read_excel(theo_file)
    meas_df = pd.read_excel(meas_file)

    columns = detect_base_columns(theo_df, meas_df)
    id_col, order_col, x_col, y_col, id_col_m, order_col_m, x_col_m, y_col_m = columns

    theo_df = forward_fill(theo_df, id_col)
    meas_df = forward_fill(meas_df, id_col_m)

    gid_col = "__gid__"
    theo_df[gid_col] = theo_df[id_col]
    meas_df[gid_col] = meas_df[id_col_m]

    theo_lines, _ = build_lines(theo_df, gid_col, order_col, x_col, y_col)
    meas_lines, meas_group_rows = build_lines(
        meas_df,
        gid_col,
        order_col_m,
        x_col_m,
        y_col_m,
    )

    st.info(
        f"Bygget {len(theo_lines)} teoretiske linjer og "
        f"{len(meas_lines)} innmålte linjer."
    )

    if not theo_lines or not meas_lines:
        st.error("Ingen linjer kunne bygges. Sjekk at hver linje har minst 2 punkt med X/Y.")
        st.stop()

    theo_rep = group_rep(theo_df, gid_col)
    meas_rep = group_rep(meas_df, gid_col)

    theoretical_geometries = [geometry for _, geometry, _ in theo_lines]
    theoretical_ids = [gid for gid, _, _ in theo_lines]
    tree = STRtree(theoretical_geometries)

    meas_out = meas_df.copy()
    meas_out = meas_df.copy(deep=True)
    original_measured = meas_df.copy(deep=True)

    # Pandas 2.2+ tillater ikke alltid tekst/tall/dato i en eksisterende
    # numerisk kolonne. Gjør overføringskolonnene fleksible før skriving.
    for column in transfer_cols:
        if column not in meas_out.columns:
            meas_out[column] = pd.Series(
                np.nan,
                index=meas_out.index,
                dtype="object",
            )
        else:
            meas_out[column] = meas_out[column].astype("object")

    match_rows = []

    for measured_gid, measured_geometry, _ in meas_lines:
        measured_attrs = (
            meas_rep.loc[measured_gid].to_dict()
            if measured_gid in meas_rep.index
            else {}
        )

        candidates = strtree_candidates_as_indices(
            tree,
            measured_geometry.buffer(max_dist),
            theoretical_geometries,
        )

        best = None

        for candidate_index in candidates:
            theoretical_gid = theoretical_ids[candidate_index]
            theoretical_geometry = theoretical_geometries[candidate_index]

            distance = float(measured_geometry.distance(theoretical_geometry))
            if distance > max_dist:
                continue

            theoretical_attrs = (
                theo_rep.loc[theoretical_gid].to_dict()
                if theoretical_gid in theo_rep.index
                else {}
            )

            attribute_matches = count_matching_pairs(
                measured_attrs,
                theoretical_attrs,
                match_rules,
                ignore_missing_in_measured=ignore_missing,
            )

            if attribute_matches < min_attr_matches:
                continue

            hits = None
            if use_snitt:
                hits = snitt_hits(
                    measured_geometry,
                    theoretical_geometry,
                    max_dist,
                    int(snitt_n),
                )
                if hits < int(snitt_min_hits):
                    continue

            score = (
                distance,
                -attribute_matches,
                -(hits if hits is not None else 0),
            )

            if best is None or score < best["score"]:
                best = {
                    "theoretical_gid": theoretical_gid,
                    "distance": distance,
                    "attribute_matches": attribute_matches,
                    "hits": hits,
                    "score": score,
                }

        if best is None:
            match_rows.append(
                {
                    "meas_id": measured_gid,
                    "status": "no_match",
                    "theo_id": None,
                    "attr_matches": 0,
                    "line_dist": None,
                    "snitt_hits": None if not use_snitt else 0,
                }
            )
            continue

        theoretical_gid = best["theoretical_gid"]
        theoretical_row = theo_rep.loc[theoretical_gid].to_dict()
        measured_indexes = meas_group_rows.get(measured_gid, [])

        # Viktig: denne løkken inkluderer både datatypehåndtering og selve
        # skrivingen. Da blir alle valgte kolonner overført, ikke bare den siste.
        # Legg teoretiske verdier inn i samme kolonner.
        # Eksisterende innmålte verdier skal beholdes; bare tomme celler fylles.
        for column in transfer_cols:
            value = theoretical_row.get(column, np.nan)
            meas_out.loc[measured_indexes, column] = value

            # En tom teoretisk verdi skal aldri overskrive eller fylle noe.
            if value is None or pd.isna(value):
                continue
            if isinstance(value, str) and not value.strip():
                continue

            current_values = meas_out.loc[measured_indexes, column]

            blank = current_values.map(
                lambda current: (
                    current is None
                    or pd.isna(current)
                    or (isinstance(current, str) and not current.strip())
                )
            )

            meas_out.loc[measured_indexes, column] = current_values.mask(
                blank,
                value,
            )

        match_rows.append(
            {
                "meas_id": measured_gid,
                "status": "matched",
                "theo_id": theoretical_gid,
                "attr_matches": int(best["attribute_matches"]),
                "line_dist": float(best["distance"]),
                "snitt_hits": None if not use_snitt else int(best["hits"]),
            }
        )

    # Sikkerhetsnett: ingen opprinnelige, ikke-tomme innmålte verdier
    # skal kunne bli overskrevet av overføringen.
    for column in original_measured.columns:
        original_values = original_measured[column]
        keep_original = original_values.map(
            lambda value: not (
                value is None
                or pd.isna(value)
                or (isinstance(value, str) and not value.strip())
            )
        )
        meas_out.loc[keep_original, column] = original_values.loc[keep_original]

    match_report = pd.DataFrame(match_rows)

    if output_id_first_only:
        meas_out = restore_gemini_id_pattern(
            meas_out,
            id_col=id_col_m,
            gid_col=gid_col,
            order_col=order_col_m,
        )

    meas_out = meas_out.drop(columns=[gid_col], errors="ignore")

    st.subheader("Resultat")
    c1, c2, c3 = st.columns(3)
    c1.metric("Innmålte linjer", int(len(meas_lines)))
    c2.metric("Matchet", int((match_report["status"] == "matched").sum()))
    c3.metric("Ikke matchet", int((match_report["status"] != "matched").sum()))

    st.dataframe(match_report, use_container_width=True)

    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
        meas_out.to_excel(writer, index=False, sheet_name="innmalt_med_attrib")
        match_report.to_excel(writer, index=False, sheet_name="match_report")

    st.download_button(
        "Last ned resultat (Excel)",
        data=output_buffer.getvalue(),
        file_name="innmalt_med_teoretiske_attrib.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
