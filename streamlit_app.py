
import streamlit as st
import pandas as pd
import numpy as np
import re
from shapely.geometry import LineString
from shapely.strtree import STRtree

st.set_page_config(page_title="VA-linje matcher (teoretisk → innmålt)", layout="wide")

# -----------------------------
# Helpers
# -----------------------------

def normalize_value(v, mode="trim_upper"):
    if pd.isna(v):
        return None
    if mode == "raw":
        return str(v)
    s = str(v).strip()
    if mode == "trim_upper":
        return s.upper()
    if mode == "digits_only":
        m = re.findall(r"\d+", s)
        return "".join(m) if m else None
    return s

def forward_fill_group_ids(df, group_col):
    # Gemini-Excel har ofte ID bare på første rad i hver gruppe.
    # Fyll ned slik at alle rader har group_id (kun internt i appen).
    df[group_col] = df[group_col].ffill()
    return df

def build_lines_from_points(df, group_col, order_col, x_col, y_col):
    """
    Returnerer:
      - lines_df: én rad per gruppe med kolonnene [group_col, 'geometry']
      - group_rows: dict[group_id] -> indeksliste i original df
    """
    df = df.copy()
    df = df.sort_values([group_col, order_col], kind="mergesort")

    group_rows = {}
    records = []

    for gid, g in df.groupby(group_col, dropna=False):
        idxs = g.index.tolist()
        group_rows[gid] = idxs

        xs = pd.to_numeric(g[x_col], errors="coerce").to_numpy()
        ys = pd.to_numeric(g[y_col], errors="coerce").to_numpy()
        mask = ~np.isnan(xs) & ~np.isnan(ys)
        xs, ys = xs[mask], ys[mask]

        if len(xs) < 2:
            geom = None
        else:
            coords = list(zip(xs.tolist(), ys.tolist()))
            geom = LineString(coords)

        records.append({group_col: gid, "geometry": geom, "n_points": int(len(xs))})

    lines_df = pd.DataFrame(records)
    lines_df = lines_df[lines_df["geometry"].notna()].reset_index(drop=True)
    return lines_df, group_rows

def group_attributes(df, group_col, prefer_first_row=True):
    """
    Samler attributter per gruppe til én rad per group_id.
    Fyller ned innen gruppe før vi tar representativ rad.
    """
    d = df.copy()
    d = d.groupby(group_col, dropna=False).apply(lambda g: g.ffill()).reset_index(drop=True)
    rep = d.groupby(group_col, dropna=False).head(1).copy() if prefer_first_row else d.groupby(group_col, dropna=False).tail(1).copy()
    return rep.set_index(group_col, dropna=False)

def sample_distances(meas_line, theo_line, n_samples=5):
    if meas_line is None or theo_line is None or meas_line.length == 0:
        return []
    ds = []
    for t in np.linspace(0, 1, n_samples):
        p = meas_line.interpolate(t, normalized=True)
        ds.append(float(p.distance(theo_line)))
    return ds

def matches_required_fields(meas_attrs, theo_attrs, field_rules):
    for rule in field_rules:
        col = rule["name"]
        mode = rule.get("mode", "trim_upper")
        mv = normalize_value(meas_attrs.get(col, None), mode=mode)
        tv = normalize_value(theo_attrs.get(col, None), mode=mode)
        if mv is None or tv is None or mv != tv:
            return False
    return True

def pick_transfer_columns(theo_rep_df, exclude_cols, selection_mode, chosen_cols):
    cols = [c for c in theo_rep_df.columns if c not in exclude_cols]
    if selection_mode == "Alle (unntatt ekskluderte)":
        return cols
    return [c for c in chosen_cols if c in cols]

def transfer_to_measured_rows(meas_df, meas_group_rows, meas_gid, theo_row, transfer_cols, fill_all_rows=True):
    idxs = meas_group_rows.get(meas_gid, [])
    if not idxs:
        return meas_df

    for col in transfer_cols:
        if col not in meas_df.columns:
            meas_df[col] = np.nan

    if fill_all_rows:
        for col in transfer_cols:
            meas_df.loc[idxs, col] = theo_row.get(col, np.nan)
    else:
        meas_df.loc[idxs[0], transfer_cols] = [theo_row.get(c, np.nan) for c in transfer_cols]
    return meas_df

def restore_gemini_id_pattern(df, id_col, gid_col, order_col):
    """
    Gjør Id-kolonnen Gemini-lik: Id kun på første punkt i hver linje,
    blank på resten. Bruker gid_col (intern, alltid fylt) for grupper.
    """
    out = df.copy()
    # Finn "første rad" per gruppe basert på minste order_col (eller første i sortert rekkefølge)
    tmp = out[[gid_col, order_col]].copy()
    tmp[order_col] = pd.to_numeric(tmp[order_col], errors="coerce")
    # sort to get first row index per group
    first_idx = (
        tmp.sort_values([gid_col, order_col], kind="mergesort")
           .groupby(gid_col, dropna=False)
           .head(1)
           .index
    )
    mask_first = out.index.isin(first_idx)
    out.loc[~mask_first, id_col] = np.nan
    return out

# -----------------------------
# UI
# -----------------------------

st.title("VA-linje matcher: teoretisk → innmålt (Excel inn/ut)")

colA, colB = st.columns(2)
with colA:
    theo_file = st.file_uploader("Teoretisk datasett (Excel)", type=["xlsx"], key="theo")
with colB:
    meas_file = st.file_uploader("Innmålt datasett (Excel)", type=["xlsx"], key="meas")

with st.expander("1) Kolonner og import/eksportvalg", expanded=True):
    st.caption("Velg kolonnene som beskriver linjene (ID, rekkefølge, koordinater). Standardene under matcher typisk Gemini-eksport.")
    group_col = st.text_input("Linje-ID kolonne", value="Id")
    order_col = st.text_input("Rekkefølge kolonne", value="Nr.")
    x_col = st.text_input("X (Øst) kolonne", value="Øst")
    y_col = st.text_input("Y (Nord) kolonne", value="Nord")

    id_ffill = st.checkbox("Fyll ned tom ID (Id) automatisk (ffill) for matching", value=True)
    output_id_first_only = st.checkbox("Skriv ID kun på første punkt per linje i output (Gemini-format)", value=True)

with st.expander("2) Match-regler", expanded=True):
    max_dist = st.number_input("Maks planavstand (meter) for kandidat (2D)", min_value=0.0, value=1.0, step=0.1)
    n_samples = st.slider("Antall snittpunkter langs innmålt linje", min_value=3, max_value=15, value=5, step=1)
    min_hits = st.slider("Min. antall snittpunkter som må være ≤ maksavstand", min_value=1, max_value=15, value=3, step=1)

    st.caption("Velg attributter som MÅ matche (f.eks. Type + Dimensjon). Verdier normaliseres før sammenligning.")
    default_rules = [{"name":"VEAS_VA.System","mode":"trim_upper"},{"name":"VEAS_VA.Dimensjon (mm)","mode":"digits_only"}]
    rules_df = st.session_state.get("rules_df", pd.DataFrame(default_rules))
    rules_df = st.data_editor(
        rules_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Kolonnenavn", required=True),
            "mode": st.column_config.SelectboxColumn("Normalisering", options=["trim_upper","digits_only","raw"])
        },
        key="rules_editor"
    )
    field_rules = rules_df.dropna(subset=["name"]).to_dict(orient="records")

with st.expander("3) Hvilke attributter skal overføres?", expanded=True):
    selection_mode = st.radio("Overfør:", ["Alle (unntatt ekskluderte)", "Utvalg"], horizontal=True)
    fill_all_rows = st.checkbox("Fyll verdiene på alle rader i samme linje (anbefalt)", value=True)
    st.caption("Ekskluderes automatisk: punktkolonner (ID, rekkefølge, X/Y) + interne felt. Du kan legge til flere ekskluderinger.")
    extra_exclude = st.text_input("Ekstra kolonner å ekskludere (kommaseparert)", value="Profilnr,Lengde,Lengde 3D,Element,Lukket")
    extra_exclude_cols = [c.strip() for c in extra_exclude.split(",") if c.strip()]

run_btn = st.button("Kjør matching", type="primary", disabled=(theo_file is None or meas_file is None))

if run_btn:
    theo_df = pd.read_excel(theo_file)
    meas_df = pd.read_excel(meas_file)

    # Fyll ned ID for matching (internt)
    if id_ffill:
        if group_col in theo_df.columns:
            theo_df = forward_fill_group_ids(theo_df, group_col)
        if group_col in meas_df.columns:
            meas_df = forward_fill_group_ids(meas_df, group_col)

    # Intern gruppekolonne (alltid fylt) brukes til all matching/grouping
    gid_col = "__gid__"
    if group_col not in theo_df.columns or group_col not in meas_df.columns:
        st.error(f"Fant ikke ID-kolonnen '{group_col}' i begge filer.")
        st.stop()
    theo_df[gid_col] = theo_df[group_col]
    meas_df[gid_col] = meas_df[group_col]

    # Sjekk nødvendige kolonner
    needed = [gid_col, order_col, x_col, y_col]
    missing_cols = [c for c in needed if c not in theo_df.columns or c not in meas_df.columns]
    if missing_cols:
        st.error(f"Mangler nødvendige kolonner i en av filene: {missing_cols}")
        st.stop()

    # Bygg linjer og representativ attributtrad per gruppe
    theo_lines, theo_group_rows = build_lines_from_points(theo_df, gid_col, order_col, x_col, y_col)
    meas_lines, meas_group_rows = build_lines_from_points(meas_df, gid_col, order_col, x_col, y_col)

    theo_rep = group_attributes(theo_df, gid_col, prefer_first_row=True)
    meas_rep = group_attributes(meas_df, gid_col, prefer_first_row=True)

    # Velg attributter å overføre
    exclude_cols = set([gid_col, group_col, order_col, x_col, y_col, "geometry", "n_points"] + extra_exclude_cols)
    all_theo_cols = [c for c in theo_rep.columns if c not in exclude_cols]

    chosen_cols = []
    if selection_mode == "Utvalg":
        chosen_cols = st.multiselect(
            "Velg attributter å overføre",
            options=all_theo_cols,
            default=[c for c in all_theo_cols if c.startswith("VEAS_")]
        )
    transfer_cols = pick_transfer_columns(theo_rep, exclude_cols, selection_mode, chosen_cols)

    # Spatial index on theoretical geometries
    theo_geoms = theo_lines["geometry"].tolist()
    theo_ids = theo_lines[gid_col].tolist()
    tree = STRtree(theo_geoms)
    geom_to_gid = {id(g): gid for g, gid in zip(theo_geoms, theo_ids)}

    meas_out = meas_df.copy()
    match_rows = []

    for _, m in meas_lines.iterrows():
        mgid = m[gid_col]
        mgeom = m["geometry"]

        if mgeom is None or mgeom.length == 0:
            match_rows.append({"meas_id": mgid, "status":"no_geometry", "theo_id": None})
            continue

        meas_attrs = meas_rep.loc[mgid].to_dict() if mgid in meas_rep.index else {}

        # Expand bbox with max_dist to get candidates
        minx, miny, maxx, maxy = mgeom.bounds
        query_env = LineString([(minx-max_dist, miny-max_dist),(maxx+max_dist, maxy+max_dist)]).envelope

        candidates = tree.query(query_env)
        best = None

        for tg in candidates:
            tgid = geom_to_gid.get(id(tg))
            if tgid is None:
                continue

            line_dist = float(mgeom.distance(tg))
            if line_dist > max_dist:
                continue

            theo_attrs = theo_rep.loc[tgid].to_dict() if tgid in theo_rep.index else {}
            if field_rules and not matches_required_fields(meas_attrs, theo_attrs, field_rules):
                continue

            dists = sample_distances(mgeom, tg, n_samples=n_samples)
            hits = sum(1 for d in dists if d <= max_dist)
            if hits < min_hits:
                continue

            median_dist = float(np.median(dists)) if dists else 1e9
            score = (-hits, median_dist, line_dist)  # lower is better

            if best is None or score < best["score"]:
                best = {"tgid": tgid, "hits": hits, "median_dist": median_dist, "line_dist": line_dist, "score": score}

        if best is None:
            match_rows.append({"meas_id": mgid, "status":"no_match", "theo_id": None})
            continue

        tgid = best["tgid"]
        theo_row = theo_rep.loc[tgid].to_dict() if tgid in theo_rep.index else {}

        meas_out = transfer_to_measured_rows(meas_out, meas_group_rows, mgid, theo_row, transfer_cols, fill_all_rows=fill_all_rows)

        match_rows.append({
            "meas_id": mgid,
            "status": "matched",
            "theo_id": tgid,
            "hits": int(best["hits"]),
            "median_dist": float(best["median_dist"]),
            "line_dist": float(best["line_dist"]),
        })

    match_report = pd.DataFrame(match_rows)

    # Rydd opp output: fjern intern gid-kolonne, og evt. gjenopprett Gemini-ID-mønster
    if output_id_first_only:
        meas_out = restore_gemini_id_pattern(meas_out, id_col=group_col, gid_col=gid_col, order_col=order_col)

    # Fjern intern kolonne uansett
    if gid_col in meas_out.columns:
        meas_out = meas_out.drop(columns=[gid_col])

    st.subheader("Resultat")
    c1, c2, c3 = st.columns(3)
    c1.metric("Innmålte linjer", int(len(meas_lines)))
    c2.metric("Matchet", int((match_report["status"]=="matched").sum()))
    c3.metric("Ikke matchet", int((match_report["status"]!="matched").sum()))

    st.write("Match-rapport (per innmålt linje):")
    st.dataframe(match_report, use_container_width=True)

    st.write("Preview av utdata (første 200 rader):")
    st.dataframe(meas_out.head(200), use_container_width=True)

    import io
    out_buf = io.BytesIO()
    with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
        meas_out.to_excel(writer, index=False, sheet_name="innmalt_med_attrib")
        match_report.to_excel(writer, index=False, sheet_name="match_report")

    st.download_button(
        "Last ned resultat (Excel)",
        data=out_buf.getvalue(),
        file_name="innmalt_med_teoretiske_attrib.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.success("Ferdig. Sjekk 'match_report' for linjer som trenger manuell vurdering.")
