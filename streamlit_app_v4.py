
import streamlit as st
import pandas as pd
import numpy as np
import re
from shapely.geometry import LineString
from shapely.strtree import STRtree

st.set_page_config(page_title="VA-linje matcher (teoretisk → innmålt)", layout="wide")

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
    df[group_col] = df[group_col].ffill()
    return df

def build_lines_from_points(df, gid_col, order_col, x_col, y_col):
    df = df.copy()
    df[order_col] = pd.to_numeric(df[order_col], errors="coerce")
    df = df.sort_values([gid_col, order_col], kind="mergesort")

    group_rows = {}
    records = []

    for gid, g in df.groupby(gid_col, dropna=False):
        idxs = g.index.tolist()
        group_rows[gid] = idxs

        xs = pd.to_numeric(g[x_col], errors="coerce").to_numpy()
        ys = pd.to_numeric(g[y_col], errors="coerce").to_numpy()
        mask = ~np.isnan(xs) & ~np.isnan(ys)
        xs, ys = xs[mask], ys[mask]

        geom = LineString(list(zip(xs.tolist(), ys.tolist()))) if len(xs) >= 2 else None
        records.append({gid_col: gid, "geometry": geom, "n_points": int(len(xs))})

    lines_df = pd.DataFrame(records)
    lines_df = lines_df[lines_df["geometry"].notna()].reset_index(drop=True)
    return lines_df, group_rows

def group_attributes(df, gid_col):
    d = df.copy()
    d = d.groupby(gid_col, dropna=False).apply(lambda g: g.ffill()).reset_index(drop=True)
    rep = d.groupby(gid_col, dropna=False).head(1).copy()
    return rep.set_index(gid_col)

def sample_distances(meas_line, theo_line, n_samples=7):
    if meas_line is None or theo_line is None or meas_line.length == 0:
        return []
    ds = []
    for t in np.linspace(0, 1, n_samples):
        p = meas_line.interpolate(t, normalized=True)
        ds.append(float(p.distance(theo_line)))
    return ds

def count_matching_fields(meas_attrs, theo_attrs, field_rules, ignore_missing_in_measured=True):
    match_count = 0
    compared = 0
    for rule in field_rules:
        col = rule["name"]
        mode = rule.get("mode", "trim_upper")
        mv = normalize_value(meas_attrs.get(col, None), mode=mode)
        tv = normalize_value(theo_attrs.get(col, None), mode=mode)
        if ignore_missing_in_measured and mv is None:
            continue
        compared += 1
        if mv is not None and tv is not None and mv == tv:
            match_count += 1
    return match_count, compared

def pick_transfer_columns(theo_rep_df, exclude_cols):
    return [c for c in theo_rep_df.columns if c not in exclude_cols]

def transfer_to_measured_rows(meas_df, meas_group_rows, meas_gid, theo_row, transfer_cols):
    idxs = meas_group_rows.get(meas_gid, [])
    if not idxs:
        return meas_df
    for col in transfer_cols:
        if col not in meas_df.columns:
            meas_df[col] = np.nan
        meas_df.loc[idxs, col] = theo_row.get(col, np.nan)
    return meas_df

def restore_gemini_id_pattern(df, id_col, gid_col, order_col):
    out = df.copy()
    tmp = out[[gid_col, order_col]].copy()
    tmp[order_col] = pd.to_numeric(tmp[order_col], errors="coerce")
    first_idx = (
        tmp.sort_values([gid_col, order_col], kind="mergesort")
           .groupby(gid_col, dropna=False)
           .head(1)
           .index
    )
    out.loc[~out.index.isin(first_idx), id_col] = np.nan
    return out

st.title("VA-linje matcher: teoretisk → innmålt (Excel → Excel)")

colA, colB = st.columns(2)
with colA:
    theo_file = st.file_uploader("Teoretisk datasett (Excel)", type=["xlsx"], key="theo")
with colB:
    meas_file = st.file_uploader("Innmålt datasett (Excel)", type=["xlsx"], key="meas")

with st.expander("1) Kolonner", expanded=True):
    group_col = st.text_input("Linje-ID kolonne (kun internt i hvert datasett)", value="Id")
    order_col = st.text_input("Punktrekkefølge", value="Nr.")
    x_col = st.text_input("X (Øst)", value="Øst")
    y_col = st.text_input("Y (Nord)", value="Nord")
    output_id_first_only = st.checkbox("Skriv ID kun på første punkt per linje i output (Gemini-format)", value=True)

with st.expander("2) Match-regler", expanded=True):
    max_dist = st.number_input("Bufferavstand (meter) (2D)", min_value=0.0, value=5.0, step=0.5)
    use_slice_test = st.checkbox("Bruk snitt-test langs linja (strengere)", value=False)
    n_samples = st.slider("Antall snittpunkter (hvis aktivert)", min_value=3, max_value=15, value=7, step=1)
    min_hits = st.slider("Min. treff innen buffer (hvis aktivert)", min_value=1, max_value=15, value=4, step=1)

    min_attr_matches = st.slider("Min. antall matchende attributter", min_value=0, max_value=5, value=2, step=1)
    ignore_missing = st.checkbox("Ignorer match-felt som er tomme i innmålt (wildcard)", value=True)

    st.caption("Match-felter (må finnes i begge filer).")
    rules_df = st.data_editor(
        pd.DataFrame([{"name":"VEAS_VA.Type og Dimensjon","mode":"trim_upper"},
                      {"name":"VEAS_VA.Dimensjon (mm)","mode":"digits_only"}]),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Kolonnenavn", required=True),
            "mode": st.column_config.SelectboxColumn("Normalisering", options=["trim_upper","digits_only","raw"])
        },
        key="rules_editor"
    )
    field_rules = rules_df.dropna(subset=["name"]).to_dict(orient="records")

with st.expander("3) Overføring", expanded=True):
    st.caption("Denne v4 overfører ALT fra teoretisk (unntatt tekniske punktfelt). Utvalg kommer senere.")
    extra_exclude = st.text_input("Ekstra kolonner å ekskludere (kommaseparert)", value="Profilnr,Lengde,Lengde 3D,Element,Lukket,S_OBJID")
    extra_exclude_cols = [c.strip() for c in extra_exclude.split(",") if c.strip()]

run_btn = st.button("Kjør matching", type="primary", disabled=(theo_file is None or meas_file is None))

if run_btn:
    theo_df = pd.read_excel(theo_file)
    meas_df = pd.read_excel(meas_file)

    # FORCE: Gemini-typisk format (Id bare på første rad) -> vi fyller alltid ned for matching
    theo_df = forward_fill_group_ids(theo_df, group_col)
    meas_df = forward_fill_group_ids(meas_df, group_col)

    gid_col = "__gid__"
    theo_df[gid_col] = theo_df[group_col]
    meas_df[gid_col] = meas_df[group_col]

    needed = [gid_col, order_col, x_col, y_col]
    missing_cols = [c for c in needed if c not in theo_df.columns or c not in meas_df.columns]
    if missing_cols:
        st.error(f"Mangler nødvendige kolonner i en av filene: {missing_cols}")
        st.stop()

    bad_rules = [r["name"] for r in field_rules if (r["name"] not in theo_df.columns or r["name"] not in meas_df.columns)]
    if bad_rules:
        st.error("Disse match-feltene finnes ikke i begge filer. Fjern dem eller endre navn:\n\n- " + "\n- ".join(bad_rules))
        st.stop()

    theo_lines, theo_group_rows = build_lines_from_points(theo_df, gid_col, order_col, x_col, y_col)
    meas_lines, meas_group_rows = build_lines_from_points(meas_df, gid_col, order_col, x_col, y_col)

    st.info(f"Bygget {len(theo_lines)} teoretiske linjer og {len(meas_lines)} innmålte linjer.")

    theo_rep = group_attributes(theo_df, gid_col)
    meas_rep = group_attributes(meas_df, gid_col)

    exclude_cols = set([gid_col, group_col, order_col, x_col, y_col, "geometry", "n_points"] + extra_exclude_cols)
    transfer_cols = pick_transfer_columns(theo_rep, exclude_cols)

    # Spatial index
    theo_geoms = theo_lines["geometry"].tolist()
    theo_ids = theo_lines[gid_col].tolist()
    tree = STRtree(theo_geoms)
    geom_to_gid = {id(g): gid for g, gid in zip(theo_geoms, theo_ids)}

    meas_out = meas_df.copy()
    match_rows = []

    for _, m in meas_lines.iterrows():
        mgid = m[gid_col]
        mgeom = m["geometry"]
        meas_attrs = meas_rep.loc[mgid].to_dict() if mgid in meas_rep.index else {}

        candidates = tree.query(mgeom.buffer(max_dist))
        best = None

        for tg in candidates:
            tgid = geom_to_gid.get(id(tg))
            if tgid is None:
                continue

            line_dist = float(mgeom.distance(tg))
            if line_dist > max_dist:
                continue

            theo_attrs = theo_rep.loc[tgid].to_dict() if tgid in theo_rep.index else {}
            mcount, _ = count_matching_fields(meas_attrs, theo_attrs, field_rules, ignore_missing_in_measured=ignore_missing)
            if mcount < min_attr_matches:
                continue

            if use_slice_test:
                dists = sample_distances(mgeom, tg, n_samples=n_samples)
                hits = sum(1 for d in dists if d <= max_dist)
                if hits < min_hits:
                    continue
                median_dist = float(np.median(dists)) if dists else 1e9
            else:
                hits = None
                median_dist = line_dist

            score = (median_dist, -mcount, line_dist)  # lower distance, more attrs
            if best is None or score < best["score"]:
                best = {"tgid": tgid, "mcount": mcount, "line_dist": line_dist, "hits": hits, "median_dist": median_dist, "score": score}

        if best is None:
            match_rows.append({"meas_id": mgid, "status":"no_match", "theo_id": None})
            continue

        tgid = best["tgid"]
        theo_row = theo_rep.loc[tgid].to_dict()
        meas_out = transfer_to_measured_rows(meas_out, meas_group_rows, mgid, theo_row, transfer_cols)

        match_rows.append({
            "meas_id": mgid,
            "status": "matched",
            "theo_id": tgid,
            "attr_matches": int(best["mcount"]),
            "line_dist": float(best["line_dist"]),
            "hits": best["hits"],
        })

    match_report = pd.DataFrame(match_rows)

    if output_id_first_only:
        meas_out = restore_gemini_id_pattern(meas_out, id_col=group_col, gid_col=gid_col, order_col=order_col)

    meas_out = meas_out.drop(columns=[gid_col], errors="ignore")

    st.subheader("Resultat")
    c1, c2, c3 = st.columns(3)
    c1.metric("Innmålte linjer", int(len(meas_lines)))
    c2.metric("Matchet", int((match_report["status"]=="matched").sum()))
    c3.metric("Ikke matchet", int((match_report["status"]!="matched").sum()))

    st.dataframe(match_report, use_container_width=True)

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
