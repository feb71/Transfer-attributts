import streamlit as st
import pandas as pd
import numpy as np
import re
import difflib
from shapely.geometry import LineString
from shapely.strtree import STRtree

st.set_page_config(page_title="VA-linje matcher (v7)", layout="wide")

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

def detect_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def forward_fill(df, col):
    if col in df.columns:
        df[col] = df[col].ffill()
    return df

def build_lines(df, gid_col, order_col, x_col, y_col):
    d = df.copy()
    d[order_col] = pd.to_numeric(d[order_col], errors="coerce")
    d = d.sort_values([gid_col, order_col], kind="mergesort")

    group_rows = {}
    lines = []
    for gid, g in d.groupby(gid_col, dropna=False):
        idxs = g.index.tolist()
        group_rows[gid] = idxs

        xs = pd.to_numeric(g[x_col], errors="coerce").to_numpy()
        ys = pd.to_numeric(g[y_col], errors="coerce").to_numpy()
        mask = ~np.isnan(xs) & ~np.isnan(ys)
        xs, ys = xs[mask], ys[mask]

        if len(xs) >= 2:
            lines.append((gid, LineString(list(zip(xs.tolist(), ys.tolist()))), int(len(xs))))
    return lines, group_rows

def group_rep(df, gid_col):
    d = df.copy()
    d = d.groupby(gid_col, dropna=False).apply(lambda g: g.ffill()).reset_index(drop=True)
    rep = d.groupby(gid_col, dropna=False).head(1).copy()
    return rep.set_index(gid_col)

def count_matching_fields(meas_attrs, theo_attrs, field_rules, ignore_missing_in_measured=True):
    match_count = 0
    for rule in field_rules:
        col = rule["name"]
        mode = rule.get("mode", "trim_upper")
        mv = normalize_value(meas_attrs.get(col, None), mode=mode)
        tv = normalize_value(theo_attrs.get(col, None), mode=mode)
        if ignore_missing_in_measured and mv is None:
            continue
        if mv is not None and tv is not None and mv == tv:
            match_count += 1
    return match_count

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

with st.expander("Match-regler", expanded=True):
    max_dist = st.number_input("Bufferavstand (meter) (2D)", min_value=0.0, value=1.0, step=0.1)
    min_attr_matches = st.slider("Min. antall matchende attributter", min_value=0, max_value=5, value=1, step=1)
    ignore_missing = st.checkbox("Ignorer match-felt som er tomme i innmålt (wildcard)", value=True)
    output_id_first_only = st.checkbox("Skriv ID kun på første punkt per linje i output (Gemini-format)", value=True)

    st.caption("Match-felter (må finnes i begge filer).")
    rules_df = st.data_editor(
        pd.DataFrame([
            {"name":"VEAS_Felles.Dimensjon","mode":"raw"},
        ]),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Kolonnenavn", required=True),
            "mode": st.column_config.SelectboxColumn("Normalisering", options=["trim_upper","digits_only","raw"])
        },
        key="rules_editor"
    )
    field_rules = rules_df.dropna(subset=["name"]).to_dict(orient="records")

run_btn = st.button("Kjør matching", type="primary", disabled=(theo_file is None or meas_file is None))

if run_btn:
    theo_df = pd.read_excel(theo_file)
    meas_df = pd.read_excel(meas_file)

    # Auto-detect key columns
    id_col = detect_col(theo_df, ["Id", "ID", "id"])
    order_col = detect_col(theo_df, ["Nr.", "Nr", "NR", "nr", "PunktNr", "punktnr"])
    x_col = detect_col(theo_df, ["Øst", "Ost", "X", "E", "East"])
    y_col = detect_col(theo_df, ["Nord", "Y", "N", "North"])

    id_col_m = detect_col(meas_df, ["Id", "ID", "id"])
    order_col_m = detect_col(meas_df, ["Nr.", "Nr", "NR", "nr", "PunktNr", "punktnr"])
    x_col_m = detect_col(meas_df, ["Øst", "Ost", "X", "E", "East"])
    y_col_m = detect_col(meas_df, ["Nord", "Y", "N", "North"])

    if None in [id_col, order_col, x_col, y_col, id_col_m, order_col_m, x_col_m, y_col_m]:
        st.error("Fant ikke obligatoriske kolonner (Id/Nr/Øst/Nord) i en av filene.")
        st.stop()

    st.write("Oppdaget kolonner:",
             {"teoretisk": {"Id": id_col, "Nr": order_col, "X": x_col, "Y": y_col},
              "innmålt": {"Id": id_col_m, "Nr": order_col_m, "X": x_col_m, "Y": y_col_m}})

    with st.expander("Vis kolonner i filene (debug)", expanded=False):
        st.write("Teoretisk kolonner:", list(theo_df.columns))
        st.write("Innmålt kolonner:", list(meas_df.columns))

    # Force Gemini pattern -> ffill IDs
    theo_df = forward_fill(theo_df, id_col)
    meas_df = forward_fill(meas_df, id_col_m)

    gid_col = "__gid__"
    theo_df[gid_col] = theo_df[id_col]
    meas_df[gid_col] = meas_df[id_col_m]

    # Validate match fields exist + suggest closest names
    missing = []
    for r in field_rules:
        name = str(r["name"])
        if name not in theo_df.columns or name not in meas_df.columns:
            missing.append(name)

    if missing:
        msg = ["Disse match-feltene finnes ikke i begge filer. Fjern dem eller endre navn:\n"]
        for name in missing:
            msg.append(f"\n• {name}")
            msg.append(f"  - Nærmeste i teoretisk: {difflib.get_close_matches(name, list(theo_df.columns), n=5, cutoff=0.6)}")
            msg.append(f"  - Nærmeste i innmålt:   {difflib.get_close_matches(name, list(meas_df.columns), n=5, cutoff=0.6)}")
        st.error("\n".join(msg))
        st.stop()

    # Build lines
    theo_lines, theo_group_rows = build_lines(theo_df, gid_col, order_col, x_col, y_col)
    meas_lines, meas_group_rows = build_lines(meas_df, gid_col, order_col_m, x_col_m, y_col_m)

    st.info(f"Bygget {len(theo_lines)} teoretiske linjer og {len(meas_lines)} innmålte linjer.")

    if len(theo_lines) == 0 or len(meas_lines) == 0:
        st.error("Ingen linjer kunne bygges (sjekk at hver linje har minst 2 punkt med X/Y).")
        st.stop()

    theo_rep = group_rep(theo_df, gid_col)
    meas_rep = group_rep(meas_df, gid_col)

    exclude = {gid_col, id_col, order_col, x_col, y_col, id_col_m, order_col_m, x_col_m, y_col_m}
    transfer_cols = [c for c in theo_rep.columns if c not in exclude]

    theo_geoms = [g for _, g, _ in theo_lines]
    theo_ids = [gid for gid, _, _ in theo_lines]
    tree = STRtree(theo_geoms)

    geom_id_to_idx = {id(g): i for i, g in enumerate(theo_geoms)}

    meas_out = meas_df.copy()
    match_rows = []

    for mgid, mgeom, _ in meas_lines:
        meas_attrs = meas_rep.loc[mgid].to_dict() if mgid in meas_rep.index else {}

        buf = mgeom.buffer(max_dist)
        q = tree.query(buf)
        if isinstance(q, (list, tuple, np.ndarray)) and len(q) > 0 and isinstance(q[0], (int, np.integer)):
            cand_idxs = [int(i) for i in q]
        else:
            cand_idxs = []
            for tg in q:
                idx = geom_id_to_idx.get(id(tg))
                if idx is not None:
                    cand_idxs.append(idx)

        best = None
        for idx in cand_idxs:
            tgid = theo_ids[idx]
            tg = theo_geoms[idx]
            d = float(mgeom.distance(tg))
            if d > max_dist:
                continue

            theo_attrs = theo_rep.loc[tgid].to_dict() if tgid in theo_rep.index else {}
            mcount = count_matching_fields(meas_attrs, theo_attrs, field_rules, ignore_missing_in_measured=ignore_missing)
            if mcount < min_attr_matches:
                continue

            score = (d, -mcount)
            if best is None or score < best["score"]:
                best = {"tgid": tgid, "d": d, "mcount": mcount, "score": score}

        if best is None:
            match_rows.append({"meas_id": mgid, "status": "no_match", "theo_id": None})
            continue

        tgid = best["tgid"]
        theo_row = theo_rep.loc[tgid].to_dict()

        idxs = meas_group_rows.get(mgid, [])
        for col in transfer_cols:
            if col not in meas_out.columns:
                meas_out[col] = np.nan
            meas_out.loc[idxs, col] = theo_row.get(col, np.nan)

        match_rows.append({"meas_id": mgid, "status": "matched", "theo_id": tgid,
                           "attr_matches": int(best["mcount"]), "line_dist": float(best["d"])})

    match_report = pd.DataFrame(match_rows)

    if output_id_first_only:
        meas_out = restore_gemini_id_pattern(meas_out, id_col=id_col_m, gid_col=gid_col, order_col=order_col_m)

    meas_out = meas_out.drop(columns=[gid_col], errors="ignore")

    st.subheader("Resultat")
    c1, c2, c3 = st.columns(3)
    c1.metric("Innmålte linjer", int(len(meas_lines)))
    c2.metric("Matchet", int((match_report["status"] == "matched").sum()))
    c3.metric("Ikke matchet", int((match_report["status"] != "matched").sum()))

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
