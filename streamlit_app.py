import streamlit as st
import pandas as pd
import numpy as np
import re
import difflib
from shapely.geometry import LineString
from shapely.strtree import STRtree

st.set_page_config(page_title="VA-linje matcher (v10)", layout="wide")


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


def strtree_candidates_as_indices(tree, query_geom, geom_id_to_idx):
    q = tree.query(query_geom)

    if isinstance(q, (list, tuple, np.ndarray)) and len(q) > 0 and isinstance(q[0], (int, np.integer)):
        return [int(i) for i in q]

    cand_idxs = []
    for g in q:
        idx = geom_id_to_idx.get(id(g))
        if idx is not None:
            cand_idxs.append(idx)
    return cand_idxs


def count_matching_pairs(meas_attrs, theo_attrs, rules, ignore_missing_in_measured=True):
    match_count = 0
    for r in rules:
        mc = r["meas_col"]
        tc = r["theo_col"]
        mode = r.get("mode", "trim_upper")

        mv = normalize_value(meas_attrs.get(mc, None), mode=mode)
        tv = normalize_value(theo_attrs.get(tc, None), mode=mode)

        if ignore_missing_in_measured and mv is None:
            continue
        if mv is not None and tv is not None and mv == tv:
            match_count += 1
    return match_count


def snitt_hits(measured_line: LineString, theoretical_line: LineString, max_dist: float, n_points: int) -> int:
    if n_points <= 0:
        return 0
    L = float(measured_line.length)
    if L <= 0:
        return 0

    hits = 0
    for i in range(n_points):
        frac = 0.0 if n_points == 1 else (i / (n_points - 1))
        p = measured_line.interpolate(frac * L)
        if p.distance(theoretical_line) <= max_dist:
            hits += 1
    return hits


# -----------------------------
# UI
# -----------------------------
st.title("VA-linje matcher (v10): teoretisk → innmålt (Excel → Excel)")

colA, colB = st.columns(2)
with colA:
    theo_file = st.file_uploader("Teoretisk datasett (Excel)", type=["xlsx"], key="theo")
with colB:
    meas_file = st.file_uploader("Innmålt datasett (Excel)", type=["xlsx"], key="meas")

# Match-regler (vises alltid)
with st.expander("Match-regler", expanded=True):
    max_dist = st.number_input("Bufferavstand (meter) (2D)", min_value=0.0, value=1.0, step=0.1)

    use_snitt = st.checkbox("Bruk snitt-kontroll langs linja", value=True)
    snitt_n = st.number_input("Antall snittpunkter", min_value=1, value=7, step=1, disabled=not use_snitt)
    snitt_min_hits = st.number_input("Minimum treff (snittpunkter innenfor buffer)", min_value=1, value=4, step=1, disabled=not use_snitt)

    min_attr_matches = st.slider("Min. antall matchende attributter", min_value=0, max_value=10, value=1, step=1)
    ignore_missing = st.checkbox("Ignorer match-felt som er tomme i innmålt (wildcard)", value=True)
    output_id_first_only = st.checkbox("Skriv ID kun på første punkt per linje i output (Gemini-format)", value=True)

with st.expander("Overføring", expanded=True):
    st.write("Innmålt geometri/punktdata blir aldri endret.")
    protected_text = st.text_input(
        "Ekstra beskyttede kolonner (kommaseparert) – i tillegg til standard (Høyde/Øst/Nord/...)",
        value="",
    )
    extra_protected = [c.strip() for c in protected_text.split(",") if c.strip()]

# Vi bygger opp “konfig” først, og kjører matching kun på knapp
config_ready = False
match_rules = []
transfer_cols = []
theo_df = meas_df = None

if theo_file is not None and meas_file is not None:
    # Les filer (dette skjer på rerun også, men det er OK — det kjører raskt)
    theo_df = pd.read_excel(theo_file)
    meas_df = pd.read_excel(meas_file)

    # Detect base columns (for info + senere matching)
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

    # -----------------------------
    # Match-attributter (mapping) – FØR knappen
    # -----------------------------
    st.subheader("1) Velg match-attributter (kolonnemapping)")
    st.caption("Velg hvilke kolonner som skal sammenlignes. Innmålt og teoretisk kan ha ulike navn.")

    meas_cols = list(meas_df.columns)
    theo_cols = list(theo_df.columns)

    # Default forslag (du kan endre i UI)
    default_rules = pd.DataFrame([
        {"meas_col": meas_cols[0], "theo_col": theo_cols[0], "mode": "trim_upper"},
    ])
    # Hvis vi har “kjente” navn, prøv å foreslå bedre default
    for cand in ["VEAS_VA.Type og Dimensjon", "VEAS_Felles.Dimensjon", "VEAS_VA.Dimensjon (mm)", "Dimensjon", "Type"]:
        if cand in meas_cols:
            default_rules.loc[0, "meas_col"] = cand
            break
    for cand in ["VEAS_VA.Type og Dimensjon", "VEAS_Felles.Dimensjon", "VEAS_VA.Dimensjon (mm)", "Dimensjon", "Type"]:
        if cand in theo_cols:
            default_rules.loc[0, "theo_col"] = cand
            break

    rules_df = st.data_editor(
        default_rules,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "meas_col": st.column_config.SelectboxColumn("Innmålt kolonne", options=meas_cols, required=True),
            "theo_col": st.column_config.SelectboxColumn("Teoretisk kolonne", options=theo_cols, required=True),
            "mode": st.column_config.SelectboxColumn("Normalisering", options=["trim_upper", "digits_only", "raw"], required=True),
        },
        key="rules_editor_v10",
    )

    match_rules = rules_df.dropna(subset=["meas_col", "theo_col"]).to_dict(orient="records")

    if len(match_rules) == 0:
        st.warning("Legg til minst én match-regel (kolonnemapping) før du kjører.")
    else:
        # ekstra guard (skal være unødvendig siden selectbox)
        missing = []
        for r in match_rules:
            if r["meas_col"] not in meas_cols or r["theo_col"] not in theo_cols:
                missing.append(r)
        if missing:
            st.error("Noen valgte match-felt finnes ikke i filene (uventet).")
            st.stop()

    # -----------------------------
    # Velg hvilke attributter som skal overføres – FØR knappen
    # -----------------------------
    st.subheader("2) Velg hvilke attributter som skal overføres")

    # Fill IDs down (Gemini-style) for representative rows later
    theo_tmp = forward_fill(theo_df.copy(), id_col)
    meas_tmp = forward_fill(meas_df.copy(), id_col_m)

    gid_col = "__gid__"
    theo_tmp[gid_col] = theo_tmp[id_col]
    meas_tmp[gid_col] = meas_tmp[id_col_m]

    theo_rep = group_rep(theo_tmp, gid_col)

    protected_cols = {
        "Id", "Nr.", "Øst", "Nord", "Høyde",
        "Profilnr", "Lengde", "Lengde 3D",
        "Z", "Kote", "NN2000", "Høyde (m)", "Kote (m)"
    }.union(set(extra_protected))

    exclude = {
        gid_col,
        id_col, order_col, x_col, y_col,
        id_col_m, order_col_m, x_col_m, y_col_m,
    }.union(protected_cols)

    candidate_transfer_cols = sorted([c for c in theo_rep.columns if c not in exclude])

    transfer_mode = st.radio("Overføring", options=["Alle", "Utvalg"], horizontal=True, key="transfer_mode_v10")

    if transfer_mode == "Utvalg":
        transfer_cols = st.multiselect(
            "Velg kolonner fra teoretisk som skal kopieres til innmålt",
            options=candidate_transfer_cols,
            default=[c for c in candidate_transfer_cols if c.lower().startswith("veas")][:20],
            key="transfer_cols_v10"
        )
    else:
        transfer_cols = candidate_transfer_cols

    if len(transfer_cols) == 0:
        st.warning("Velg minst én kolonne å overføre (eller velg 'Alle').")

    # Klar til kjøring når vi har minst én matchregel og minst én transfer-kolonne
    config_ready = (len(match_rules) > 0) and (len(transfer_cols) > 0)

else:
    st.info("Last opp begge Excel-filene for å få valg av match-attributter og overføringskolonner.")


# -----------------------------
# Run button (only executes matching)
# -----------------------------
run_btn = st.button("Kjør matching", type="primary", disabled=not config_ready)

if run_btn:
    # Re-load cleanly from upload objects (safe)
    theo_df = pd.read_excel(theo_file)
    meas_df = pd.read_excel(meas_file)

    # Detect base columns again
    id_col = detect_col(theo_df, ["Id", "ID", "id"])
    order_col = detect_col(theo_df, ["Nr.", "Nr", "NR", "nr", "PunktNr", "punktnr"])
    x_col = detect_col(theo_df, ["Øst", "Ost", "X", "E", "East"])
    y_col = detect_col(theo_df, ["Nord", "Y", "N", "North"])

    id_col_m = detect_col(meas_df, ["Id", "ID", "id"])
    order_col_m = detect_col(meas_df, ["Nr.", "Nr", "NR", "nr", "PunktNr", "punktnr"])
    x_col_m = detect_col(meas_df, ["Øst", "Ost", "X", "E", "East"])
    y_col_m = detect_col(meas_df, ["Nord", "Y", "N", "North"])

    # Force Gemini pattern -> ffill IDs
    theo_df = forward_fill(theo_df, id_col)
    meas_df = forward_fill(meas_df, id_col_m)

    gid_col = "__gid__"
    theo_df[gid_col] = theo_df[id_col]
    meas_df[gid_col] = meas_df[id_col_m]

    # Build lines
    theo_lines, theo_group_rows = build_lines(theo_df, gid_col, order_col, x_col, y_col)
    meas_lines, meas_group_rows = build_lines(meas_df, gid_col, order_col_m, x_col_m, y_col_m)

    st.info(f"Bygget {len(theo_lines)} teoretiske linjer og {len(meas_lines)} innmålte linjer.")

    if len(theo_lines) == 0 or len(meas_lines) == 0:
        st.error("Ingen linjer kunne bygges (sjekk at hver linje har minst 2 punkt med X/Y).")
        st.stop()

    theo_rep = group_rep(theo_df, gid_col)
    meas_rep = group_rep(meas_df, gid_col)

    # Spatial index on theoretical lines
    theo_geoms = [g for _, g, _ in theo_lines]
    theo_ids = [gid for gid, _, _ in theo_lines]
    tree = STRtree(theo_geoms)
    geom_id_to_idx = {id(g): i for i, g in enumerate(theo_geoms)}

    meas_out = meas_df.copy()
    match_rows = []

    for mgid, mgeom, _ in meas_lines:
        meas_attrs = meas_rep.loc[mgid].to_dict() if mgid in meas_rep.index else {}

        buf = mgeom.buffer(max_dist)
        cand_idxs = strtree_candidates_as_indices(tree, buf, geom_id_to_idx)

        best = None
        for idx in cand_idxs:
            tgid = theo_ids[idx]
            tgeom = theo_geoms[idx]

            d = float(mgeom.distance(tgeom))
            if d > max_dist:
                continue

            theo_attrs = theo_rep.loc[tgid].to_dict() if tgid in theo_rep.index else {}

            mcount = count_matching_pairs(
                meas_attrs, theo_attrs, match_rules, ignore_missing_in_measured=ignore_missing
            )
            if mcount < min_attr_matches:
                continue

            hits = None
            if use_snitt:
                hits = snitt_hits(mgeom, tgeom, max_dist, int(snitt_n))
                if hits < int(snitt_min_hits):
                    continue

            score = (d, -mcount, -(hits if hits is not None else 0))
            if best is None or score < best["score"]:
                best = {"tgid": tgid, "d": d, "mcount": mcount, "hits": hits, "score": score}

        if best is None:
            match_rows.append({
                "meas_id": mgid,
                "status": "no_match",
                "theo_id": None,
                "attr_matches": 0,
                "line_dist": None,
                "snitt_hits": None if not use_snitt else 0,
            })
            continue

        tgid = best["tgid"]
        theo_row = theo_rep.loc[tgid].to_dict()

        idxs = meas_group_rows.get(mgid, [])
        for col in transfer_cols:
            if col not in meas_out.columns:
                meas_out[col] = np.nan
            meas_out.loc[idxs, col] = theo_row.get(col, np.nan)

        match_rows.append({
            "meas_id": mgid,
            "status": "matched",
            "theo_id": tgid,
            "attr_matches": int(best["mcount"]),
            "line_dist": float(best["d"]),
            "snitt_hits": None if not use_snitt else int(best["hits"]),
        })

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
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
