"""
F1 Podium & Winner Predictor
Individual Deployment — Ahnaf Harappa
DATA 4382 Capstone II

Focus: Given a driver in the top-10, will they reach the podium?
       And if so, can they win?

Uses Models B (Podium Filter) and C (Winner Specialist)
from the cascade classifier pipeline.

Run:
    pip install -r requirements.txt
    streamlit run app_ahnaf.py

Data folder structure:
    app_ahnaf.py
    requirements.txt
    data/
        races.csv, results.csv, qualifying.csv,
        constructor_standings.csv, driver_standings.csv, status.csv
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="F1 Podium & Winner Predictor · Ahnaf Harappa",
    page_icon="🏆",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;700;800&family=Inter:wght@300;400;500&display=swap');
:root {
    --gold:#FFD700; --navy:#15151E; --crimson:#E8002D;
    --silver:#C0C0C0; --card:#1e1e2e; --border:#2e2e40;
    --text:#e8e8f0; --muted:#7a7a90; --teal:#00d2ff;
}
html,body,[class*="css"]          { font-family:'Inter',sans-serif; background:var(--navy)!important; color:var(--text); }
[data-testid="stAppViewContainer"] { background:var(--navy); }
[data-testid="stHeader"]           { background:var(--navy); }
[data-testid="stSidebar"]          { background:#0d0d18!important; }
[data-testid="stSidebar"] *        { color:#aaa!important; }
[data-testid="stSidebar"] h2       { color:var(--gold)!important; font-family:'Barlow Condensed',sans-serif; text-transform:uppercase; }
h1,h2,h3 { font-family:'Barlow Condensed',sans-serif; text-transform:uppercase; letter-spacing:.03em; }
.card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:20px 24px; margin-bottom:8px; }
.badge-win  { background:#FFD70022;color:#FFD700;border:1.5px solid #FFD700; }
.badge-pod  { background:#C0C0C022;color:#C0C0C0;border:1.5px solid #C0C0C0; }
.badge-pts  { background:#00d2ff22;color:#00d2ff;border:1.5px solid #00d2ff; }
.badge-win,.badge-pod,.badge-pts {
    border-radius:999px; padding:6px 20px;
    font-family:'Barlow Condensed',sans-serif;
    font-size:1.1rem; font-weight:700; display:inline-block; }
.stButton>button {
    background:var(--gold)!important; color:#15151E!important;
    font-family:'Barlow Condensed',sans-serif!important;
    font-size:1.1rem!important; font-weight:800!important;
    border:none!important; border-radius:8px!important;
    padding:12px 0!important; width:100%!important; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = "data"

# ── Data & Feature Engineering ─────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading F1 data…")
def load_data():
    races         = pd.read_csv(f"{DATA_DIR}/races.csv")
    results       = pd.read_csv(f"{DATA_DIR}/results.csv")
    qualifying    = pd.read_csv(f"{DATA_DIR}/qualifying.csv")
    con_standings = pd.read_csv(f"{DATA_DIR}/constructor_standings.csv")
    drv_standings = pd.read_csv(f"{DATA_DIR}/driver_standings.csv")
    status        = pd.read_csv(f"{DATA_DIR}/status.csv")

    modern = races[races["year"] >= 2010][["raceId","year","round","circuitId","name"]].copy()
    modern.rename(columns={"name":"race_name"}, inplace=True)

    res = results[results["raceId"].isin(modern["raceId"])][
        ["raceId","driverId","constructorId","grid","positionOrder","statusId","points"]].copy()
    res = res.merge(status[["statusId","status"]], on="statusId", how="left")
    res["dnf"] = res["status"].str.contains(
        "Accident|Collision|Engine|Gearbox|Hydraulics|Mechanical|Retired|Suspension|"
        "Electrical|Brakes|Wheel|Oil|Water|Puncture|Fire|Clutch|Overheating|Power Unit",
        case=False, na=False).astype(int)

    qual = qualifying[qualifying["raceId"].isin(modern["raceId"])][
        ["raceId","driverId","position"]].copy()
    qual.rename(columns={"position":"quali_pos"}, inplace=True)

    drv_st = drv_standings[drv_standings["raceId"].isin(modern["raceId"])][
        ["raceId","driverId","position","points"]].copy()
    drv_st.rename(columns={"position":"drv_champ_rank","points":"drv_champ_pts"}, inplace=True)

    con_st = con_standings[con_standings["raceId"].isin(modern["raceId"])][
        ["raceId","constructorId","position","points"]].copy()
    con_st.rename(columns={"position":"con_champ_rank","points":"con_champ_pts"}, inplace=True)

    df = res.merge(modern, on="raceId", how="left")
    df = df.merge(qual, on=["raceId","driverId"], how="left")
    df = df.merge(drv_st, on=["raceId","driverId"], how="left")
    df = df.merge(con_st, on=["raceId","constructorId"], how="left")

    df["quali_pos"] = df["quali_pos"].fillna(df["grid"])
    df["grid"] = df["grid"].replace(0, np.nan).fillna(df["quali_pos"])
    df = df.dropna(subset=["grid","positionOrder"])
    df["grid"] = df["grid"].astype(int)
    df["positionOrder"] = df["positionOrder"].astype(int)
    df = df.sort_values(["year","round","driverId"]).reset_index(drop=True)

    race_stats = df.groupby("raceId")["grid"].agg(["mean","std"]).reset_index()
    race_stats.columns = ["raceId","grid_mean","grid_std"]
    df = df.merge(race_stats, on="raceId", how="left")
    df["pace_z_score"] = -(df["grid"] - df["grid_mean"]) / (df["grid_std"] + 1e-5)

    df["rolling_finish"] = (df.groupby("driverId")["positionOrder"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()))
    df["rolling_pace"] = -df["rolling_finish"].fillna(10)

    df["team_pts_rolling_avg"] = (df.groupby("constructorId")["con_champ_pts"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())).fillna(0)

    df["prev_driver_rank"] = df["drv_champ_rank"].fillna(10)

    df["reliability_risk"] = (df.groupby("constructorId")["dnf"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())).fillna(0.1)

    circ_avg = (df.groupby(["driverId","circuitId"])["positionOrder"]
        .transform(lambda x: x.shift(1).expanding().mean()))
    df["circuit_specialty"] = -(circ_avg.fillna(df["positionOrder"].mean()))
    df["grid_pace_delta"] = df["pace_z_score"] - (df["grid"] - 10) * 0.05

    def cls(pos):
        if pos == 1: return 3
        if pos <= 3: return 2
        if pos <= 10: return 1
        return 0
    df["outcome"] = df["positionOrder"].apply(cls)

    features = ["grid","pace_z_score","team_pts_rolling_avg","prev_driver_rank",
                "reliability_risk","circuit_specialty","grid_pace_delta","rolling_pace"]
    keep = features + ["outcome","year","raceId","driverId","constructorId",
                       "race_name","positionOrder","circuitId"]
    return df[keep].dropna(), features


# ── Train Models B & C ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training Podium & Winner models…")
def train_models(_df, features):
    X = _df[features].values
    y = _df["outcome"].values
    years = _df["year"].values

    train_mask = years <= 2022
    test_mask  = years == 2023
    X_tr, X_te = X[train_mask], X[test_mask]
    y_tr, y_te = y[train_mask], y[test_mask]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    params = dict(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8)

    # Model A (needed to filter top-10 for B training)
    y_a_tr = (y_tr >= 1).astype(int)
    model_a = GradientBoostingClassifier(**params, random_state=42)
    sw_a = compute_sample_weight("balanced", y_a_tr)
    model_a.fit(X_tr_s, y_a_tr, sample_weight=sw_a)
    pred_a_tr = model_a.predict(X_tr_s)
    pred_a_te = model_a.predict(X_te_s)

    # Model B — Podium vs P4-10 (trained on predicted top-10 only)
    top10_tr = pred_a_tr == 1
    top10_te = pred_a_te == 1
    y_b_tr = (y_tr[top10_tr] >= 2).astype(int)
    y_b_te = (y_te[top10_te] >= 2).astype(int)
    sw_b = compute_sample_weight("balanced", y_b_tr)
    model_b = GradientBoostingClassifier(**params, random_state=7)
    model_b.fit(X_tr_s[top10_tr], y_b_tr, sample_weight=sw_b)
    pred_b_te = model_b.predict(X_te_s[top10_te]) if top10_te.sum() > 0 else np.array([])
    rep_b = (classification_report(y_b_te, pred_b_te,
                                   target_names=["P4-10","Podium"], output_dict=True)
             if len(y_b_te) > 1 else {})
    auc_b = (roc_auc_score(y_b_te, model_b.predict_proba(X_te_s[top10_te])[:,1])
             if len(y_b_te) > 1 and len(np.unique(y_b_te)) > 1 else 0.0)

    # Model C — Winner vs P2/3
    podium_tr = top10_tr & (y_tr >= 2)
    podium_te_idx = np.where(top10_te)[0]
    podium_te_mask = np.zeros(len(y_te), dtype=bool)
    if len(pred_b_te) > 0:
        podium_te_mask[podium_te_idx[pred_b_te == 1]] = True
    y_c_tr = (y_tr[podium_tr] == 3).astype(int)
    y_c_te = (y_te[podium_te_mask] == 3).astype(int)
    sw_c = compute_sample_weight("balanced", y_c_tr) if len(np.unique(y_c_tr)) > 1 else None
    model_c = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                         learning_rate=0.05, random_state=99)
    model_c.fit(X_tr_s[podium_tr], y_c_tr, sample_weight=sw_c)
    pred_c_te = model_c.predict(X_te_s[podium_te_mask]) if podium_te_mask.sum() > 0 else np.array([])
    rep_c = (classification_report(y_c_te, pred_c_te,
                                   target_names=["P2/3","P1 Winner"], output_dict=True)
             if len(y_c_te) > 1 else {})
    auc_c = (roc_auc_score(y_c_te, model_c.predict_proba(X_te_s[podium_te_mask])[:,1])
             if len(y_c_te) > 1 and len(np.unique(y_c_te)) > 1 else 0.0)

    # Feature importance
    fi = pd.concat([
        pd.DataFrame({"feature": features, "importance": model_b.feature_importances_, "model": "Model B – Podium Filter"}),
        pd.DataFrame({"feature": features, "importance": model_c.feature_importances_, "model": "Model C – Winner Specialist"}),
    ])

    # Historical podium rates by grid
    podium_by_grid = (_df[_df["grid"] <= 20].groupby("grid")
        .apply(lambda x: pd.Series({
            "Podium Rate %":  (x["outcome"] >= 2).mean() * 100,
            "Win Rate %":     (x["outcome"] == 3).mean() * 100,
        })).reset_index())

    return dict(
        model_a=model_a, model_b=model_b, model_c=model_c,
        scaler=scaler, features=features,
        rep_b=rep_b, rep_c=rep_c, auc_b=auc_b, auc_c=auc_c,
        fi=fi, podium_by_grid=podium_by_grid,
    )


# ── Guard ──────────────────────────────────────────────────────────────────────
if not os.path.exists(DATA_DIR) or not os.path.exists(f"{DATA_DIR}/results.csv"):
    st.error("📁 Place Kaggle F1 CSV files in a `data/` folder next to this app.")
    st.stop()

df, features = load_data()
bundle = train_models(df, features)
model_a = bundle["model_a"]
model_b = bundle["model_b"]
model_c = bundle["model_c"]
scaler  = bundle["scaler"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
feat_means = df[features].mean()
feat_q95   = df[features].quantile(0.95)

with st.sidebar:
    st.markdown("## 🏆 Podium Predictor")
    st.markdown("Ahnaf Harappa · DATA 4382")
    st.markdown("---")
    st.markdown("**Assumes driver is already in Top 10**")
    st.markdown("This tool answers: *from the points scorers, who reaches the podium — and who wins?*")
    st.markdown("---")

    grid_pos  = st.slider("Grid Position", 1, 10, 2, help="Top-10 starters only")
    pace_z    = st.slider("Pace Z-Score vs Field", -3.0, 3.0, 1.2, 0.05)
    team_pts  = st.slider("Constructor Rolling Pts Avg", 0.0,
                          float(max(feat_q95["team_pts_rolling_avg"], 1)),
                          float(feat_means["team_pts_rolling_avg"]))
    prev_rank = st.slider("Driver Championship Rank", 1, 10, 1)
    rel_risk  = st.slider("Reliability Risk", 0.0, 1.0, 0.05, 0.01)
    circ_spec = st.slider("Circuit Specialty Score", -15.0, 0.0,
                          float(feat_means["circuit_specialty"]), 0.1)
    gpd       = st.slider("Grid-Pace Delta", -3.0, 3.0,
                          float(round(pace_z - (grid_pos - 10) * 0.05, 2)), 0.05)
    roll_pace = st.slider("Rolling Pace (last 3 races)", -15.0, 0.0,
                          float(feat_means["rolling_pace"]), 0.1)
    st.markdown("---")
    st.button("🏆 Predict Podium / Win")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#15151E 50%,#1a1200);
            border-top:3px solid #FFD700;border-radius:12px;
            padding:26px 32px;margin-bottom:18px;">
  <div style="font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;
              color:#FFD700;margin-bottom:6px;">
    DATA 4382 Capstone II · Individual Deployment · Ahnaf Harappa
  </div>
  <h1 style="color:#f0f0f8;margin:0 0 4px;font-size:1.9rem;">
    🏆 F1 Podium & Winner Predictor
  </h1>
  <div style="color:#7a7a90;font-size:.88rem;">
    Model B (Podium Filter) + Model C (Winner Specialist) · From Top-10 Drivers Only
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPIs ───────────────────────────────────────────────────────────────────────
def metric_card(col, value, label, color):
    col.markdown(f"""
    <div class="card" style="border-left:4px solid {color};text-align:center;padding:14px;">
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.9rem;
                  font-weight:800;color:{color};">{value}</div>
      <div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;
                  color:#7a7a90;margin-top:3px;">{label}</div>
    </div>""", unsafe_allow_html=True)

m1,m2,m3,m4 = st.columns(4)
metric_card(m1, f"{bundle['auc_b']:.3f}", "Podium Model AUC", "#FFD700")
metric_card(m2, f"{bundle['auc_c']:.3f}", "Winner Model AUC", "#E8002D")
metric_card(m3, f"{bundle['rep_b'].get('Podium',{}).get('f1-score',0)*100:.1f}%" if bundle['rep_b'] else "N/A",
            "Podium F1-Score", "#C0C0C0")
metric_card(m4, f"{bundle['rep_c'].get('P1 Winner',{}).get('recall',0)*100:.1f}%" if bundle['rep_c'] else "N/A",
            "Winner Recall", "#00d2ff")

st.markdown("---")

# ── TABS ───────────────────────────────────────────────────────────────────────
tab_pred, tab_insight, tab_perf = st.tabs([
    "🏆  Predict",
    "📊  Podium Insights",
    "📈  Model Performance",
])

# ── TAB 1: PREDICT ─────────────────────────────────────────────────────────────
with tab_pred:
    inp   = np.array([[grid_pos, pace_z, team_pts, prev_rank,
                       rel_risk, circ_spec, gpd, roll_pace]])
    inp_s = scaler.transform(inp)

    # Run cascade
    pred_a = model_a.predict(inp_s)[0]
    prob_b = model_b.predict_proba(inp_s)[0]
    pred_b = model_b.predict(inp_s)[0]

    prob_c = None
    pred_c = None
    if pred_b == 1:
        prob_c = model_c.predict_proba(inp_s)[0]
        pred_c = model_c.predict(inp_s)[0]

    # Determine result
    if pred_b == 0:
        label  = "P4–P10 · Points, Not Podium"
        badge  = "badge-pts"
        conf   = prob_b[0] * 100
        color  = "#00d2ff"
        note   = "Model B: Pace and team form do not indicate a podium finish from this grid slot."
    elif pred_c == 0:
        label  = "P2 / P3 · Podium"
        badge  = "badge-pod"
        conf   = prob_c[0] * 100
        color  = "#C0C0C0"
        note   = "Model B: Podium ✓  ·  Model C: Not P1 — strong result but not the win."
    else:
        label  = "🏆  P1 · Race Winner"
        badge  = "badge-win"
        conf   = (prob_c[1] * 100) if prob_c is not None else 75.0
        color  = "#FFD700"
        note   = "Model B: Podium ✓  ·  Model C: Winner ✓"

    c1, c2 = st.columns([1, 1.5])

    with c1:
        st.markdown(f"""
        <div class="card" style="border-left:4px solid {color};
                                  text-align:center;padding:32px 20px;">
          <div style="font-size:.78rem;letter-spacing:.1em;text-transform:uppercase;
                      color:#7a7a90;margin-bottom:12px;">
            Prediction · Grid P{grid_pos}
          </div>
          <span class="{badge}">{label}</span>
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:3.5rem;
                      font-weight:800;color:{color};line-height:1;margin-top:18px;">
            {conf:.0f}%
          </div>
          <div style="font-size:.75rem;letter-spacing:.1em;text-transform:uppercase;
                      color:#7a7a90;margin-top:4px;">Confidence</div>
          <div style="font-size:.82rem;color:#9090b0;margin-top:14px;line-height:1.55;">
            {note}
          </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        # Podium vs Win probability gauge
        labels_g = ["P4-10\n(Points)", "P2/3\n(Podium)", "P1\n(Win)"]
        probs_g  = [
            prob_b[0] * 100,
            (prob_b[1] * prob_c[0] * 100) if prob_c is not None else prob_b[1] * 100,
            (prob_b[1] * prob_c[1] * 100) if prob_c is not None else 0,
        ]
        colors_g = ["#00d2ff", "#C0C0C0", "#FFD700"]
        fig = go.Figure()
        for lbl, p, clr in zip(labels_g, probs_g, colors_g):
            fig.add_trace(go.Bar(
                x=[lbl.replace("\n"," ")], y=[p],
                marker_color=clr,
                text=f"{p:.1f}%", textposition="outside",
                textfont=dict(color=clr, size=13, family="Barlow Condensed"),
                showlegend=False,
            ))
        fig.update_layout(
            title=dict(text="Outcome Probability Breakdown",
                       font=dict(color="#9090b0", size=13)),
            yaxis=dict(range=[0,115], showgrid=False, color="#555"),
            xaxis=dict(color="#555"),
            height=310, paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
            font=dict(family="Inter", size=11, color="#9090b0"),
            margin=dict(l=10,r=10,t=40,b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Decision trace
    st.markdown("#### Model Decision Trace")
    trace = [
        ("Model B — Podium Filter",
         prob_b[1]*100, prob_b[0]*100, pred_b==1,
         "→ Podium contender" if pred_b==1 else "→ Points finish (P4-10)"),
        ("Model C — Winner Specialist",
         (prob_c[1]*100) if prob_c is not None else None,
         (prob_c[0]*100) if prob_c is not None else None,
         pred_c==1 if pred_c is not None else None,
         ("→ Race Winner 🏆" if pred_c==1 else "→ P2/3 Podium")
         if pred_c is not None else "Not reached — not a podium contender"),
    ]
    for name, p_pos, p_neg, result, verdict in trace:
        if p_pos is None:
            clr, bg, pct = "#3a3a50", "#1a1a2a", "Not reached"
            bar = ""
        elif result:
            clr, bg = "#22C55E", "#0a1a10"
            pct = f"{p_pos:.1f}% → pass"
            bar = f'<div style="background:#22C55E;height:4px;width:{p_pos:.0f}%;border-radius:2px;margin-top:6px;max-width:100%;"></div>'
        else:
            clr, bg = "#FFD700", "#1a1500"
            pct = f"{p_neg:.1f}% → stop"
            bar = f'<div style="background:#FFD700;height:4px;width:{(p_neg or 0):.0f}%;border-radius:2px;margin-top:6px;max-width:100%;"></div>'

        st.markdown(f"""
        <div style="background:{bg};border-left:3px solid {clr};
                    border-radius:0 8px 8px 0;padding:10px 16px;margin-bottom:8px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-weight:600;color:#e0e0f0;font-size:.9rem;">{name}</span>
            <span style="font-size:.82rem;color:{clr};font-weight:600;">{pct}</span>
          </div>
          <div style="font-size:.8rem;color:#9090b0;margin-top:3px;">{verdict}</div>
          {bar}
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#1a1a2e;border-left:3px solid #FFD700;
                border-radius:0 8px 8px 0;padding:10px 16px;margin-top:10px;">
      <div style="font-size:.82rem;color:#9090b0;">
        <strong style="color:#FFD700;">Limitation:</strong>
        This tool assumes the driver is already predicted to finish in the top 10.
        It does not account for live race events (safety cars, crashes, weather).
        Use as a pre-race baseline — not a lap-by-lap oracle.
      </div>
    </div>""", unsafe_allow_html=True)


# ── TAB 2: PODIUM INSIGHTS ─────────────────────────────────────────────────────
with tab_insight:
    st.markdown("### Podium & Win Rate by Grid Position (Real Data 2010–2023)")

    pgrid = bundle["podium_by_grid"]
    fig_pg = go.Figure()
    fig_pg.add_trace(go.Scatter(
        x=pgrid["grid"], y=pgrid["Win Rate %"],
        name="Win Rate (P1)", line=dict(color="#FFD700", width=2.5),
        fill="tozeroy", fillcolor="rgba(255,215,0,0.08)"))
    fig_pg.add_trace(go.Scatter(
        x=pgrid["grid"], y=pgrid["Podium Rate %"],
        name="Podium Rate (P1-3)", line=dict(color="#C0C0C0", width=2.5),
        fill="tozeroy", fillcolor="rgba(192,192,192,0.05)"))
    fig_pg.update_layout(
        title="Actual Podium & Win Rate by Grid — F1 2010–2023",
        xaxis=dict(title="Grid Position", dtick=1, color="#555"),
        yaxis=dict(title="Rate (%)", color="#555", gridcolor="#2e2e40"),
        height=340, paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        font=dict(family="Inter", size=11, color="#9090b0"),
        legend=dict(bgcolor="#15151e", bordercolor="#2e2e40"),
        margin=dict(l=20,r=20,t=40,b=20),
    )
    st.plotly_chart(fig_pg, use_container_width=True)

    st.markdown("### Feature Importance — What Drives Podium vs Win Predictions")
    fi = bundle["fi"]
    fig_fi = px.bar(
        fi.sort_values("importance"),
        x="importance", y="feature", color="model", barmode="group",
        orientation="h",
        color_discrete_map={
            "Model B – Podium Filter":    "#C0C0C0",
            "Model C – Winner Specialist":"#FFD700",
        },
    )
    fig_fi.update_layout(
        height=380, paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        font=dict(family="Inter", size=11, color="#9090b0"),
        legend=dict(bgcolor="#15151e", bordercolor="#2e2e40"),
        margin=dict(l=20,r=20,t=20,b=20),
        yaxis=dict(color="#555"), xaxis=dict(color="#555", gridcolor="#2e2e40"),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("""
    <div style="background:#1e1e2e;border-left:3px solid #FFD700;
                border-radius:0 8px 8px 0;padding:12px 16px;margin-top:8px;">
      <div style="font-size:.88rem;color:#9090b0;">
        <strong style="color:#FFD700;">Key finding:</strong>
        pace_z_score dominates Model C (Winner) — pure speed separates P1 from P2/3.
        Model B (Podium) weights team form and grid position more equally,
        reflecting that reaching the podium is a team achievement as much as a driver one.
      </div>
    </div>""", unsafe_allow_html=True)


# ── TAB 3: PERFORMANCE ────────────────────────────────────────────────────────
with tab_perf:
    st.markdown("### Model B — Podium Filter · Classification Report")
    if bundle["rep_b"]:
        rows_b = []
        for cls, lbl in [("P4-10","P4-10 (Points, Not Podium)"), ("Podium","Podium (P1-3)")]:
            if cls in bundle["rep_b"]:
                r = bundle["rep_b"][cls]
                rows_b.append({"Class": lbl, "Precision": f"{r['precision']:.3f}",
                               "Recall": f"{r['recall']:.3f}", "F1": f"{r['f1-score']:.3f}",
                               "Support": int(r['support'])})
        st.dataframe(pd.DataFrame(rows_b), use_container_width=True, hide_index=True)

    st.markdown("### Model C — Winner Specialist · Classification Report")
    if bundle["rep_c"]:
        rows_c = []
        for cls, lbl in [("P2/3","P2/3 Podium"), ("P1 Winner","P1 Race Winner")]:
            if cls in bundle["rep_c"]:
                r = bundle["rep_c"][cls]
                rows_c.append({"Class": lbl, "Precision": f"{r['precision']:.3f}",
                               "Recall": f"{r['recall']:.3f}", "F1": f"{r['f1-score']:.3f}",
                               "Support": int(r['support'])})
        st.dataframe(pd.DataFrame(rows_c), use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="background:#1e1e2e;border-left:3px solid #C0C0C0;
                border-radius:0 8px 8px 0;padding:12px 16px;margin-top:8px;">
      <div style="font-size:.88rem;color:#9090b0;">
        <strong style="color:#C0C0C0;">Validation:</strong>
        Both models trained on 2010–2022, tested on the full 2023 season (held-out).
        Time-Series Split prevents any future race data from appearing in training.
        Winner Recall reflects the inherent difficulty of predicting rare events in motorsport.
      </div>
    </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#3a3a50;font-size:.78rem;padding:8px 0;">
  DATA 4382 – Capstone II · Ahnaf Harappa · Individual Deployment ·
  Models B & C: Podium Filter + Winner Specialist · Kaggle F1 Dataset (Rohan Rao, 2010–2023)
</div>""", unsafe_allow_html=True)
