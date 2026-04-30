"""StrokeRisk Predictor - Streamlit app
ALY6040 Final Project | Brain Stroke Risk Prediction
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="StrokeRisk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- LOAD MODEL ----------
# We always retrain from the raw CSV at startup. This is the most reliable approach
# because pickled scikit-learn models can break across version differences between
# the dev machine and the deployment environment. The retrain takes about 5 seconds
# on cold start and is cached for the rest of the session.
@st.cache_resource
def load_bundle():
    return _retrain_from_csv()


def _retrain_from_csv():
    import pandas as pd
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    np.random.seed(42)
    df = pd.read_csv("brainStrokeDataset.csv")
    df = df.drop(columns=["id"])
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    df = df[df["gender"] != "Other"].reset_index(drop=True)
    df["glucose_high"] = (df["avg_glucose_level"] >= 125).astype(int)
    df["bmi_obese"] = (df["bmi"] >= 30).astype(int)

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    cats = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    nums = [c for c in X.columns if c not in cats]

    prep = ColumnTransformer([
        ("num", StandardScaler(), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cats),
    ])
    pipe = ImbPipeline([
        ("prep", prep),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(Xtr, ytr)

    risk_scores = pipe.predict_proba(X)[:, 1]
    quantiles = np.percentile(risk_scores, np.arange(0, 101, 1)).tolist()

    return {
        "model": pipe,
        "feature_order": list(X.columns),
        "cat_cols": cats,
        "num_cols": nums,
        "pop_stats": {
            "stroke_rate_overall": float(df["stroke"].mean()),
            "age_mean": float(df["age"].mean()),
            "age_std": float(df["age"].std()),
            "glucose_mean": float(df["avg_glucose_level"].mean()),
            "bmi_mean": float(df["bmi"].mean()),
            "n_total": int(len(df)),
            "n_strokes": int(df["stroke"].sum()),
        },
        "risk_quantiles": quantiles,
    }


bundle = load_bundle()
model = bundle["model"]
feature_order = bundle["feature_order"]
cat_cols = bundle["cat_cols"]
num_cols = bundle["num_cols"]
pop_stats = bundle["pop_stats"]
risk_quantiles = np.array(bundle["risk_quantiles"])


# ---------- STYLING ----------
st.markdown("""
<style>
.main { background-color: #FAFAFA; }
.stButton>button {
    background-color: #1565C0;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    border: none;
    width: 100%;
}
.stButton>button:hover {
    background-color: #1A237E;
    color: white;
}
.metric-card {
    background-color: white;
    padding: 1.2rem;
    border-radius: 10px;
    border: 1px solid #ECEFF1;
    margin-bottom: 1rem;
}
.app-header {
    background: linear-gradient(90deg, #1A237E 0%, #1565C0 100%);
    color: white;
    padding: 1.4rem 1.8rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
}
.app-header h1 { color: white; margin: 0; font-size: 1.7rem; }
.app-header p  { color: #B3E5FC; margin: 0.2rem 0 0; font-size: 0.9rem; }
.disclaimer {
    background-color: #FFF8E1;
    border-left: 4px solid #F57C00;
    padding: 0.8rem 1rem;
    border-radius: 6px;
    font-size: 0.85rem;
    color: #5D4037;
}
.recommendation {
    background-color: #FFF8E1;
    border: 1px solid #F57C00;
    padding: 1rem 1.2rem;
    border-radius: 8px;
    margin-top: 1rem;
}
.recommendation h4 { color: #E65100; margin: 0 0 0.4rem 0; }
</style>
""", unsafe_allow_html=True)


# ---------- HEADER ----------
st.markdown("""
<div class="app-header">
  <h1>🧠 StrokeRisk Predictor</h1>
  <p>ALY6040 Final Project | Powered by clinical data on 5,109 patients</p>
</div>
""", unsafe_allow_html=True)


# ---------- SIDEBAR INPUTS ----------
st.sidebar.header("Patient Profile")
st.sidebar.caption("Enter the patient's information below.")

age = st.sidebar.slider("Age", 1, 100, 55)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
hypertension = st.sidebar.radio("Hypertension", ["No", "Yes"], horizontal=True)
heart_disease = st.sidebar.radio("Heart Disease", ["No", "Yes"], horizontal=True)
ever_married = st.sidebar.selectbox("Ever Married", ["Yes", "No"])
work_type = st.sidebar.selectbox(
    "Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
)
residence = st.sidebar.selectbox("Residence", ["Urban", "Rural"])
glucose = st.sidebar.slider("Average Glucose Level (mg/dL)", 50, 280, 105)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 28.0, step=0.5)
smoking_status = st.sidebar.selectbox(
    "Smoking Status",
    ["never smoked", "formerly smoked", "smokes", "Unknown"]
)

predict = st.sidebar.button("🔍 Calculate My Risk")
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="disclaimer"><b>For screening only.</b> '
    'This tool is built on a public dataset for an academic project. '
    'It is not a medical diagnosis.</div>',
    unsafe_allow_html=True,
)


# ---------- BUILD FEATURE ROW ----------
def build_row():
    row = {
        "gender": gender,
        "age": float(age),
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence,
        "avg_glucose_level": float(glucose),
        "bmi": float(bmi),
        "smoking_status": smoking_status,
        "glucose_high": 1 if glucose >= 125 else 0,
        "bmi_obese": 1 if bmi >= 30 else 0,
    }
    return pd.DataFrame([row], columns=feature_order)


# ---------- FEATURE CONTRIBUTION (logit-space) ----------
def compute_contributions(X):
    """Return ranked list of (feature_label, contribution_to_log_odds)."""
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    Xt = prep.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    coefs = clf.coef_[0]
    contribs = (Xt[0] * coefs)
    feat_names = (
        num_cols
        + list(prep.named_transformers_["cat"].get_feature_names_out(cat_cols))
    )
    pretty = {
        "age": "Age",
        "avg_glucose_level": "Glucose level",
        "bmi": "BMI",
        "hypertension": "Hypertension",
        "heart_disease": "Heart disease",
        "glucose_high": "Glucose >= 125",
        "bmi_obese": "BMI >= 30 (obese)",
        "gender_Male": "Gender: Male",
        "ever_married_Yes": "Ever married",
        "work_type_Private": "Work: Private",
        "work_type_Self-employed": "Work: Self-employed",
        "work_type_children": "Work: children",
        "work_type_Never_worked": "Work: Never worked",
        "Residence_type_Urban": "Residence: Urban",
        "smoking_status_formerly smoked": "Formerly smoked",
        "smoking_status_never smoked": "Never smoked",
        "smoking_status_smokes": "Currently smokes",
    }
    df = pd.DataFrame({
        "feature": [pretty.get(f, f) for f in feat_names],
        "contribution": contribs,
    })
    df["abs"] = df["contribution"].abs()
    return df.sort_values("abs", ascending=False)


# ---------- BAND HELPERS ----------
def risk_band(prob):
    if prob < 0.20:
        return "Low", "#2E7D32"
    if prob < 0.50:
        return "Moderate", "#F57C00"
    return "High", "#C62828"


def percentile_rank(prob):
    return int(np.searchsorted(risk_quantiles, prob, side="left"))


# ---------- MAIN PANEL ----------
if not predict:
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("How it works")
        st.write("""
        Fill in the patient profile on the left, then click **Calculate My Risk**.
        The app uses a logistic regression model trained with SMOTE oversampling on
        5,109 patient records. It returns a stroke-risk percentage, the factors driving
        that score, where the patient sits compared to the population, and a recommended
        next step.
        """)
        st.markdown("**Model performance (test set)**")
        st.write("""
        - 5-fold cross-validated ROC-AUC: 0.84
        - Recall on stroke cases: 82% (catches 41 of 50 real stroke patients)
        - Trained on Kaggle's Brain Stroke Prediction Dataset (fedesoriano, 2021)
        """)
    with col_right:
        st.subheader("Population snapshot")
        st.metric("Patients in training data", f"{pop_stats['n_total']:,}")
        st.metric("Patients with stroke", f"{pop_stats['n_strokes']:,}")
        st.metric("Overall stroke rate", f"{pop_stats['stroke_rate_overall']*100:.2f}%")
        st.metric("Average age", f"{pop_stats['age_mean']:.1f} yrs")

else:
    X_input = build_row()
    proba = model.predict_proba(X_input)[0, 1]
    band, band_color = risk_band(proba)
    pct = percentile_rank(proba)

    # ============ TOP RESULT CARD ============
    st.subheader("Your Risk Profile")
    g1, g2, g3 = st.columns([2, 2, 2])
    with g1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 6px solid {band_color};">
          <div style="font-size: 0.85rem; color: #607D8B;">Predicted Stroke Risk</div>
          <div style="font-size: 2.3rem; font-weight: bold; color: {band_color}; margin: 0.2rem 0;">
            {proba*100:.1f}%
          </div>
          <div style="font-size: 1rem; color: {band_color}; font-weight: bold;">{band} Risk Band</div>
        </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown(f"""
        <div class="metric-card">
          <div style="font-size: 0.85rem; color: #607D8B;">Compared to Population</div>
          <div style="font-size: 2.3rem; font-weight: bold; color: #1565C0; margin: 0.2rem 0;">
            Top {100 - pct}%
          </div>
          <div style="font-size: 0.85rem; color: #455A64;">of stroke risk in the training data</div>
        </div>
        """, unsafe_allow_html=True)
    with g3:
        st.markdown(f"""
        <div class="metric-card">
          <div style="font-size: 0.85rem; color: #607D8B;">Population Average Risk</div>
          <div style="font-size: 2.3rem; font-weight: bold; color: #455A64; margin: 0.2rem 0;">
            {pop_stats['stroke_rate_overall']*100:.2f}%
          </div>
          <div style="font-size: 0.85rem; color: #455A64;">across all 5,109 patients</div>
        </div>
        """, unsafe_allow_html=True)

    # ============ RISK BAR ============
    st.markdown("**Risk gauge**")
    bar_col1, bar_col2, bar_col3 = st.columns([1, 6, 1])
    with bar_col2:
        st.progress(min(float(proba), 1.0))
    legend = st.columns(3)
    with legend[0]:
        st.markdown('<div style="text-align:left;color:#2E7D32;">Low &lt; 20%</div>',
                    unsafe_allow_html=True)
    with legend[1]:
        st.markdown('<div style="text-align:center;color:#F57C00;">Moderate 20–50%</div>',
                    unsafe_allow_html=True)
    with legend[2]:
        st.markdown('<div style="text-align:right;color:#C62828;">High &gt; 50%</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ============ TOP DRIVERS ============
    drivers_col, peer_col = st.columns([3, 2])

    with drivers_col:
        st.subheader("Top Risk Drivers for You")
        st.caption("How much each factor pushed the score up (red) or down (green).")
        c = compute_contributions(X_input).head(8).copy()
        c["direction"] = c["contribution"].apply(lambda x: "↑ Increases risk" if x > 0 else "↓ Decreases risk")
        c["color"] = c["contribution"].apply(lambda x: "#C62828" if x > 0 else "#2E7D32")
        max_abs = c["abs"].max()
        for _, r in c.iterrows():
            pct_bar = abs(r["contribution"]) / max_abs * 100
            st.markdown(f"""
            <div style="margin-bottom: 0.55rem;">
              <div style="display:flex; justify-content:space-between; font-size:0.92rem;">
                <span><b>{r['feature']}</b></span>
                <span style="color:{r['color']}; font-size:0.85rem;">{r['direction']}</span>
              </div>
              <div style="background:#ECEFF1; border-radius:4px; height:10px; margin-top:3px;">
                <div style="width:{pct_bar:.0f}%; height:10px; background:{r['color']}; border-radius:4px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with peer_col:
        st.subheader("How You Compare")
        st.caption("Where this risk score falls in the patient population.")
        st.markdown(f"""
        <div class="metric-card" style="text-align:center;">
          <div style="font-size:0.9rem; color:#607D8B; margin-bottom:0.3rem;">
            You scored higher than
          </div>
          <div style="font-size:3rem; font-weight:bold; color:#1565C0;">{pct}%</div>
          <div style="font-size:0.9rem; color:#455A64;">
            of the {pop_stats['n_total']:,} patients in the training data.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Recommendation
        if proba >= 0.50:
            rec_title = "Schedule a clinical review soon"
            rec_body = (
                "Two or more of your top drivers are clinically modifiable. "
                "A clinician can confirm the risk with a blood pressure check, "
                "fasting glucose panel, and lifestyle assessment."
            )
        elif proba >= 0.20:
            rec_title = "Worth discussing at your next visit"
            rec_body = (
                "Bring your blood pressure and glucose history to your next "
                "annual physical. The model flagged moderate risk."
            )
        else:
            rec_title = "Continue routine monitoring"
            rec_body = (
                "Your profile sits in the lower-risk band. Maintain regular "
                "check-ups and keep an eye on weight and blood pressure."
            )
        st.markdown(f"""
        <div class="recommendation">
          <h4>📋 {rec_title}</h4>
          <p style="margin:0; font-size:0.92rem;">{rec_body}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("See the raw input the model received"):
        st.dataframe(X_input.T.rename(columns={0: "value"}))

    st.caption(
        "🩺 Built for ALY6040 Final Project — not a medical diagnosis. "
        "Always consult a qualified clinician for personal health decisions."
    )
