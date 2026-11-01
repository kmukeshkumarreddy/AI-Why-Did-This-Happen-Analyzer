import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    import shap
except Exception:
    shap = None
try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

st.set_page_config(page_title="Why Did This Happen? Analyzer", page_icon="🔍", layout="wide")

st.markdown("""
<style>
.hero{background:linear-gradient(120deg,#4f46e5,#7c3aed 55%,#db2777);padding:28px 32px;border-radius:18px;color:#fff;margin-bottom:18px}
.hero h1{margin:0;font-size:2.1rem;color:#fff}
.hero p{margin:6px 0 0;opacity:.92;font-size:1.02rem}
.card{background:#f8f7ff;border:1px solid #e4e0ff;border-left:6px solid #7c3aed;border-radius:12px;padding:14px 18px;margin:8px 0;color:#1f1b3a}
.stTabs [data-baseweb="tab"]{font-weight:600}
</style>
<div class="hero"><h1>🔍 AI "Why Did This Happen?" Analyzer</h1>
<p>Don't just predict the outcome — discover what drove it. Statistics + Machine Learning + Explainable AI.</p></div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data
def demo_data(n=2000, seed=42):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({
        "age": r.integers(18, 75, n),
        "subscription_months": r.integers(1, 72, n),
        "monthly_charges": r.normal(70, 25, n).clip(20, 150).round(2),
        "complaints": r.poisson(1.2, n),
        "usage_hours": r.normal(40, 15, n).clip(1, 120).round(1),
        "satisfaction": r.integers(1, 11, n),
        "service_type": r.choice(["Basic", "Standard", "Premium"], n, p=[.4, .4, .2]),
        "contract": r.choice(["Monthly", "Yearly"], n, p=[.65, .35]),
    })
    z = (-1.2 + 0.55 * d.complaints + 0.018 * (d.monthly_charges - 70)
         - 0.03 * d.subscription_months - 0.32 * (d.satisfaction - 5)
         + 0.9 * (d.contract == "Monthly") - 0.012 * (d.usage_hours - 40))
    d["churn"] = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return d


def prepare(df, target):
    y_raw = df[target]
    y = pd.Series(LabelEncoder().fit_transform(y_raw.astype(str)), index=df.index)
    X = df.drop(columns=[target]).copy()
    X = X.loc[:, X.nunique() > 1]
    for c in X.columns:
        if X[c].dtype == "O" or X[c].dtype.name == "category":
            X[c] = X[c].fillna("Unknown")
        else:
            X[c] = X[c].fillna(X[c].median())
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    X = X.drop(columns=[c for c in cat if X[c].nunique() > 20])
    cat = [c for c in cat if c in X.columns]
    Xe = pd.get_dummies(X, columns=cat, drop_first=True).astype(float)
    return X, Xe, y, num, cat


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Setup")
    up = st.file_uploader("Upload your CSV", type=["csv"])
    df = pd.read_csv(up) if up else demo_data()
    st.caption("Using your file" if up else "Using built-in demo: customer churn data")
    target = st.selectbox("Outcome to explain", df.columns,
                          index=list(df.columns).index("churn") if "churn" in df.columns else len(df.columns) - 1)
    alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)
    model_name = st.radio("Model", ["Random Forest"] + (["XGBoost"] if XGBClassifier else []))
    st.markdown("---")
    st.caption("Built with Streamlit · scikit-learn · SHAP · LIME · SciPy")

if df[target].nunique() > 10:
    st.error("Pick a target with 10 or fewer classes (e.g. churn yes/no).")
    st.stop()

X, Xe, y, num, cat = prepare(df, target)
classes = sorted(y.unique())
positive = classes[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Features", X.shape[1])
c3.metric("Outcome classes", len(classes))
c4.metric("Positive-outcome rate", f"{(y == positive).mean():.1%}")

tabs = st.tabs(["📊 Overview", "🔗 Relationships", "🧪 Hypothesis Tests", "🤖 Model + SHAP",
                "🍋 LIME (single case)", "🧭 PCA & Anomalies", "💡 The Verdict"])


# ---------------------------------------------------------------- stats (shared)
@st.cache_data
def run_stats(X, y, num, cat):
    rows = []
    for c in num:
        groups = [X.loc[y == k, c] for k in sorted(y.unique())]
        if len(groups) == 2:
            _, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            test = "t-test"
        else:
            _, p = stats.f_oneway(*groups)
            test = "ANOVA"
        rho, _ = stats.spearmanr(X[c], y)
        rows.append([c, "numeric", test, p, abs(rho)])
    for c in cat:
        _, p, _, _ = stats.chi2_contingency(pd.crosstab(X[c], y))
        n = len(X)
        chi2 = stats.chi2_contingency(pd.crosstab(X[c], y))[0]
        k = min(pd.crosstab(X[c], y).shape) - 1
        v = np.sqrt(chi2 / (n * max(k, 1)))
        rows.append([c, "categorical", "Chi-square", p, v])
    out = pd.DataFrame(rows, columns=["feature", "type", "test", "p_value", "effect_size"])
    return out.sort_values("effect_size", ascending=False).reset_index(drop=True)


st_tbl = run_stats(X, y, num, cat)

# ---------------------------------------------------------------- tab 1
with tabs[0]:
    st.subheader("Dataset preview")
    st.dataframe(df.head(50), use_container_width=True)
    a, b = st.columns(2)
    with a:
        vc = df[target].value_counts().reset_index()
        vc.columns = [target, "count"]
        st.plotly_chart(px.pie(vc, names=target, values="count", hole=.5, title="Outcome distribution",
                               color_discrete_sequence=px.colors.sequential.Purples_r), use_container_width=True)
    with b:
        if num:
            f = st.selectbox("Feature distribution by outcome", num)
            st.plotly_chart(px.histogram(df, x=f, color=df[target].astype(str), barmode="overlay",
                                         opacity=.65, title=f"{f} by {target}"), use_container_width=True)

# ---------------------------------------------------------------- tab 2
with tabs[1]:
    st.subheader("Pearson / Spearman correlation")
    if len(num) >= 2:
        method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
        tmp = X[num].copy()
        tmp["_outcome"] = y
        corr = tmp.corr(method=method)
        st.plotly_chart(px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, aspect="auto"), use_container_width=True)
        s = corr["_outcome"].drop("_outcome").sort_values(key=abs, ascending=False)
        st.plotly_chart(px.bar(s[::-1], orientation="h", title=f"{method.title()} correlation with outcome",
                               color=s[::-1], color_continuous_scale="RdBu_r"), use_container_width=True)
    else:
        st.info("Need at least 2 numeric columns.")

# ---------------------------------------------------------------- tab 3
with tabs[2]:
    st.subheader("Is each difference real, or just chance?")
    t = st_tbl.copy()
    t["significant"] = np.where(t.p_value < alpha, "✅ Yes", "❌ No")
    t["p_value"] = t.p_value.map(lambda v: f"{v:.2e}")
    t["effect_size"] = t.effect_size.round(3)
    st.dataframe(t, use_container_width=True, hide_index=True)
    st.caption("t-test/ANOVA for numeric features, Chi-square for categorical ones. "
               "Effect size = |Spearman ρ| (numeric) or Cramér's V (categorical).")
    st.plotly_chart(px.bar(st_tbl.iloc[::-1], x="effect_size", y="feature", orientation="h",
                           color="p_value", color_continuous_scale="Purples_r",
                           title="Strength of association with outcome"), use_container_width=True)

# ---------------------------------------------------------------- tab 4
Xtr, Xte, ytr, yte = train_test_split(Xe, y, test_size=.25, random_state=42,
                                      stratify=y if y.value_counts().min() > 1 else None)
if model_name == "XGBoost":
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=.08, eval_metric="logloss")
else:
    model = RandomForestClassifier(n_estimators=250, max_depth=8, random_state=42, n_jobs=-1)
model.fit(Xtr, ytr)
pred, proba = model.predict(Xte), model.predict_proba(Xte)


@st.cache_data
def global_importance(_model, Xs, use_shap):
    if use_shap and shap is not None:
        try:
            sv = shap.TreeExplainer(_model).shap_values(Xs)
            if isinstance(sv, list):
                sv = sv[-1]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, -1]
            return pd.Series(np.abs(sv).mean(0), index=Xs.columns), "SHAP", sv
        except Exception:
            pass
    pi = permutation_importance(_model, Xs, yte.loc[Xs.index], n_repeats=5, random_state=0)
    return pd.Series(pi.importances_mean, index=Xs.columns), "Permutation", None


with tabs[3]:
    st.subheader(f"{model_name} performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{accuracy_score(yte, pred):.3f}")
    m2.metric("F1 score", f"{f1_score(yte, pred, average='weighted'):.3f}")
    try:
        auc = roc_auc_score(yte, proba[:, 1]) if len(classes) == 2 else roc_auc_score(yte, proba, multi_class="ovr")
        m3.metric("ROC-AUC", f"{auc:.3f}")
    except Exception:
        m3.metric("ROC-AUC", "n/a")

    Xs = Xte.sample(min(300, len(Xte)), random_state=1)
    imp, kind, sv = global_importance(model, Xs, True)
    imp = imp.sort_values(ascending=False)
    st.subheader(f"Global feature importance ({kind})")
    top = imp.head(15)[::-1]
    st.plotly_chart(px.bar(top, orientation="h", color=top, color_continuous_scale="Purples",
                           labels={"value": "mean |impact|", "index": "feature"}), use_container_width=True)
    if sv is not None:
        st.subheader("SHAP: how feature values push the outcome")
        f = st.selectbox("Feature", imp.index[:10])
        j = list(Xs.columns).index(f)
        st.plotly_chart(px.scatter(x=Xs[f], y=sv[:, j], labels={"x": f, "y": "SHAP value (push toward outcome)"},
                                   color=sv[:, j], color_continuous_scale="RdBu_r"), use_container_width=True)
    else:
        st.info("Install `shap` to see SHAP dependence plots.")

# ---------------------------------------------------------------- tab 5
with tabs[4]:
    st.subheader("Why did THIS specific case happen?")
    if LimeTabularExplainer is None:
        st.info("Install `lime` to enable this tab.")
    else:
        idx = st.number_input("Test-set row", 0, len(Xte) - 1, 0)
        exp = LimeTabularExplainer(Xtr.values, feature_names=list(Xtr.columns), mode="classification",
                                   class_names=[str(c) for c in classes], random_state=1)
        e = exp.explain_instance(Xte.iloc[idx].values, model.predict_proba, num_features=8)
        lab = list(e.local_exp.keys())[-1]
        w = pd.DataFrame(e.as_list(label=lab), columns=["condition", "weight"])
        w["direction"] = np.where(w.weight > 0, "pushes toward outcome", "pushes away")
        pp = model.predict_proba(Xte.iloc[[idx]])[0]
        st.markdown(f'<div class="card">Predicted probability of outcome <b>{classes[-1]}</b>: '
                    f'<b>{pp[-1]:.1%}</b> · Actual: <b>{yte.iloc[idx]}</b></div>', unsafe_allow_html=True)
        st.plotly_chart(px.bar(w[::-1], x="weight", y="condition", orientation="h", color="direction",
                               color_discrete_map={"pushes toward outcome": "#dc2626", "pushes away": "#2563eb"}),
                        use_container_width=True)
        st.dataframe(Xte.iloc[[idx]].T.rename(columns={Xte.index[idx]: "value"}), use_container_width=True)

# ---------------------------------------------------------------- tab 6
with tabs[5]:
    a, b = st.columns(2)
    with a:
        st.subheader("PCA (dimensionality analysis)")
        Z = StandardScaler().fit_transform(Xe)
        p = PCA(n_components=min(5, Z.shape[1])).fit(Z)
        comp = p.transform(Z)
        pc = pd.DataFrame(comp[:, :2], columns=["PC1", "PC2"])
        pc["outcome"] = y.astype(str).values
        st.plotly_chart(px.scatter(pc, x="PC1", y="PC2", color="outcome", opacity=.6), use_container_width=True)
        st.plotly_chart(px.bar(x=[f"PC{i+1}" for i in range(len(p.explained_variance_ratio_))],
                               y=p.explained_variance_ratio_, labels={"x": "", "y": "variance explained"}),
                        use_container_width=True)
    with b:
        st.subheader("Anomaly detection (Isolation Forest)")
        cont = st.slider("Expected anomaly share", .01, .20, .05, .01)
        iso = IsolationForest(contamination=cont, random_state=0).fit(Z)
        flag = iso.predict(Z) == -1
        pc["anomaly"] = np.where(flag, "Anomaly", "Normal")
        st.plotly_chart(px.scatter(pc, x="PC1", y="PC2", color="anomaly",
                                   color_discrete_map={"Anomaly": "#dc2626", "Normal": "#c4b5fd"}),
                        use_container_width=True)
        r_an, r_no = (y[flag] == positive).mean(), (y[~flag] == positive).mean()
        st.markdown(f'<div class="card">Outcome rate among anomalies: <b>{r_an:.1%}</b> vs normal records: '
                    f'<b>{r_no:.1%}</b>.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- tab 7
with tabs[6]:
    st.subheader("💡 Plain-English explanation")
    drivers = imp.index.tolist()
    base_names, seen = [], set()
    for d in drivers:  # map one-hot columns back to original feature
        orig = next((c for c in X.columns if d == c or d.startswith(c + "_")), d)
        if orig not in seen:
            seen.add(orig)
            base_names.append(orig)
    st.markdown(f"The outcome **{target} = {positive}** occurs in **{(y == positive).mean():.1%}** of records. "
                f"The model ({model_name}, AUC-validated) says the biggest drivers are:")
    for i, f in enumerate(base_names[:5], 1):
        row = st_tbl[st_tbl.feature == f]
        sig = ""
        if len(row):
            pv = row.p_value.iloc[0]
            sig = f" — statistically {'significant' if pv < alpha else 'not significant'} (p={pv:.1e})"
        if f in num:
            hi, lo = X.loc[y == positive, f].mean(), X.loc[y != positive, f].mean()
            word = "higher" if hi > lo else "lower"
            txt = (f"cases with the outcome have a **{word}** average {f} "
                   f"({hi:.1f} vs {lo:.1f} for others)")
        else:
            g = pd.crosstab(X[f], y, normalize="index")[positive].sort_values(ascending=False)
            txt = f"outcome rate is highest for **{g.index[0]}** ({g.iloc[0]:.1%}) and lowest for **{g.index[-1]}** ({g.iloc[-1]:.1%})"
        st.markdown(f'<div class="card"><b>#{i} {f}</b>: {txt}{sig}.</div>', unsafe_allow_html=True)
    st.caption("Association ≠ causation: these are the factors most strongly linked to the outcome in your data; "
               "confirm with domain knowledge or an experiment before acting.")
