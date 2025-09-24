import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# ===============================
# CONFIG APP
# ===============================
st.set_page_config(page_title="Application de Scoring", layout="wide")

# ===============================
# CHARGEMENT DU MODÈLE CALIBRÉ
# ===============================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "gbc_all_final_calibrated.pkl")
MODEL_PATH = os.path.abspath(MODEL_PATH)

model = joblib.load(MODEL_PATH)

# ===============================
# FEATURES PRINCIPALES À SAISIR
# ===============================
FEATURE_LABELS = {
    "AMT_INCOME_TOTAL": "Revenu total du client",
    "AMT_CREDIT": "Montant du crédit",
    "AMT_ANNUITY": "Annuité du prêt",
    "DAYS_BIRTH": "Âge du client (en années)",
    "DAYS_EMPLOYED": "Ancienneté professionnelle (en années)",
    "montant_en_retard": "Montant total en retard",
    "nb_paiements": "Nombre de paiements effectués",
    "taux_refus": "Taux de refus antérieurs"
}
IMPORTANT_FEATURES = list(FEATURE_LABELS.keys())

# ===============================
# IMPORT DES FICHIERS EXPORTÉS
# ===============================
EXPORT_DIR = "notebooks/exports"
feature_importance_path = os.path.join(EXPORT_DIR, "feature_importance_top20.csv")
metrics_model_path = os.path.join(EXPORT_DIR, "metrics_model_final.csv")
seuils_cout_path = os.path.join(EXPORT_DIR, "seuils_cout_gb.csv")

df_importance = pd.read_csv(feature_importance_path) if os.path.exists(feature_importance_path) else None
df_metrics = pd.read_csv(metrics_model_path) if os.path.exists(metrics_model_path) else None
df_seuils = pd.read_csv(seuils_cout_path) if os.path.exists(seuils_cout_path) else None

# ===============================
# MENU
# ===============================
menu = st.sidebar.radio("Navigation", [
    "👤 Scoring Client",
    "📊 Feature Importance",
    "📑 Dashboard métier",
    "🔍 Drift (Evidently)"
])

# ===============================
# PAGE SCORING CLIENT
# ===============================
if menu == "👤 Scoring Client":
    st.title("👤 Application de Scoring Client")

    # Inputs utilisateur
    inputs = {}
    for feature, label in FEATURE_LABELS.items():
        if "années" in label:
            val = st.number_input(label, min_value=0, step=1)
            if feature == "DAYS_BIRTH":
                inputs[feature] = -val * 365   # âge en années → jours négatifs
            elif feature == "DAYS_EMPLOYED":
                inputs[feature] = -val * 365   # ancienneté en années → jours négatifs
        else:
            val = st.number_input(label, min_value=0.0, step=100.0)
            inputs[feature] = val

    if st.button("🚀 Lancer le scoring"):
        # DataFrame au bon format
        X_input = pd.DataFrame([inputs])

        # Prédiction avec le modèle calibré
        proba = model.predict_proba(X_input)[0, 1]
        prediction = int(proba > 0.5)

        st.success("✅ Résultat du scoring")
        st.metric("Probabilité d'être mauvais payeur", f"{proba:.2%}")
        st.write("Prédiction :", "❌ Risqué" if prediction == 1 else "✔️ Fiable")

# ===============================
# PAGE FEATURE IMPORTANCE
# ===============================
elif menu == "📊 Feature Importance":
    st.title("📊 Importance des variables")

    if df_importance is not None:
        df_sorted = df_importance.sort_values(by="importance", ascending=False)
        st.bar_chart(df_sorted.set_index("feature"))
        st.dataframe(df_sorted)
    else:
        st.warning("Aucune donnée d’importance disponible.")

# ===============================
# PAGE DASHBOARD MÉTIER
# ===============================
elif menu == "📑 Dashboard métier":
    st.title("📑 Dashboard métier")
    if df_metrics is not None:
        st.subheader("📈 Metrics du modèle final")
        st.dataframe(df_metrics)

    if df_seuils is not None:
        st.subheader("⚖️ Optimisation du seuil de décision")
        fig, ax = plt.subplots()
        ax.plot(df_seuils["Seuil"], df_seuils["Cout_total"], label="Coût total métier")
        ax.set_xlabel("Seuil")
        ax.set_ylabel("Coût total (FN*6 + FP*1)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Pas de données sur les seuils de coûts.")

# ===============================
# PAGE DRIFT
# ===============================
elif menu == "🔍 Drift (Evidently)":
    st.title("🔍 Monitoring Drift")
    st.markdown("👉 Le drift est suivi via **Evidently**. Ouvre les rapports HTML générés dans `monitoring/reports/`.")


