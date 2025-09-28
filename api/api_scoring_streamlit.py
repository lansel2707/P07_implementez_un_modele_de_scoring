import streamlit as st
import requests
import pandas as pd
import os
import matplotlib.pyplot as plt

# =========================
# CONFIG APP
# =========================
st.set_page_config(page_title="Application de Scoring", layout="wide")
BASE_URL = "http://localhost:8000"  # API FastAPI locale

# =========================
# Features principales à saisir par l’utilisateur
# =========================
FEATURE_LABELS = {
    "AMT_INCOME_TOTAL": "Revenu total du client",
    "AMT_CREDIT": "Montant du crédit",
    "AMT_ANNUITY": "Annuité du prêt",
    "DAYS_BIRTH": "Âge du client (en jours)",
    "DAYS_EMPLOYED": "Ancienneté professionnelle (en jours)",
    "montant_en_retard": "Montant total en retard",
    "nb_paiements": "Nombre de paiements effectués",
    "taux_refus": "Taux de refus antérieurs"
}

IMPORTANT_FEATURES = list(FEATURE_LABELS.keys())

# =========================
# Toutes les features attendues par le modèle
# =========================
ALL_FEATURES = [
    "CNT_CHILDREN","AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE","DAYS_BIRTH","DAYS_EMPLOYED","DAYS_REGISTRATION","DAYS_ID_PUBLISH",
    "OWN_CAR_AGE","FLAG_MOBIL","FLAG_EMP_PHONE","FLAG_WORK_PHONE","FLAG_CONT_MOBILE","FLAG_PHONE","FLAG_EMAIL",
    "CNT_FAM_MEMBERS","REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY","HOUR_APPR_PROCESS_START",
    "REG_REGION_NOT_LIVE_REGION","REG_REGION_NOT_WORK_REGION","LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY","REG_CITY_NOT_WORK_CITY","LIVE_CITY_NOT_WORK_CITY",
    "EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3",
    "APARTMENTS_AVG","BASEMENTAREA_AVG","YEARS_BEGINEXPLUATATION_AVG","YEARS_BUILD_AVG",
    "COMMONAREA_AVG","ELEVATORS_AVG","ENTRANCES_AVG","FLOORSMAX_AVG","FLOORSMIN_AVG","LANDAREA_AVG",
    "LIVINGAPARTMENTS_AVG","LIVINGAREA_AVG","NONLIVINGAPARTMENTS_AVG","NONLIVINGAREA_AVG",
    "APARTMENTS_MODE","BASEMENTAREA_MODE","YEARS_BEGINEXPLUATATION_MODE","YEARS_BUILD_MODE",
    "COMMONAREA_MODE","ELEVATORS_MODE","ENTRANCES_MODE","FLOORSMAX_MODE","FLOORSMIN_MODE","LANDAREA_MODE",
    "LIVINGAPARTMENTS_MODE","LIVINGAREA_MODE","NONLIVINGAPARTMENTS_MODE","NONLIVINGAREA_MODE",
    "APARTMENTS_MEDI","BASEMENTAREA_MEDI","YEARS_BEGINEXPLUATATION_MEDI","YEARS_BUILD_MEDI",
    "COMMONAREA_MEDI","ELEVATORS_MEDI","ENTRANCES_MEDI","FLOORSMAX_MEDI","FLOORSMIN_MEDI","LANDAREA_MEDI",
    "LIVINGAPARTMENTS_MEDI","LIVINGAREA_MEDI","NONLIVINGAPARTMENTS_MEDI","NONLIVINGAREA_MEDI","TOTALAREA_MODE",
    "OBS_30_CNT_SOCIAL_CIRCLE","DEF_30_CNT_SOCIAL_CIRCLE","OBS_60_CNT_SOCIAL_CIRCLE","DEF_60_CNT_SOCIAL_CIRCLE",
    "DAYS_LAST_PHONE_CHANGE","FLAG_DOCUMENT_2","FLAG_DOCUMENT_3","FLAG_DOCUMENT_4","FLAG_DOCUMENT_5",
    "FLAG_DOCUMENT_6","FLAG_DOCUMENT_7","FLAG_DOCUMENT_8","FLAG_DOCUMENT_9","FLAG_DOCUMENT_10","FLAG_DOCUMENT_11",
    "FLAG_DOCUMENT_12","FLAG_DOCUMENT_13","FLAG_DOCUMENT_14","FLAG_DOCUMENT_15","FLAG_DOCUMENT_16","FLAG_DOCUMENT_17",
    "FLAG_DOCUMENT_18","FLAG_DOCUMENT_19","FLAG_DOCUMENT_20","FLAG_DOCUMENT_21",
    "AMT_REQ_CREDIT_BUREAU_HOUR","AMT_REQ_CREDIT_BUREAU_DAY","AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON","AMT_REQ_CREDIT_BUREAU_QRT","AMT_REQ_CREDIT_BUREAU_YEAR",
    "nb_bureau_credit","montant_total_credit_bureau","montant_credit_moyen_bureau",
    "montant_en_retard","nb_previous","taux_refus","montant_moyen_pret","nb_paiements",
    "retard_moyen","montant_paiement_moyen"
]

# =========================
# MENU
# =========================
menu = st.sidebar.radio("Navigation", [
    "🧑‍💻 Scoring Client",
    "📊 Feature Importance",
    "📑 Dashboard métier",
    "🔎 Drift (Evidently)"
])

# =========================
# PAGE 1 : Scoring
# =========================
if menu == "🧑‍💻 Scoring Client":
    st.title("👩‍💼 Application de Scoring Client")
    st.write("Entrez les informations principales du client pour obtenir un score de risque.")

    with st.form("form_scoring"):
        cols = st.columns(2)
        user_inputs = {}
        for i, (feat, label) in enumerate(FEATURE_LABELS.items()):
            with cols[i % 2]:
                user_inputs[feat] = st.number_input(label, value=0.0, step=1.0, format="%.2f")
        submitted = st.form_submit_button("🚀 Lancer le scoring")

    if submitted:
        payload = {"data": {feat: 0 for feat in ALL_FEATURES}}
        for feat, value in user_inputs.items():
            payload["data"][feat] = value


        try:
            response = requests.post(f"{BASE_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Résultat reçu depuis l'API")
                st.json(result)
            else:
                st.error(f"❌ Erreur API {response.status_code} : {response.text}")
        except Exception as e:
            st.error(f"❌ Impossible d'appeler l'API : {e}")
# =========================
# PAGE 2 : Feature Importance
# =========================
elif menu == "📊 Feature Importance":
    from pathlib import Path

    st.title("📊 Importance des variables")
    st.write("Affichage des importances calculées du modèle (Top 20).")

    # On remonte à la racine du projet puis on pointe vers notebooks/exports
    base_dir = Path(__file__).resolve().parent.parent
    path_importances = base_dir / "notebooks" / "exports" / "feature_importance_top20.csv"

    if path_importances.exists():
        df_importances = pd.read_csv(path_importances)

        # Harmonisation colonnes
        df_importances.columns = [c.lower() for c in df_importances.columns]

        # Trie et affichage
        df_top = df_importances.sort_values("importance", ascending=False).head(20)
        st.bar_chart(df_top.set_index("feature")["importance"])
        st.dataframe(df_top, use_container_width=True)
    else:
        st.warning(f"⚠️ Fichier des importances non trouvé : {path_importances}")

# =========================
# PAGE 3 : Dashboard métier
# =========================
elif menu == "📑 Dashboard métier":
    st.title("📑 Dashboard métier")
    st.write("Métriques globales du modèle et analyse des seuils de coûts.")

    metrics_path = os.path.join("notebooks", "exports", "metrics_model_final.csv")
    seuils_path = os.path.join("notebooks", "exports", "seuils_cout_gb.csv")

    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
        st.subheader("📊 Metrics du modèle final")
        st.dataframe(df_metrics)

    if os.path.exists(seuils_path):
        df_seuils = pd.read_csv(seuils_path)
        st.subheader("⚖️ Seuils de coûts optimisés")

        fig, ax = plt.subplots()
        if "seuil" in df_seuils.columns and "cout_total" in df_seuils.columns:
            ax.plot(df_seuils["seuil"], df_seuils["cout_total"])
            ax.set_xlabel("Seuil")
            ax.set_ylabel("Coût total")
            ax.set_title("Coût total en fonction du seuil")
            st.pyplot(fig)

        st.dataframe(df_seuils)

# =========================
# PAGE 4 : Drift Evidently
# =========================
elif menu == "🔎 Drift (Evidently)":
    st.title("🔎 Détection de drift avec Evidently")
    drift_report_path = os.path.join("monitoring", "reports", "drift_report.html")

    if os.path.exists(drift_report_path):
        st.components.v1.html(open(drift_report_path, "r").read(), height=800, scrolling=True)
    else:
        st.warning("⚠️ Rapport Evidently non trouvé.")
