import requests
import json
import pandas as pd
import streamlit as st

# === URL de ton API FastAPI ===
BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Application de Scoring Client", layout="wide")
st.title("📊 Application de Scoring Client")
st.write("Entrez les informations principales du client pour obtenir un score de risque.")

# === Liste complète des features attendues par FastAPI ===
ALL_FEATURES = [
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH", "OWN_CAR_AGE", "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL", "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY", "HOUR_APPR_PROCESS_START",
    "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "APARTMENTS_AVG", "BASEMENTAREA_AVG",
    "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG",
    "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG", "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG",
    "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAREA_AVG", "APARTMENTS_MODE",
    "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BUILD_MODE", "COMMONAREA_MODE",
    "ELEVATORS_MODE", "ENTRANCES_MODE", "FLOORSMAX_MODE", "FLOORSMIN_MODE", "LANDAREA_MODE",
    "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE", "NONLIVINGAPARTMENTS_MODE",
    "NONLIVINGAREA_MODE", "APARTMENTS_MEDI", "BASEMENTAREA_MEDI", "YEARS_BEGINEXPLUATATION_MEDI",
    "YEARS_BUILD_MEDI", "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI", "FLOORSMAX_MEDI",
    "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI", "LIVINGAREA_MEDI",
    "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI", "TOTALAREA_MODE",
    "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE", "DAYS_LAST_PHONE_CHANGE", "FLAG_DOCUMENT_2",
    "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6",
    "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13", "FLAG_DOCUMENT_14",
    "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16", "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18",
    "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21", "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR", "nb_bureau_credit",
    "montant_total_credit_bureau", "montant_credit_moyen_bureau", "montant_en_retard",
    "nb_previous", "taux_refus", "montant_moyen_pret", "nb_paiements", "retard_moyen",
    "montant_paiement_moyen"
]

# === Features importantes à afficher dans le formulaire ===
IMPORTANT_FEATURES = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH",
    "DAYS_EMPLOYED", "montant_en_retard", "nb_paiements", "taux_refus"
]

# === Formulaire utilisateur ===
with st.form("scoring_form"):
    st.subheader("📝 Données principales du client :")

    values = {}
    for feat in IMPORTANT_FEATURES:
        values[feat] = st.number_input(
            feat,
            value=0.0,
            step=0.1,
            format="%.2f"
        )

    submitted = st.form_submit_button("🚀 Lancer le scoring")

# === Requête API ===
if submitted:
    # Payload avec toutes les features par défaut = 0
    payload = {"data": {f: 0 for f in ALL_FEATURES}}

    # Remplacer par les valeurs saisies
    for feat in IMPORTANT_FEATURES:
        payload["data"][feat] = values[feat]

    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=15)
        r.raise_for_status()
        result = r.json()

        # Récupération du score
        score = result.get("prediction", None)
        prob = result.get("probability_bad_payer", None)

        if score is not None and prob is not None:
            st.success("✅ Prédiction réalisée avec succès !")
            st.metric("Score du Client", f"{score}")
            st.metric("Probabilité d'être mauvais payeur", f"{prob:.2%}")

            # Données saisies
            st.subheader("📋 Données saisies")
            df = pd.DataFrame(values.items(), columns=["Variable", "Valeur"])
            st.table(df)

        else:
            st.warning("⚠️ Pas de score reçu depuis l'API.")

    except Exception as e:
        st.error(f"❌ Erreur lors de l'appel à l'API : {e}")
