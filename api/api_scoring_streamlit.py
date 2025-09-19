# api_scoring_streamlit.py
import json
import requests
import streamlit as st
import pandas as pd

BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Client Scoring App", page_icon="📊", layout="wide")
st.title("📊 Application de Scoring Client")
st.markdown("Entrez les informations principales du client pour obtenir un score de risque.")

# =============================
# Variables importantes (top 8 tirées du notebook)
# =============================
important_features = [
    "montant_en_retard",
    "nb_previous",
    "taux_refus",
    "montant_moyen_pret",
    "nb_paiements",
    "retard_moyen",
    "montant_paiement_moyen",
    "revenu_annuel"
]

# =============================
# Formulaire utilisateur
# =============================
with st.form("scoring_form"):
    st.subheader("Données principales du client :")
    values = {}
    for feat in important_features:
        values[feat] = st.number_input(feat, value=0.0, step=0.01, format="%.2f")

    submitted = st.form_submit_button("🚀 Lancer le scoring")

# =============================
# Requête API
# =============================
if submitted:
    payload = {"data": values}
    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=15)
        r.raise_for_status()
        result = r.json()

        # Récupération du score
        score = result.get("score", None)

        if score is not None:
            st.success("✅ Prédiction réalisée avec succès !")
            st.metric("Score du client", f"{score:.2f}")

            # Si l’API renvoie aussi l’importance des features
            if "feature_importance" in result:
                fi = pd.DataFrame(result["feature_importance"])
                st.subheader("🔎 Importance des variables dans le modèle")
                st.bar_chart(fi.set_index("feature"))
        else:
            st.warning("⚠️ Pas de score reçu depuis l'API.")

    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")
