# api_scoring_streamlit.py – version projet P8
import json
import requests
import streamlit as st

# URL de ton API FastAPI (à adapter si déployée ailleurs que local)
BASE_URL = "http://localhost:8000"

# Config de la page
st.set_page_config(page_title="Scoring Client", page_icon="📊", layout="wide")

# Titre principal
st.title("📊 Scoring Client – API MLflow Registry")
st.markdown(
    "Cette application permet de **renseigner les caractéristiques d’un client** "
    "et d’obtenir une **prédiction de scoring** via le modèle MLflow exposé en API."
)

# Chargement des features disponibles via l’API
try:
    resp = requests.get(f"{BASE_URL}/features", timeout=5)
    resp.raise_for_status()
    features = resp.json()  # L’API renvoie la liste des features attendues
except Exception as e:
    st.error(f"❌ Impossible de récupérer la liste des features : {e}")
    st.stop()

# Formulaire Streamlit
st.subheader("📝 Remplir les caractéristiques du client")
with st.form("scoring_form"):
    values = {}
    for feat in features:
        values[feat] = st.number_input(
            feat, value=0.0, step=1.0, format="%.3f"
        )

    submitted = st.form_submit_button("🚀 Lancer le scoring")

# Envoi de la requête à l’API
if submitted:
    payload = {"data": values}
    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        out = r.json()

        st.success("✅ Prédiction réalisée avec succès !")

        # Affichage des résultats
        score = out.get("score", None)
        decision = out.get("decision", None)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Score du client", f"{score:.3f}")
        with col2:
            if decision == 1:
                st.success("🟢 Accordé")
            else:
                st.error("🔴 Refusé")

        # Debug JSON complet (optionnel)
        with st.expander("Voir la réponse brute JSON"):
            st.json(out)

    except Exception as e:
        st.error(f"❌ Erreur lors de l’appel à l’API : {e}")

