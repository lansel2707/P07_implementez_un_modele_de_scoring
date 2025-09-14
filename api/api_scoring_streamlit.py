# api_scoring_streamlit.py — version simple (celle qui tournait)
import json
import requests
import streamlit as st

BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Scoring Client — API MLflow Registry", page_icon="🤖")

st.title("Scoring Client – API MLflow Registry")
st.caption("Formulaire minimal pour appeler l'API FastAPI qui charge le modèle depuis MLflow.")

# Récupération des features
try:
    resp = requests.get(f"{BASE_URL}/features", timeout=5)
    resp.raise_for_status()
    features = resp.json()  # l'API renvoie directement la liste
except Exception as e:
    st.error(f"Erreur lors de la récupération des features : {e}")
    st.stop()

with st.form("scoring_form"):
    st.subheader("Renseigne les caractéristiques du client :")
    values = {}
    for feat in features:
        values[feat] = st.number_input(feat, value=0.0, step=1.0, format="%.3f")

    submitted = st.form_submit_button("Lancer le scoring !")

if submitted:
    payload = {"data": values}
    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        out = r.json()
        st.success("OK")
        st.json(out)
    except Exception as e:
        st.error(f"Erreur d'appel à /predict : {e}")
