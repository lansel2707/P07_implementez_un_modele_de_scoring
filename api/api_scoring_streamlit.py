# api_scoring_streamlit.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path

# (Optionnel pour la jauge accessible)
import plotly.graph_objects as go

import joblib

# Pour distinguer mode cloud / local
IS_CLOUD = os.getenv("RUN_MODE", "local") == "cloud"


# =========================
# CONFIG APP
# =========================
st.set_page_config(page_title="Application de Scoring", layout="wide")
BASE_URL = "http://localhost:8000"  # API FastAPI locale
DEFAULT_THRESHOLD = 0.14            # Seuil métier utilisé partout (cohérent avec FastAPI)

# =========================
# Features principales à saisir par l’utilisateur
# (UI simplifiée – on convertit ensuite pour l’API)
# =========================
FEATURE_LABELS = {
    "AMT_INCOME_TOTAL": "Revenu total du client (€)",
    "AMT_CREDIT": "Montant du crédit (€)",
    "AMT_ANNUITY": "Annuité du prêt (€)",
    # Dans le dataset modèle: DAYS_BIRTH est en jours (souvent négatifs).
    # Ici on demande l'âge en ANNÉES et on convertit → jours (positif) pour l'API.
    "AGE_YEARS": "Âge du client (en années)",
    # Idem pour l'ancienneté pro (DAYS_EMPLOYED au modèle).
    "EMP_YEARS": "Ancienneté professionnelle (en années)",
    "montant_en_retard": "Montant total en retard (€)",
    "nb_paiements": "Nombre de paiements effectués",
    "taux_refus": "Taux de refus antérieurs (0–1)",
}

IMPORTANT_FEATURES = list(FEATURE_LABELS.keys())

# =========================
# Toutes les features attendues par le modèle (ordre/nom exacts)
# ⚠️ Conserver cette liste telle quelle (c’est elle qui dicte le payload)
# =========================
ALL_FEATURES = ['CNT_CHILDREN' 'AMT_INCOME_TOTAL' 'AMT_CREDIT' 'AMT_ANNUITY'
 'AMT_GOODS_PRICE' 'REGION_POPULATION_RELATIVE' 'DAYS_BIRTH'
 'DAYS_EMPLOYED' 'DAYS_REGISTRATION' 'DAYS_ID_PUBLISH' 'OWN_CAR_AGE'
 'FLAG_MOBIL' 'FLAG_EMP_PHONE' 'FLAG_WORK_PHONE' 'FLAG_CONT_MOBILE'
 'FLAG_PHONE' 'FLAG_EMAIL' 'CNT_FAM_MEMBERS' 'REGION_RATING_CLIENT'
 'REGION_RATING_CLIENT_W_CITY' 'HOUR_APPR_PROCESS_START'
 'REG_REGION_NOT_LIVE_REGION' 'REG_REGION_NOT_WORK_REGION'
 'LIVE_REGION_NOT_WORK_REGION' 'REG_CITY_NOT_LIVE_CITY'
 'REG_CITY_NOT_WORK_CITY' 'LIVE_CITY_NOT_WORK_CITY' 'EXT_SOURCE_1'
 'EXT_SOURCE_2' 'EXT_SOURCE_3' 'APARTMENTS_AVG' 'BASEMENTAREA_AVG'
 'YEARS_BEGINEXPLUATATION_AVG' 'YEARS_BUILD_AVG' 'COMMONAREA_AVG'
 'ELEVATORS_AVG' 'ENTRANCES_AVG' 'FLOORSMAX_AVG' 'FLOORSMIN_AVG'
 'LANDAREA_AVG' 'LIVINGAPARTMENTS_AVG' 'LIVINGAREA_AVG'
 'NONLIVINGAPARTMENTS_AVG' 'NONLIVINGAREA_AVG' 'APARTMENTS_MODE'
 'BASEMENTAREA_MODE' 'YEARS_BEGINEXPLUATATION_MODE' 'YEARS_BUILD_MODE'
 'COMMONAREA_MODE' 'ELEVATORS_MODE' 'ENTRANCES_MODE' 'FLOORSMAX_MODE'
 'FLOORSMIN_MODE' 'LANDAREA_MODE' 'LIVINGAPARTMENTS_MODE'
 'LIVINGAREA_MODE' 'NONLIVINGAPARTMENTS_MODE' 'NONLIVINGAREA_MODE'
 'APARTMENTS_MEDI' 'BASEMENTAREA_MEDI' 'YEARS_BEGINEXPLUATATION_MEDI'
 'YEARS_BUILD_MEDI' 'COMMONAREA_MEDI' 'ELEVATORS_MEDI' 'ENTRANCES_MEDI'
 'FLOORSMAX_MEDI' 'FLOORSMIN_MEDI' 'LANDAREA_MEDI' 'LIVINGAPARTMENTS_MEDI'
 'LIVINGAREA_MEDI' 'NONLIVINGAPARTMENTS_MEDI' 'NONLIVINGAREA_MEDI'
 'TOTALAREA_MODE' 'OBS_30_CNT_SOCIAL_CIRCLE' 'DEF_30_CNT_SOCIAL_CIRCLE'
 'OBS_60_CNT_SOCIAL_CIRCLE' 'DEF_60_CNT_SOCIAL_CIRCLE'
 'DAYS_LAST_PHONE_CHANGE' 'FLAG_DOCUMENT_2' 'FLAG_DOCUMENT_3'
 'FLAG_DOCUMENT_4' 'FLAG_DOCUMENT_5' 'FLAG_DOCUMENT_6' 'FLAG_DOCUMENT_7'
 'FLAG_DOCUMENT_8' 'FLAG_DOCUMENT_9' 'FLAG_DOCUMENT_10' 'FLAG_DOCUMENT_11'
 'FLAG_DOCUMENT_12' 'FLAG_DOCUMENT_13' 'FLAG_DOCUMENT_14'
 'FLAG_DOCUMENT_15' 'FLAG_DOCUMENT_16' 'FLAG_DOCUMENT_17'
 'FLAG_DOCUMENT_18' 'FLAG_DOCUMENT_19' 'FLAG_DOCUMENT_20'
 'FLAG_DOCUMENT_21' 'AMT_REQ_CREDIT_BUREAU_HOUR'
 'AMT_REQ_CREDIT_BUREAU_DAY' 'AMT_REQ_CREDIT_BUREAU_WEEK'
 'AMT_REQ_CREDIT_BUREAU_MON' 'AMT_REQ_CREDIT_BUREAU_QRT'
 'AMT_REQ_CREDIT_BUREAU_YEAR' 'nb_bureau_credit'
 'montant_total_credit_bureau' 'montant_credit_moyen_bureau'
 'montant_en_retard' 'nb_previous' 'taux_refus' 'montant_moyen_pret'
 'nb_paiements' 'retard_moyen' 'montant_paiement_moyen']

# =========================
# MENU
# =========================
menu = st.sidebar.radio("Navigation", [
    "👨‍💼 Scoring Client",
    "📊 Feature Importance",
    "🆚 Comparaison client"
])


# =========================
# Helpers
# =========================
def years_to_days(years: float) -> int:
    if years is None:
        return 0
    return -int(round(float(years) * 365))  # Home Credit: DAYS_* négatifs

def build_payload_from_inputs(user_inputs: dict) -> dict:
    """
    Construit le payload complet attendu par le modèle local ou l'API :
    - Remplit toutes les features de ALL_FEATURES (0 par défaut)
    - Mappe AGE_YEARS → DAYS_BIRTH et EMP_YEARS → DAYS_EMPLOYED
    - Insère les 6 champs saisis sur la 1re page
    """
    # Conversion années -> jours (positif)
    days_birth = years_to_days(user_inputs.get("AGE_YEARS", 0))
    days_emp = years_to_days(user_inputs.get("EMP_YEARS", 0))

    # Création d’un dictionnaire complet des features avec 0 par défaut
    data_dict = {feat: 0 for feat in ALL_FEATURES}

    # Mise à jour des champs saisis par l'utilisateur
    data_dict.update({
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_emp,
        "AMT_INCOME_TOTAL": float(user_inputs.get("AMT_INCOME_TOTAL", 0) or 0),
        "AMT_CREDIT": float(user_inputs.get("AMT_CREDIT", 0) or 0),
        "AMT_ANNUITY": float(user_inputs.get("AMT_ANNUITY", 0) or 0),
        "montant_en_retard": float(user_inputs.get("montant_en_retard", 0) or 0),
        "nb_paiements": int(user_inputs.get("nb_paiements", 0) or 0),
        "taux_refus": float(user_inputs.get("taux_refus", 0) or 0),
    })

    # Transformation en liste ordonnée pour correspondre à ALL_FEATURES
    data = [data_dict.get(feat, 0) for feat in ALL_FEATURES]

    return {"data": data}


def plot_gauge(prob, threshold=DEFAULT_THRESHOLD):
    """Jauge accessible: vert (zone acceptée) / orange (proche seuil) / rouge (risque)"""
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(prob),
            number={'valueformat':'.3f'},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "#444"},  # aiguille sombre (accessibilité)
                'steps': [
                    {'range': [0, threshold*0.8], 'color': '#71C562'},       # vert
                    {'range': [threshold*0.8, threshold], 'color': '#F4C542'}, # ambre proche seuil
                    {'range': [threshold, 1], 'color': '#E45757'},          # rouge
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.8,
                    'value': threshold,
                }
            },
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Probabilité (seuil métier = {threshold:.2f})"}
        )
    )
    st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

def accessible_fig(figsize=(8,4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(alpha=0.25)
    return fig, ax

# =====================
# PAGE 1 : Scoring
# =====================
if menu == "👨‍💼 Scoring Client":
    st.title("🧮 Application de Scoring Client")
    st.write("Entrez les informations principales du client pour obtenir un score de risque.")

    with st.form("form_scoring"):
        cols = st.columns(2)
        user_inputs = {}

        # Bornes et validations (cohérentes avec ce que tu as demandé)
        with cols[0]:
            user_inputs["AMT_INCOME_TOTAL"] = st.number_input(
                FEATURE_LABELS["AMT_INCOME_TOTAL"], min_value=0.0, max_value=5_000_000.0,
                value=0.0, step=1000.0, format="%.2f"
            )

            user_inputs["AMT_ANNUITY"] = st.number_input(
                FEATURE_LABELS["AMT_ANNUITY"], min_value=0.0, max_value=500_000.0,
                value=0.0, step=100.0, format="%.2f"
            )

            user_inputs["EMP_YEARS"] = st.number_input(
                FEATURE_LABELS["EMP_YEARS"], min_value=0.0, max_value=50.0,
                value=0.0, step=0.5, format="%.1f"
            )

            user_inputs["nb_paiements"] = st.number_input(
                FEATURE_LABELS["nb_paiements"], min_value=0, max_value=300, value=0, step=1
            )

        with cols[1]:
            user_inputs["AMT_CREDIT"] = st.number_input(
                FEATURE_LABELS["AMT_CREDIT"], min_value=0.0, max_value=2_000_000.0,
                value=0.0, step=1000.0, format="%.2f"
            )

            user_inputs["AGE_YEARS"] = st.number_input(
                FEATURE_LABELS["AGE_YEARS"], min_value=18.0, max_value=100.0,
                value=30.0, step=1.0, format="%.0f"
            )

            user_inputs["montant_en_retard"] = st.number_input(
                FEATURE_LABELS["montant_en_retard"], min_value=0.0, max_value=2_000_000.0,
                value=0.0, step=100.0, format="%.2f"
            )

            user_inputs["taux_refus"] = st.number_input(
                FEATURE_LABELS["taux_refus"], min_value=0.0, max_value=1.0,
                value=0.0, step=0.01, format="%.2f"
            )

        submitted = st.form_submit_button("🚀 Lancer le scoring")

# ==============================
# Chargement du modèle (Cloud)
# ==============================
model_path = Path(__file__).resolve().parent.parent / "streamlit_exports" / "gbc_all_final_pipeline.pkl"

model = None
if IS_CLOUD:
    try:
        model = joblib.load(model_path)
        st.success(f"✅ Modèle chargé depuis {model_path.name}")
    except Exception as e:
        st.error(f"❌ Impossible de charger le modèle : {e}")


# ==============================
# Lancement du scoring
# ==============================
submitted = st.button("🚀 Lancer le scoring")

if submitted:
    payload = build_payload_from_inputs(user_inputs)

    try:
        # --- Mode Cloud : prédire en local avec le .pkl ---
        if IS_CLOUD and model is not None:
            X = pd.DataFrame([payload["data"]], columns=ALL_FEATURES)
            if not hasattr(model, "predict_proba"):
                raise RuntimeError("Le modèle ne supporte pas predict_proba().")
            prob_bad = float(model.predict_proba(X)[0][1])     # proba classe 'mauvais payeur'
            result = {"probability_bad_payer": prob_bad}
            st.success("✅ Résultat calculé côté app (mode Cloud)")

        # --- Mode local : appeler l'API FastAPI ---
        else:
            response = requests.post(
                f"{BASE_URL}/predict?threshold={DEFAULT_THRESHOLD}",
                json=payload,
                timeout=15
            )
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Résultat reçu depuis l'API")
            else:
                st.error(f"⚠️ Erreur API {response.status_code} : {response.text}")
                raise RuntimeError(f"API error {response.status_code}")

        # --- Contenu principal du résultat ---
        prob_bad = float(result.get("probability_bad_payer", result.get("probability", 0.0)))

        # Conversion pour affichage : on veut la probabilité d’un bon payeur
        prob_good = 1 - prob_bad

        # Seuil métier affiché pour les bons payeurs (0.14 pour mauvais → 0.86 pour bons)
        seuil = 1 - DEFAULT_THRESHOLD

        # Jauge avec seuil métier
        plot_gauge(prob_good, threshold=seuil)

        # Interprétation lisible du score
        if prob_good >= seuil:
            message = "🟢 Bon payeur probable (faible risque)"
            color = "#4CAF50"  # vert
        elif prob_good >= 0.5:
            message = "🟠 Zone limite : à surveiller"
            color = "#FFC107"  # orange
        else:
            message = "🔴 Mauvais payeur probable (risque élevé)"
            color = "#F44336"  # rouge

        # Affichage du message interprété
        st.markdown(
            f"<h2 style='text-align:center; color:{color};'>{message}</h2>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"❌ Problème lors du scoring : {e}")



# =========================
# PAGE 2 : Feature Importance
# =========================
elif menu == "📊 Feature Importance":
    st.title("📊 Importance des variables")
    st.write("Affichage des importances calculées du modèle (Top 20).")

    import os
    import streamlit as st
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt

    # ✅ Lecture du chemin depuis les secrets Streamlit si défini
    path_importances_str = st.secrets.get("PATH_FEATURE_IMPORTANCE", "")
    if path_importances_str and Path(path_importances_str).exists():
        path_importances = Path(path_importances_str)
    else:
        # fallback local (utile pour exécution locale)
        base_dir = Path(__file__).resolve().parent.parent
        path_importances = base_dir / "notebooks" / "exports" / "feature_importance_top20.csv"

    # Vérification et lecture
    if path_importances.exists():
        df_importances = pd.read_csv(path_importances)
        df_importances.columns = [c.lower() for c in df_importances.columns]

        # Tri décroissant et top 20
        df_top = df_importances.sort_values(by="importance", ascending=False).head(20)

        # Histogramme vertical
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(df_top["feature"], df_top["importance"])
        ax.set_xlabel("Feature")
        ax.set_ylabel("Importance")
        plt.xticks(rotation=80, ha="right")
        st.pyplot(fig)

        # Tableau compact
        st.dataframe(df_top.reset_index(drop=True), use_container_width=True)

    else:
        st.warning(f"⚠️ Fichier des importances non trouvé : {path_importances}")


# ============================================================
# PAGE 3 : Comparaison Client vs Population (alignée équilibrée)
# ============================================================
elif menu == "🆚 Comparaison client":

    from pathlib import Path
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    st.title("🆚 Comparaison Client vs Population")
    st.markdown(
        "Comparez les caractéristiques du client avec la moyenne de la population, "
        "des bons payeurs et des mauvais payeurs. Aucun réentraînement n’est effectué."
    )

    # ------------------------------
    # Charger les statistiques JSON
    # ------------------------------
    stats_path = Path("notebooks/exports/feature_stats.json")
    if not stats_path.exists():
        st.error(f"❌ Fichier {stats_path} introuvable.")
        st.stop()

    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    feature_stats = data["features"]

    # ------------------------------
    # Variables et labels
    # ------------------------------
    VAR_LABELS = {
        "AMT_INCOME_TOTAL": "Revenu total (€)",
        "AMT_CREDIT": "Montant du crédit (€)",
        "AMT_ANNUITY": "Annuité du prêt (€)",
        "DAYS_BIRTH": "Âge du client (années)",
        "DAYS_EMPLOYED": "Ancienneté professionnelle (années)",
    }

    default_vars = list(VAR_LABELS.keys())
    available_names = [f["name"] for f in feature_stats]
    options_names = [v for v in default_vars if v in available_names]

    name_to_label = {k: VAR_LABELS[k] for k in options_names}
    label_to_name = {v: k for k, v in name_to_label.items()}

    options_labels = list(label_to_name.keys())
    default_labels = [name_to_label[v] for v in options_names]

    selected_labels = st.multiselect(
        "Sélectionnez les variables à afficher (1 à 5) :",
        options=options_labels,
        default=default_labels,
        max_selections=5,
        help="Cochez ou décochez pour afficher les jauges correspondantes.",
    )
    selected_vars = [label_to_name[l] for l in selected_labels]
    if not selected_vars:
        st.info("Sélectionnez au moins une variable.")
        st.stop()

    # ------------------------------
    # Orientation métier
    # ------------------------------
    ORIENTATION = {
        "AMT_INCOME_TOTAL": "higher_is_better",
        "AMT_CREDIT": "lower_is_better",
        "AMT_ANNUITY": "lower_is_better",
        "DAYS_BIRTH": "higher_is_better",
        "DAYS_EMPLOYED": "higher_is_better",
    }

    def _fmt_compact(x: float) -> str:
        """Affichage compact des valeurs (k, M)."""
        try:
            x = float(x)
        except Exception:
            return str(x)
        sign = "-" if x < 0 else ""
        a = abs(x)
        if a >= 1_000_000:
            return f"{sign}{a/1_000_000:.3g}M"
        if a >= 1_000:
            return f"{sign}{a/1_000:.3g}k"
        return f"{x:.0f}"

    # ------------------------------
    # Section affichage
    # ------------------------------
    st.subheader("👤 Valeurs Client et comparaison")
    st.caption("Ajustez les valeurs du client à gauche, la jauge correspondante s’aligne entre l’entrée et la moyenne affichée dessous.")

    better_count = 0
    total_count = len(selected_vars)

    for var in selected_vars:
        feat = next(f for f in feature_stats if f["name"] == var)
        pop = feat["population"]
        mean_pop = float(pop["mean"])
        vmin = float(pop.get("min", mean_pop))
        vmax = float(pop.get("max", mean_pop if mean_pop > 0 else 1.0))
        if vmax <= vmin:
            vmax = vmin + 1.0

        # Conversion jours -> années
        if var in ["DAYS_BIRTH", "DAYS_EMPLOYED"]:
            mean_pop = abs(mean_pop) / 365
            vmin = abs(vmin) / 365
            vmax = abs(vmax) / 365

        label_fr = VAR_LABELS[var]
        unit = "€" if "AMT_" in var else "années"

        # --- Colonnes alignées horizontalement ---
        colL, colG, colR = st.columns([1.4, 3.8, 0.9], vertical_alignment="center")

        with colL:
            client_val = st.number_input(
                label_fr,
                value=float(round(mean_pop, 2)),
                step=1.0,
                format="%.2f",
                key=f"in_{var}",
            )
            # moyenne en dessous de l'input
            st.caption(
                f"Moyenne population : {mean_pop:,.2f} {unit}".replace(",", " ").replace(".", ",")
            )

        with colG:
            # --- Jauge fine et centrée ---
            norm = (client_val - vmin) / (vmax - vmin)
            norm = float(np.clip(norm, 0.0, 1.0))
            mean_norm = (mean_pop - vmin) / (vmax - vmin)
            mean_norm = float(np.clip(mean_norm, 0.0, 1.0))

            fig, ax = plt.subplots(figsize=(6, 0.12))
            ax.barh([0], [1.0], color="#FAD2D2", height=0.08)
            ax.barh([0], [norm], color="#7EC8E3", height=0.08)
            ax.axvline(mean_norm, color="red", linestyle="-", linewidth=1.5)

            # alignement centré entre input et caption
            plt.subplots_adjust(top=0.8, bottom=0.2)

            ax.set_xlim(0, 1)
            ax.axis("off")

            st.pyplot(fig, use_container_width=True)

        with colR:
            # --- Delta et interprétation ---
            orientation = ORIENTATION.get(var, "higher_is_better")
            if orientation == "higher_is_better":
                is_better = client_val >= mean_pop
                delta_value = client_val - mean_pop
                delta_color = "normal"
            else:
                is_better = client_val <= mean_pop
                delta_value = client_val - mean_pop
                delta_color = "inverse"

            if is_better:
                better_count += 1

            st.metric(
                label="",
                value=_fmt_compact(client_val),
                delta=_fmt_compact(delta_value),
                delta_color=delta_color,
            )

        st.divider()

    # ------------------------------
    # Bandeau synthèse
    # ------------------------------
    ratio = better_count / total_count if total_count else 0.0
    msg = f"{better_count}/{total_count} critère(s) favorables ({ratio*100:.1f}%)"

    if ratio >= 0.6:
        st.success(f"✅ Situation globalement favorable : {msg}")
    elif ratio >= 0.4:
        st.warning(f"⚠️ Situation mitigée : {msg}")
    else:
        st.error(f"❗ À surveiller : {msg}")
