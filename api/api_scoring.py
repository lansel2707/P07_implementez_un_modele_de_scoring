# api_scoring.py
# ======================================================
# ✅ API Scoring — Pipeline complet (on ne touche PAS Streamlit)
# ======================================================

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os

app = FastAPI(
    title="API Scoring Crédit – Gradient Boosting (Pipeline Complet)",
    description="API FastAPI pour prédire la probabilité de défaut client à partir du pipeline complet.",
    version="1.0.0",
)

# --- 🔧 Chemin du modèle (garde ton chemin actuel si déjà correct)
MODEL_PATH = "/Users/delatouf/Documents/P07_Implementez_un_modele_de_scoring/notebooks/models/gbc_all_final_pipeline.pkl"  # <-- adapte si besoin

# --- (optionnel) fichier de moyennes (si tu en as exporté un depuis ton notebook)
FEATURE_MEANS_PATH = "/Users/delatouf/Documents/P07_Implementez_un_modele_de_scoring/notebooks/models/feature_means.json"

# ======================================================
# Chargement du modèle et préparation des features
# ======================================================
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Modèle chargé avec succès depuis : {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"❌ Erreur lors du chargement du modèle : {e}")

# Récupère la liste des features attendues par le pipeline
try:
    FEATURE_NAMES = list(model.feature_names_in_)  # sklearn >= 1.0
except Exception:
    # si jamais feature_names_in_ n’est pas dispo (rare), on échoue explicitement
    raise RuntimeError("❌ Impossible d’obtenir model.feature_names_in_. Vérifie que tu charges bien le pipeline entraîné.")

# Charge un vecteur par défaut = moyennes, sinon 0
DEFAULT_VECTOR = {f: float('nan') for f in FEATURE_NAMES}
if os.path.exists(FEATURE_MEANS_PATH):
    try:
        with open(FEATURE_MEANS_PATH, "r") as f:
            means = json.load(f)
        # ne garder que les features présentes dans le modèle
        DEFAULT_VECTOR.update({k: float(means.get(k, 0)) for k in FEATURE_NAMES})
        print("ℹ️ Vecteur par défaut = MOYENNES (fichier trouvé).")
    except Exception as e:
        print(f"⚠️ Impossible de charger les moyennes ({e}). On utilisera 0 pour les features manquantes.")
else:
    print("ℹ️ Vecteur par défaut = ZÉROS (aucun fichier de moyennes trouvé).")

# Affiche un gabarit JSON prêt à coller dans Swagger ("Edit Value")
TEMPLATE_JSON = {"data": {k: DEFAULT_VECTOR[k] for k in FEATURE_NAMES}}
print("\n================= JSON TEMPLATE (copier/coller dans Edit Value) =================")
print(json.dumps(TEMPLATE_JSON, ensure_ascii=False, indent=2))
print("===============================================================================\n")
print(f"🔢 Nombre total de features : {len(FEATURE_NAMES)}")

# ======================================================
# Schéma d'entrée
# ======================================================
class Features(BaseModel):
    data: dict

# ======================================================
# Endpoint utilitaire pour récupérer le template
# ======================================================
@app.get("/template")
def template():
    """
    Renvoie le JSON complet attendu par /predict.
    """
    return TEMPLATE_JSON
# ============================
# Fonction de normalisation des features (à insérer ici)
# ============================
def _normalize_for_model(feat: dict) -> dict:
    """Accepte AGE_YEARS / EMP_YEARS (positifs) ou DAYS_* (quelque soit le signe),
    et renvoie toujours DAYS_BIRTH / DAYS_EMPLOYED négatifs comme dans le training."""
    out = dict(feat)

    # 1) Alias -> DAYS_* si AGE_YEARS / EMP_YEARS fournis (UI en années positives)
    if "AGE_YEARS" in out and "DAYS_BIRTH" in FEATURE_NAMES:
        try:
            out["DAYS_BIRTH"] = -int(round(float(out["AGE_YEARS"]) * 365))
        except Exception:
            pass
        out.pop("AGE_YEARS", None)

    if "EMP_YEARS" in out and "DAYS_EMPLOYED" in FEATURE_NAMES:
        try:
            out["DAYS_EMPLOYED"] = -int(round(float(out["EMP_YEARS"]) * 365))
        except Exception:
            pass
        out.pop("EMP_YEARS", None)

    # 2) Si on a reçu des DAYS_* positifs, force le signe négatif
    if "DAYS_BIRTH" in out and out["DAYS_BIRTH"] is not None:
        try:
            out["DAYS_BIRTH"] = -abs(int(float(out["DAYS_BIRTH"])))
        except Exception:
            pass

    if "DAYS_EMPLOYED" in out and out["DAYS_EMPLOYED"] is not None:
        try:
            out["DAYS_EMPLOYED"] = -abs(int(float(out["DAYS_EMPLOYED"])))
        except Exception:
            pass

    return out
# ======================================================
# Route principale : /predict
# ======================================================
@app.post("/predict")
def predict(features: Features):
    """
    Reçoit {"data": {...}} avec (idéalement) toutes les features.
    Remplit les features manquantes avec les valeurs par défaut (moyennes si dispo, sinon 0),
    réordonne les colonnes, puis renvoie la prédiction + la proba de défaut.
    """
    try:
        incoming = features.data or {}

        # 1) Merge sur le vecteur par défaut (copy pour ne pas muter l’original)
        merged = DEFAULT_VECTOR.copy()
        for k, v in incoming.items():
            # seuls les features connus sont pris en compte
            if k in merged:
                # cast simple -> float si possible
                try:
                    merged[k] = float(v)
                except Exception:
                    # si vraiment pas castable, on laisse la valeur telle quelle (pandas gèrera si numérique)
                    merged[k] = v

        # 2) DataFrame à une ligne + réordonnancement des colonnes
        X_input = pd.DataFrame([merged])[FEATURE_NAMES]

        # 3) Prédiction
        proba_bad = float(model.predict_proba(X_input)[:, 1][0])
        prediction = int(model.predict(X_input)[0])

        # 4) Sortie (on garde ton nom de champ pour Streamlit)
        return {
            "prediction": prediction,
            "probability_bad_payer": round(proba_bad, 6),
        }

    except Exception as e:
        return {"error": f"Erreur pendant la prédiction : {str(e)}"}

# ======================================================
# Petit endpoint de santé
# ======================================================
@app.get("/")
def home():
    return {"message": "✅ API Scoring prête. Utilisez /predict ou /template pour récupérer le JSON complet."}
