# 📊 Projet 7 – Implémentez un modèle de scoring

## 🎯 Objectif
Mettre en place un pipeline complet de **scoring de crédit** comprenant :
- Préparation et nettoyage des données  
- Entraînement et suivi des modèles  
- API de scoring (FastAPI)  
- Monitoring de dérive des données (Evidently)  

---

## 🛠️ Démarche et étapes

### 1. Préparation des données
- Nettoyage, feature engineering et sélection de variables dans les notebooks.  
- Données brutes placées dans `data/` (non versionnées sur GitHub car trop volumineuses).  
- Données transformées et prêtes à l’usage dans `data_processed/`.  

### 2. Modélisation
- Entraînement du modèle de scoring avec différents algorithmes.  
- Sélection du meilleur modèle, export en `.pkl` (pickle).  
- Suivi des expériences avec **MLflow**.  
- Sauvegarde des artefacts modèles dans `models/`.  

### 3. Mise à disposition du modèle
- Déploiement d’une **API de scoring avec FastAPI** (fichiers dans `api/`).  
- Documentation interactive via **Swagger** (`input_example_swagger.json`).  
- Interface utilisateur pour la démonstration via **Streamlit**.  

### 4. Monitoring et dérive des données
- Détection du drift avec **Evidently** :  
  - Script `monitoring/run_evidently.py` pour générer un rapport.  
  - Script `monitoring/schedule_drift.py` pour lancer un suivi automatique.  
- Rapports générés automatiquement en HTML dans `monitoring/reports/`.  

---

## 📂 Structure du dépôt


P07_model_scoring_clean/
├── api/ # API FastAPI (scoring + Swagger)
├── data/ # Données brutes (non versionnées, trop volumineuses)
├── data_processed/ # Données transformées et nettoyées
├── monitoring/ # Scripts + rapports Evidently pour le drift
├── notebooks/ # EDA, nettoyage, modélisation, scoring
├── models/ # Artefacts modèles (pickle, etc.)
├── tests/ # Scripts de tests (API, sanity check)
├── utils/ # Scripts utilitaires
├── README.md # Présentation du projet
└── .gitignore # Exclusions (CSV lourds, artefacts MLflow, etc.)


---

## ⚙️ Outils et technologies
- **Python** : 3.12.2  
- **Bibliothèques principales** :  
  - `pandas`, `numpy`, `scikit-learn`  
  - `mlflow` (suivi des expériences)  
  - `fastapi`, `uvicorn` (API de scoring)  
  - `streamlit` (interface utilisateur)  
  - `evidently` (monitoring et détection du drift)  

---

## 🔗 Liens
- 📂 Dépôt GitHub : https://github.com/lansel2707/P07_implementez_un_modele_de_scoring

## 📌 Note
Ce dépôt public contient tout le code et la documentation nécessaires pour reproduire le pipeline complet de scoring de crédit.  
L'API n'est pas hébergée publiquement ; elle est conçue pour être utilisée en local.  
Toutes les instructions de lancement sont disponibles dans ce README.
