import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from pathlib import Path

def main(sample_size=None):
    # Charger les datasets
    reference = pd.read_csv("data_processed/application_train_clean.csv")
    current = pd.read_csv("data/application_test.csv")

    # Si on veut accélérer : on prend un échantillon
    if sample_size:
        reference = reference.sample(sample_size, random_state=42)
        current = current.sample(sample_size, random_state=42)

    # Supprimer la colonne TARGET (pas dispo dans le test)
    if "TARGET" in reference.columns:
        reference = reference.drop(columns=["TARGET"])

    # Garder uniquement les colonnes communes
    common_cols = list(set(reference.columns) & set(current.columns))
    reference = reference[common_cols]
    current = current[common_cols]

    # Créer le rapport Evidently
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    # Sauvegarde en HTML
    output_dir = Path("monitoring/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "drift_report.html"

    report.save_html(str(output_file))
    print(f"✅ Rapport Evidently généré : {output_file}")

if __name__ == "__main__":
    # Mets sample_size=5000 pour tester rapidement
    main(sample_size=5000)



