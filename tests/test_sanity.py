def test_math_sanity():
    assert 1 + 1 == 2
import pandas as pd
import os

def test_clean_data_exists():
    path = "data_processed/application_train_clean.csv"
    assert os.path.exists(path), "Le fichier nettoyé est introuvable"
    df = pd.read_csv(path)
    assert not df.empty, "Le fichier est vide"
    assert "TARGET" in df.columns, "La colonne TARGET doit exister"
