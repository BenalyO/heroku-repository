import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import os
import glob
from pathlib import Path
import json

col_date: str = "date_heure"
col_donnees: str = "consommation"
cols: List[str] = [col_date, col_donnees]
fic_export_data: str = "data/interim/data.csv"


def load_data():
    list_fic: list[str] = [Path(e) for e in glob.glob("data/raw/*json")]
    list_df: list[pd.DataFrame] = []
    for p in list_fic:
        # list_df.append(pd.read_json(p))
        with open(p, "r") as f:
            dict_data: dict = json.load(f)
            df: pd.DataFrame = pd.DataFrame.from_dict(dict_data.get("results"))
            list_df.append(df)

    df: pd.DataFrame = pd.concat(list_df, ignore_index=True)
    return df


def format_data(df: pd.DataFrame):
    # typage
    df[col_date] = pd.to_datetime(df[col_date])
    # ordre
    df = df.sort_values(col_date)
    # filtrage colonnes
    df = df[cols]
    # dédoublonnage
    df = df.drop_duplicates()
    return df


def export_data(df: pd.DataFrame):
    os.makedirs("data/interim/", exist_ok=True)
    df.to_csv(fic_export_data, index=False)


def consommation_par_heure(df: pd.DataFrame, heure: int):
    """
    Calcule la consommation moyenne pour une heure donnée de la journée.

    Arguments :
    df : pd.DataFrame
        Le DataFrame contenant les données de consommation.
    heure : int
        L'heure de la journée pour laquelle calculer la consommation (entre 0 et 23).

    Returns :
    float
        La consommation moyenne pour cette heure de la journée.
    """
    # Filtrer les données pour l'heure donnée
    consommations_heure = df[df[col_date].dt.hour == heure][col_donnees]
    # Calculer la moyenne
    moyenne_heure = np.mean(consommations_heure)
    return moyenne_heure


def moyenne_consommation_par_heure(df: pd.DataFrame):
    """
    Génère un graphique montrant la moyenne de la consommation par heure de la journée.

    Arguments :
    df : pd.DataFrame
        Le DataFrame contenant les données de consommation.
    """
    heures = np.arange(24)
    moyennes = [consommation_par_heure(df, heure) for heure in heures]

    # Création du graphique
    plt.figure(figsize=(10, 6))
    plt.plot(heures, moyennes, label='Moyenne de la consommation', linestyle='--', color='red')
    plt.xlabel('Heure de la journée')
    plt.ylabel('Consommation moyenne (kWh)')
    plt.title('Moyenne de la consommation par heure de la journée')
    plt.xticks(heures)
    plt.legend()
    plt.grid(True)
    plt.show()


def main_process():
    df: pd.DataFrame = load_data()
    df = format_data(df)
    export_data(df)
    moyenne_consommation_par_heure(df)


if __name__ == "__main__":
    main_process()
