# CreditScoring

Projet Python/FastAPI pour explorer un flux simple de credit scoring: chargement de donnees, nettoyage, preparation de variables et entrainement de modeles de classification.

## Confidentialite des donnees

Les donnees reelles ne sont pas partagees dans ce depot pour des raisons de confidentialite. Elles peuvent contenir des informations client, bancaires ou operationnelles qui ne doivent pas etre publiees.

Le fichier `.gitignore` exclut donc par defaut les fichiers de donnees brutes, les notebooks de travail et les sorties locales (`*.csv`, `*.txt`, `*.ipynb`, `*_output.csv`, etc.). Les seuls CSV publics autorises sont ceux du dossier `sample_data/`, qui sont entierement fictifs.

## Donnees factices

Le dossier `sample_data/` contient des exemples synthetiques pour tester la structure du projet sans exposer de donnees confidentielles:

- `tableReferentielClients.csv`
- `tableComptes.csv`
- `tablePrets.csv`
- `tablePerformance.csv`
- `tableTransactions.csv`
- `tableEvenements.csv`

Ces donnees ne representent aucune personne, aucun client et aucune operation bancaire reels.

## Installation

```bash
pip install -r requirements.txt
```

## Lancer l'API

```bash
uvicorn main:app --reload
```

L'API est ensuite disponible par defaut sur `http://127.0.0.1:8000`.

## Utilisation locale des donnees

Par defaut, le code utilise les fichiers de `sample_data/`. Pour travailler avec des donnees reelles en local, placez-les hors du depot ou dans des fichiers ignores par Git, puis configurez les chemins via les variables d'environnement:

```bash
REFERENTIEL_CLIENTS_PATH=/chemin/local/tableReferentielClients.csv
PERFORMANCE_PRET_PATH=/chemin/local/tablePerformance.csv
OUTPUT_REFERENTIEL_CLIENTS_PATH=/chemin/local/tableReferentielClients_output.csv
PREDICTIONS_OUTPUT_PATH=/chemin/local/predictions.csv
```

Avant de publier une modification, verifier qu'aucun fichier de donnees reelles n'est suivi par Git:

```bash
git status --short
git ls-files "*.csv" "*.txt"
```
