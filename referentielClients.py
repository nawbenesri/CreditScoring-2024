
# In[1]:


import general as gen


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

import numpy as np
import csv
import os
from pathlib import Path


# Préparation du dataset

# 1.0 Chargement des données

# In[2]:


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"

REFERENTIEL_CLIENTS = os.getenv(
    "REFERENTIEL_CLIENTS_PATH",
    str(SAMPLE_DATA_DIR / "tableReferentielClients.csv"),
)
PERFORMANCE_DE_PRET = os.getenv(
    "PERFORMANCE_PRET_PATH",
    str(SAMPLE_DATA_DIR / "tablePerformance.csv"),
)
OUTPUT_REFERENTIEL_CLIENTS = os.getenv(
    "OUTPUT_REFERENTIEL_CLIENTS_PATH",
    str(BASE_DIR / "tableReferentielClients_output.csv"),
)
PREDICTIONS_OUTPUT = os.getenv(
    "PREDICTIONS_OUTPUT_PATH",
    str(BASE_DIR / "meow.csv"),
)

def define_remboursement(row):
    # Fonction pour déterminer la valeur de remboursement basée sur les données de performance
    if row["StatutDePaiement"] == "1":
        return "1"
    else:
        return "0"

def add_columns(input_file, output_file, performance_file):
    with open(input_file, mode='r', newline='', encoding='UTF8') as infile, \
         open(output_file, mode='w', newline='', encoding='UTF8') as outfile, \
         open(performance_file, mode='r', newline='', encoding='UTF8') as perf_file:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["Remboursement"]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        performance_reader = csv.DictReader(perf_file)

        # Créer un dictionnaire pour stocker le remboursement pour chaque ClientID
        remboursement_dict = {row["PerformanceID"]: define_remboursement(row) for row in performance_reader}

        for row in reader:
            # Mettre à jour la valeur de remboursement si le ClientID a une correspondance dans performance_pret
            if str(row["ClientID"]) in remboursement_dict:
                row["Remboursement"] = remboursement_dict[str(row["ClientID"])]
            else:
                row["Remboursement"] = "0"  # Par défaut, si pas de correspondance trouvée

            writer.writerow(row)

# Appel de la fonction pour ajouter la colonne de remboursement
add_columns(REFERENTIEL_CLIENTS, OUTPUT_REFERENTIEL_CLIENTS, PERFORMANCE_DE_PRET)


# In[3]:


referentiel_client = gen.create_df(OUTPUT_REFERENTIEL_CLIENTS)
referentiel_client.sort_values(by = "ClientID", ascending=True, inplace=True)
referentiel_client


# Prendre les premières lignes du fichier

# In[4]:


print(referentiel_client.head(3))


# 1.1 Filtrer les colonnes

# In[5]:


columns = ["ClientID", "Nom", "Prenom", "DateDeNaissance", "Secteur", "Residence", "DateRelation", "StatutMatrimonial", "Remboursement"]
gen.filtrer_columns(referentiel_client, columns)
referentiel_client


# 1.2 Nettoyage des données

# In[6]:


gen.clean_data(referentiel_client)
referentiel_client


# 1.3 Correction types data

# In[7]:


def correct_columns_clients(df):
    df["ClientID"]= df["ClientID"].astype("int") #ici je ne peux pas faire int[PK] mais je peux créer des fonctions pour affirmer Client comme PK
    df["Nom"]= df["Nom"].astype("string")
    df["Prenom"]= df["Prenom"].astype("string")
    df["DateDeNaissance"]= df["DateDeNaissance"].astype("int") #implementer date
    df["Secteur"]= df["Secteur"].astype("string")
    df["Residence"]= df["Residence"].astype("string")
    df["DateRelation"]= df["DateRelation"].astype("int") #implementer date
    df["StatutMatrimonial"]= df["StatutMatrimonial"].astype("string")
    df["Remboursement"]= df["Remboursement"].astype("string")


correct_columns_clients(referentiel_client)
referentiel_client
referentiel_client.info()


# In[8]:


referentiel_client.to_csv(OUTPUT_REFERENTIEL_CLIENTS, index=False)


# Train Test Split

# In[9]:


# Sélectionner les colonnes pertinentes
X = referentiel_client[["Nom", "Prenom", "DateDeNaissance"]]
y = referentiel_client["Remboursement"]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Créer le pipeline de prétraitement pour les colonnes catégorielles et numériques
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["DateDeNaissance"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Nom", "Prenom"])
    ]
)

# Appliquer le prétraitement aux données d'entraînement et de test
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

print("X_train:", X_train)
print("X_test:", X_test)
print("y_train:", y_train.values)
print("y_test:", y_test.values)


# Logistic Regression

# In[10]:


outcome_path = PREDICTIONS_OUTPUT
gen.logistic_regression(X_test, y_train, X_train, y_test, outcome_path)


# Random Forest

# In[11]:


gen.random_forest(X_test, y_train, X_train, y_test)


# KNN

# In[12]:


gen.knn(X_test, y_train, X_train, y_test)


# SVM

# In[13]:


gen.svm(X_test, y_train, X_train, y_test)


# Confusion Matrix

# In[14]:


# Example usage:
# Assuming y_test and y_pred are defined
classes = np.array(["Class 0", "Class 1"])
gen.plot_confusion_matrix(X_test, y_train, X_train, y_test, outcome_path, classes=classes, normalize=False)

