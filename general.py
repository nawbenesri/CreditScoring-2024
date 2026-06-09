import pandas as pd
import general as gen


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import numpy as np


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline



def create_df(path) :

    with open(path, "r", encoding="UTF8") as file:
        df_referentiel = pd.read_csv(file)
    return df_referentiel


def filtrer_columns(df, columns) :
    columns_to_remove = (column for column in df.columns if column not in columns)
    df.drop(columns=columns_to_remove, inplace=True)


def clean_data(df:pd.DataFrame):
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)


# LOGISTIC REGRESSION

def logistic_regression(X_test, y_train, X_train, y_test, outcome_path):
    classifier =  LogisticRegression()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # print(np.unique(y_train))
    # Assurez-vous que y_train contient deux classes distinctes
    print("Valeurs uniques de y_train :", np.unique(y_train))

    # Assurez-vous que X_train et y_train ont les dimensions attendues
    print("Shape de X_train :", X_train.shape)
    print("Shape de y_train :", y_train.shape)


    print(accuracy_score(y_test, y_pred))

    predictions = classifier.predict_proba(X_test)
    predictions


    df_prediction_prob = pd.DataFrame(predictions, columns=['prob_0', 'prob_1'])
    df_prediction_target = pd.DataFrame(y_pred, columns=['predicted_TARGET'])
    df_test_dataset = pd.DataFrame({"Actual Outcome": y_test.values})


    #df_test_dataset['predicted_TARGET'] = y_pred

    # Combiner les DataFrames
    dfx = pd.concat([df_test_dataset, df_prediction_prob, df_prediction_target], axis=1)

    # Enregistrer dans un fichier CSV
    dfx.to_csv(outcome_path, sep=',', encoding='UTF-8', index=False)

    # Afficher le résultat
    print(dfx.head())

    return y_pred


# RANDOM FOREST

def random_forest(X_test, y_train, X_train, y_test):
    forest = RandomForestClassifier(max_depth=2, random_state=0)
    forest.fit(X_train, y_train)

    pred_forest = forest.predict(X_test)
    print(accuracy_score(y_test, pred_forest))


# KNN

def knn(X_test, y_train, X_train, y_test):
        
    n_neighbors = min(10, len(X_test))

    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(X_train, y_train)

    preKnn = neigh.predict(X_test)

    print(accuracy_score(y_test, preKnn))


# SVM

def svm(X_test, y_train, X_train, y_test):
    svm = SVC(gamma='auto',class_weight="balanced")
    svm.fit(X_train, y_train)
    pred_svm = svm.predict(X_test)
    print(accuracy_score(y_test, pred_svm))

# CONFUSION MATRIX

def plot_confusion_matrix(X_test, y_train, X_train, y_test, outcome_path, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Greys):
    
    y_pred=logistic_regression(X_test, y_train, X_train, y_test, outcome_path)
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Only use the labels that appear in the data
    unique_classes = unique_labels(y_test, y_pred)
    classes = classes[unique_classes.astype(int)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()  # Add this line to actually display the plot
    return ax

