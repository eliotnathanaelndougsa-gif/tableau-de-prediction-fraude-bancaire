# 🛡️ Détection de Fraude Bancaire - Application Streamlit

Application de Machine Learning pour la détection de fraudes aux cartes de crédit.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Description

Cette application Streamlit utilise des techniques de Machine Learning pour détecter les transactions frauduleuses par carte de crédit. Elle implémente un modèle Random Forest avec équilibrage des classes via SMOTE pour gérer le déséquilibre des données.

## 🚀 Fonctionnalités

- **Chargement automatique des données** - Le fichier CSV est chargé automatiquement
- **Upload de fichier personnalisé** - Possibilité de télécharger votre propre fichier CSV
- **Analyse interactive** - Visualisation des métriques de performance
- **Matrice de confusion** - Visualisation interactive des résultats
- **Courbes ROC et Precision-Recall** - Évaluation graphique du modèle
- **Prédiction en temps réel** - Testez de nouvelles transactions

## 📊 Format des données

Le fichier CSV doit contenir les colonnes suivantes:

| Colonne | Description |
|---------|-------------|
| Time | Temps entre les transactions (secondes) |
| V1 à V28 | Features anonymisées (résultats PCA) |
| Amount | Montant de la transaction |
| Class | Variable cible (0 = Légitime, 1 = Fraude) |

## 🛠️ Installation

1. Clonez le projet ou téléchargez les fichiers

2. Installez les dépendances:
```
bash
pip install -r requirements.txt
```

3. Lancez l'application:
```
bash
streamlit run app.py
```

L'application sera accessible à l'adresse: `http://localhost:8501`

## 📁 Fichiers

- `app.py` - Application principale Streamlit
- `requirements.txt` - Liste des dépendances Python
- `creditcard_reduced.csv` - Jeu de données d'exemple
- `README.md` - Documentation du projet

## 🔧 Dépendances

- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- plotly >= 5.15.0
- scikit-learn >= 1.3.0
- imbalanced-learn >= 0.11.0

## 💡 Utilisation

1. **Lancez l'application** avec `streamlit run app.py`

2. **Page d'accueil** - Consultez les instructions et fonctionnalités

3. **Onglet Analyse** - Visualisez:
   - Métriques de performance (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
   - Distribution des classes
   - Matrice de confusion
   - Courbes ROC et Precision-Recall

4. **Onglet Prédiction** - Entrez les caractéristiques d'une transaction:
   - Montant (Amount)
   - Features V1 à V28
   - Obtenez une prédiction instantanée

## 🔬 Modèle

- **Algorithme**: Random Forest Classifier
- **Équilibrage**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Métriques**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## 📈 Exemple de résultats

Après traitement des données, vous verrez:
- Un modèle entraîné avec ~99% de performance
- Visualisations interactives avec Plotly
- Possibilité de tester des prédictions

## 🤖 Déploiement

Pour déployer sur le cloud:
- **Streamlit Cloud**: Poussez sur GitHub et connectez à Streamlit Cloud
- **Render**: Service de déploiement gratuit
- **Heroku**: Plateforme PaaS

## 📝 Auteurs

- Développé avec Streamlit et scikit-learn

## 📄 Licence

MIT License
