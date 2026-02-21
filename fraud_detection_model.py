"""
Modèle de détection de fraude à la carte de crédit
Sujet : Classification binaire avec classes déséquilibrées
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CHARGEMENT ET EXPLORATION DES DONNÉES
# ============================================================================

print("=" * 80)
print("ÉTAPE 1 : CHARGEMENT ET EXPLORATION DES DONNÉES")
print("=" * 80)

df = pd.read_csv('creditcard.csv', sep=',', quotechar='"')

print(f"\nShape du dataset : {df.shape}")
print(f"\nPremières lignes :\n{df.head()}")
print(f"\nStatistiques descriptives :\n{df.describe()}")
print(f"\nTypes de données :\n{df.dtypes}")
print(f"\nValeurs manquantes :\n{df.isnull().sum()}")

# ============================================================================
# 2. ANALYSE DU DÉSÉQUILIBRE DES CLASSES
# ============================================================================

print("\n" + "=" * 80)
print("ÉTAPE 2 : ANALYSE DU DÉSÉQUILIBRE DES CLASSES")
print("=" * 80)

# Convertir la colonne 'Class' en entier si nécessaire
df['Class'] = df['Class'].astype(int)

class_distribution = df['Class'].value_counts()
print(f"\nDistribution des classes :\n{class_distribution}")
print(f"\nPourcentage :\n{df['Class'].value_counts(normalize=True) * 100}")

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogramme
class_distribution.plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Distribution des classes (Fraude vs Non-Fraude)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Nombre de transactions')
axes[0].set_xticklabels(['Non-Fraude (0)', 'Fraude (1)'], rotation=0)

# Camembert
axes[1].pie(class_distribution.values, labels=['Non-Fraude (0)', 'Fraude (1)'],
            autopct='%1.2f%%', colors=['green', 'red'], startangle=90)
axes[1].set_title('Proportion des classes', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 3. ANALYSE DESCRIPTIVE
# ============================================================================

print("\n" + "=" * 80)
print("ÉTAPE 3 : ANALYSE DESCRIPTIVE DES VARIABLES")
print("=" * 80)

# Analyse par classe
print("\nComparaison de la variable Amount par classe :")
print(df.groupby('Class')['Amount'].describe())

# Visualisation de la colonne 'Amount'
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogramme de Amount
df[df['Class'] == 0]['Amount'].hist(ax=axes[0], bins=50, label='Non-Fraude', alpha=0.7, color='green')
df[df['Class'] == 1]['Amount'].hist(ax=axes[0], bins=50, label='Fraude', alpha=0.7, color='red')
axes[0].set_title('Distribution de la variable Amount', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Montant')
axes[0].set_ylabel('Fréquence')
axes[0].legend()

# Boîte à moustaches
df.boxplot(column='Amount', by='Class', ax=axes[1])
axes[1].set_title('Boîte à moustaches de Amount par classe', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Classe')
axes[1].set_ylabel('Montant')
plt.suptitle('')  # Supprimer le titre par défaut

plt.tight_layout()
plt.savefig('amount_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 4. PRÉPARATION DES DONNÉES
# ============================================================================

print("\n" + "=" * 80)
print("ÉTAPE 4 : PRÉPARATION DES DONNÉES")
print("=" * 80)

# Séparation des features et de la cible
X = df.drop('Class', axis=1)
y = df['Class']

print(f"\nFeatures shape : {X.shape}")
print(f"Target shape : {y.shape}")

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nX_train shape : {X_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"Distribution y_train :\n{y_train.value_counts()}")
print(f"Distribution y_test :\n{y_test.value_counts()}")

# ============================================================================
# 5. NORMALISATION DES DONNÉES
# ============================================================================

print("\n" + "=" * 80)
print("ÉTAPE 5 : NORMALISATION DES DONNÉES")
print("=" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nMoyenne de X_train_scaled : {X_train_scaled.mean(axis=0)[:5]}...")
print(f"Écart-type de X_train_scaled : {X_train_scaled.std(axis=0)[:5]}...")

# ============================================================================
# 6. ENTRAÎNEMENT DES MODÈLES
# ============================================================================

print("\n" + "=" * 80)
print("ÉTAPE 6 : ENTRAÎNEMENT DES MODÈLES")
print("=" * 80)

# Dictionnaire pour stocker les modèles et leurs performances
models = {}
results = {}

# -------- K-Nearest Neighbors --------
print("\n[1/5] Entraînement du modèle KNN...")
knn_scores = {}

for k in [3, 5, 7, 9, 15]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, knn.predict_proba(X_test_scaled)[:, 1])
    
    knn_scores[k] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}
    print(f"  k={k}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

# Sélectionner le meilleur k
best_k = max(knn_scores, key=lambda x: knn_scores[x]['f1'])
print(f"\nMeilleur k pour KNN : {best_k} (F1={knn_scores[best_k]['f1']:.4f})")

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
models['KNN'] = knn_best
results['KNN'] = knn_scores[best_k]

# -------- Logistic Regression --------
print("\n[2/5] Entraînement du modèle Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])

models['LogisticRegression'] = lr
results['LogisticRegression'] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}
print(f"  Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

# -------- Random Forest --------
print("\n[3/5] Entraînement du modèle Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1])

models['RandomForest'] = rf
results['RandomForest'] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}
print(f"  Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

# -------- Gradient Boosting --------
print("\n[4/5] Entraînement du modèle Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train_scaled, y_train)
y_pred = gb.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, gb.predict_proba(X_test_scaled)[:, 1])

models['GradientBoosting'] = gb
results['GradientBoosting'] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}
print(f"  Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

print("\n[5/5] Entraînement terminé !")

# ============================================================================
# 7. COMPARAISON DES MODÈLES
# ============================================================================

print("\n" + "=" * 80)
print("ÉTAPE 7 : COMPARAISON DES MODÈLES")
print("=" * 80)

results_df = pd.DataFrame(results).T
print(f"\n{results_df}")

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['accuracy', 'precision', 'recall', 'f1']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    results_df[metric].plot(kind='bar', ax=ax, color=['skyblue', 'orange', 'green', 'red'])
    ax.set_title(f'{metric.capitalize()} par modèle', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.capitalize())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 8. ANALYSE DÉTAILLÉE DU MEILLEUR MODÈLE
# ============================================================================

print("\n" + "=" * 80)
print("ÉTAPE 8 : ANALYSE DÉTAILLÉE DU MEILLEUR MODÈLE")
print("=" * 80)

# Trouver le meilleur modèle
best_model_name = results_df['f1'].idxmax()
best_model = models[best_model_name]
print(f"\nMeilleur modèle : {best_model_name} (F1-Score : {results_df.loc[best_model_name, 'f1']:.4f})")

# Prédictions et métriques détaillées
y_pred_best = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

print(f"\nRapport de classification :")
print(classification_report(y_test, y_pred_best))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_best)
print(f"\nMatrice de confusion :\n{cm}")

# Visualisation de la matrice de confusion
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matrice de confusion
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f'Matrice de confusion - {best_model_name}', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Vraie classe')
axes[0].set_xlabel('Classe prédite')

# Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_score:.4f}')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('Taux de faux positifs')
axes[1].set_ylabel('Taux de vrais positifs')
axes[1].set_title('Courbe ROC', fontsize=12, fontweight='bold')
axes[1].legend(loc="lower right")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('confusion_matrix_roc.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 9. SAUVEGARDE DES MODÈLES
# ============================================================================

print("\n" + "=" * 80)
print("ÉTAPE 9 : SAUVEGARDE DES MODÈLES")
print("=" * 80)

# Sauvegarder le meilleur modèle
model_data = {
    'model': best_model,
    'scaler': scaler,
    'model_name': best_model_name,
    'feature_names': X.columns.tolist(),
    'metrics': results[best_model_name]
}

with open('fraud_detection_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✓ Modèle {best_model_name} sauvegardé dans 'fraud_detection_model.pkl'")

# Sauvegarder tous les résultats
with open('models_results.pkl', 'wb') as f:
    pickle.dump({'models': models, 'results': results, 'scaler': scaler}, f)

print(f"✓ Tous les modèles sauvegardés dans 'models_results.pkl'")

print("\n" + "=" * 80)
print("ANALYSE COMPLÈTE TERMINÉE !")
print("=" * 80)
