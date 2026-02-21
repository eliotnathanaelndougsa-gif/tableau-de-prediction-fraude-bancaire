"""
Application Streamlit pour la détection de fraude à la carte de crédit
Permet de prédire si une transaction est frauduleuse ou non
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thème
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# CHARGE DU MODÈLE
# ============================================================================

@st.cache_resource
def load_model():
    """Charger le modèle sauvegardé"""
    try:
        with open('fraud_detection_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("❌ Le fichier 'fraud_detection_model.pkl' n'a pas été trouvé.")
        st.info("Veuillez d'abord exécuter 'fraud_detection_model.py' pour entraîner et sauvegarder le modèle.")
        st.stop()

# Charger le modèle
model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
model_name = model_data['model_name']
feature_names = model_data['feature_names']
metrics = model_data['metrics']

# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

# En-tête
st.title("🏦 Détection de Fraude à la Carte de Crédit")
st.markdown("""
Bienvenue dans l'application de détection de fraude bancaire ! 
Cette application utilise un modèle de **machine learning** pour prédire si une transaction 
est frauduleuse ou non. Les données sont basées sur des transactions réelles.
""")

# Barre latérale
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio(
    "Sélectionnez une page :",
    ["📊 Prédiction Simple", "📈 Prédiction Batch", "📋 Informations du Modèle", "📚 Guide d'Utilisation"]
)

# ============================================================================
# PAGE 1 : PRÉDICTION SIMPLE
# ============================================================================

if page == "📊 Prédiction Simple":
    st.header("Prédiction Simple")
    st.markdown("---")
    
    st.write("""
    Entrez les détails d'une transaction pour vérifier si elle est potentiellement frauduleuse.
    """)
    
    col1, col2 = st.columns(2)
    
    # Créer un dictionnaire pour stocker les entrées
    input_data = {}
    
    # Parcourir les features et créer des sliders/inputs
    with col1:
        st.subheader("Variables PCA (V1-V14)")
        for i in range(1, 15):
            feature = f'V{i}'
            input_data[feature] = st.slider(
                f"{feature}",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.1
            )
    
    with col2:
        st.subheader("Variables PCA (V15-V28)")
        for i in range(15, 29):
            feature = f'V{i}'
            input_data[feature] = st.slider(
                f"{feature}",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.1
            )
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Autres Variables")
        input_data['Time'] = st.number_input(
            "Time (secondes depuis première transaction)",
            min_value=0,
            max_value=172800,
            value=0
        )
        input_data['Amount'] = st.number_input(
            "Amount (montant en €)",
            min_value=0.0,
            max_value=25691.0,
            value=100.0,
            step=0.01
        )
    
    # Bouton de prédiction
    if st.button("🔍 Analyser la Transaction", key="predict_button", use_container_width=True):
        # Préparer les données dans le bon ordre
        X_input = pd.DataFrame([input_data])
        X_input = X_input[feature_names]
        
        # Normaliser
        X_input_scaled = scaler.transform(X_input)
        
        # Prédiction
        prediction = model.predict(X_input_scaled)[0]
        probability = model.predict_proba(X_input_scaled)[0]
        
        # Afficher les résultats
        st.markdown("---")
        st.subheader("📊 Résultats de la Prédiction")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            if prediction == 0:
                st.metric("Prédiction", "✅ Non-Frauduleuse", 
                         f"Confiance: {probability[0]*100:.2f}%")
            else:
                st.metric("Prédiction", "🚨 FRAUDULEUSE", 
                         f"Confiance: {probability[1]*100:.2f}%")
        
        with col_res2:
            st.metric("Probabilité de Fraude", f"{probability[1]*100:.2f}%")
        
        with col_res3:
            st.metric("Probabilité de Non-Fraude", f"{probability[0]*100:.2f}%")
        
        # Graphique de probabilité
        fig, ax = plt.subplots(figsize=(10, 5))
        categories = ['Non-Fraude', 'Fraude']
        colors = ['#2ecc71', '#e74c3c']
        bars = ax.bar(categories, probability * 100, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        ax.set_ylabel('Probabilité (%)', fontsize=12, fontweight='bold')
        ax.set_title('Distribution des Probabilités de Prédiction', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bar, prob in zip(bars, probability):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob*100:.2f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        st.pyplot(fig)
        
        # Recommandations
        st.markdown("---")
        st.subheader("💡 Recommandations")
        if probability[1] > 0.7:
            st.warning("⚠️ **ALERTE FRAUDE** : Cette transaction présente un risque élevé de fraude. "
                      "Vérifiez auprès du titulaire de la carte.")
        elif probability[1] > 0.4:
            st.info("⚠️ **RISQUE MODÉRÉ** : Vérifiez cette transaction avec prudence.")
        else:
            st.success("✅ **TRANSACTION SÛRE** : La transaction semble légitime.")

# ============================================================================
# PAGE 2 : PRÉDICTION BATCH
# ============================================================================

elif page == "📈 Prédiction Batch":
    st.header("Prédiction en Batch")
    st.markdown("---")
    
    st.write("""
    Téléchargez un fichier CSV avec plusieurs transactions pour analyser.
    Le fichier doit avoir les mêmes colonnes que le dataset d'entraînement.
    """)
    
    uploaded_file = st.file_uploader(
        "Choisissez un fichier CSV",
        type=['csv'],
        help="Le fichier doit contenir les colonnes V1-V28, Time et Amount"
    )
    
    if uploaded_file is not None:
        try:
            # Lire le fichier
            df_batch = pd.read_csv(uploaded_file, sep=';')
            
            st.write(f"✓ Fichier chargé : {uploaded_file.name}")
            st.write(f"  Nombre de transactions : {len(df_batch)}")
            st.write(f"  Colonnes : {', '.join(df_batch.columns.tolist())}")
            
            # Vérifier les colonnes requises
            missing_cols = [col for col in feature_names if col not in df_batch.columns]
            if missing_cols:
                st.error(f"❌ Colonnes manquantes : {missing_cols}")
            else:
                st.success("✓ Toutes les colonnes requises sont présentes")
                
                # Prédictions
                if st.button("🔍 Analyser les Transactions", use_container_width=True):
                    with st.spinner("Analyse en cours..."):
                        # Préparer les données
                        X_batch = df_batch[feature_names]
                        X_batch_scaled = scaler.transform(X_batch)
                        
                        # Prédictions
                        predictions = model.predict(X_batch_scaled)
                        probabilities = model.predict_proba(X_batch_scaled)[:, 1]
                        
                        # Ajouter les résultats
                        df_batch['Prediction'] = predictions
                        df_batch['Fraud_Probability'] = probabilities
                        df_batch['Prediction_Label'] = df_batch['Prediction'].apply(
                            lambda x: '🚨 FRAUDE' if x == 1 else '✅ NON-FRAUDE'
                        )
                        
                        # Statistiques
                        st.markdown("---")
                        st.subheader("📊 Résultats Généraux")
                        
                        fraud_count = (predictions == 1).sum()
                        non_fraud_count = (predictions == 0).sum()
                        fraud_rate = fraud_count / len(predictions) * 100
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total de Transactions", len(predictions))
                        with col2:
                            st.metric("Frauduleuses", fraud_count, f"{fraud_rate:.2f}%")
                        with col3:
                            st.metric("Non-Frauduleuses", non_fraud_count, f"{100-fraud_rate:.2f}%")
                        with col4:
                            st.metric("Probabilité Moyenne", f"{probabilities.mean()*100:.2f}%")
                        
                        # Graphique
                        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                        
                        # Histogramme des prédictions
                        pred_counts = pd.Series(predictions).value_counts()
                        axes[0].bar(['Non-Fraude', 'Fraude'], 
                                   [pred_counts.get(0, 0), pred_counts.get(1, 0)],
                                   color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
                        axes[0].set_ylabel('Nombre de Transactions')
                        axes[0].set_title('Distribution des Prédictions')
                        axes[0].grid(axis='y', alpha=0.3)
                        
                        # Graphique des probabilités
                        axes[1].hist(probabilities * 100, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
                        axes[1].axvline(50, color='red', linestyle='--', linewidth=2, label='Seuil (50%)')
                        axes[1].set_xlabel('Probabilité de Fraude (%)')
                        axes[1].set_ylabel('Nombre de Transactions')
                        axes[1].set_title('Distribution des Probabilités')
                        axes[1].legend()
                        axes[1].grid(alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Tableau des résultats
                        st.markdown("---")
                        st.subheader("📋 Détails des Transactions")
                        
                        # Afficher les transactions frauduleuses
                        fraud_transactions = df_batch[df_batch['Prediction'] == 1]
                        if len(fraud_transactions) > 0:
                            st.warning(f"**⚠️ {len(fraud_transactions)} Transaction(s) Frauduleuse(s) Détectée(s)**")
                            display_cols = ['Time', 'Amount', 'Fraud_Probability', 'Prediction_Label']
                            st.dataframe(fraud_transactions[display_cols].style.applymap(
                                lambda x: 'background-color: #ffcccc' if isinstance(x, str) and '🚨' in x else ''
                            ))
                        else:
                            st.success("✓ Aucune fraude détectée")
                        
                        # Télécharger les résultats
                        st.markdown("---")
                        csv = df_batch.to_csv(sep=';', index=False)
                        st.download_button(
                            label="📥 Télécharger les Résultats (CSV)",
                            data=csv,
                            file_name="fraud_predictions.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement du fichier : {str(e)}")

# ============================================================================
# PAGE 3 : INFORMATIONS DU MODÈLE
# ============================================================================

elif page == "📋 Informations du Modèle":
    st.header("Informations du Modèle")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Modèle Utilisé")
        st.info(f"**{model_name}**")
    
    with col2:
        st.subheader("📊 Performances Globales")
        st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        st.metric("Precision", f"{metrics['precision']*100:.2f}%")
        st.metric("Recall", f"{metrics['recall']*100:.2f}%")
        st.metric("F1-Score", f"{metrics['f1']*100:.2f}%")
        st.metric("AUC-ROC", f"{metrics['auc']*100:.2f}%")
    
    st.markdown("---")
    st.subheader("📝 Description des Métriques")
    
    metrics_desc = {
        "Accuracy": "Proportion de prédictions correctes (TP + TN) / Total",
        "Precision": "Parmi les fraudes détectées, combien sont vraiment des fraudes ? TP / (TP + FP)",
        "Recall": "Parmi les vraies fraudes, combien ont été détectées ? TP / (TP + FN)",
        "F1-Score": "Moyenne harmonique entre Precision et Recall",
        "AUC-ROC": "Aire sous la courbe ROC, mesure de la capacité discriminante du modèle"
    }
    
    for metric, desc in metrics_desc.items():
        st.write(f"- **{metric}** : {desc}")
    
    st.markdown("---")
    st.subheader("📚 Variables du Modèle")
    
    st.write(f"Le modèle utilise **{len(feature_names)}** variables :")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Variables PCA (V1-V14)** :")
        st.write(", ".join([f"V{i}" for i in range(1, 15)]))
    
    with col2:
        st.write("**Variables PCA (V15-V28)** :")
        st.write(", ".join([f"V{i}" for i in range(15, 29)]))
    
    st.write("**Autres variables :**")
    st.write("- Time : Temps écoulé en secondes depuis la première transaction")
    st.write("- Amount : Montant de la transaction en euros")

# ============================================================================
# PAGE 4 : GUIDE D'UTILISATION
# ============================================================================

elif page == "📚 Guide d'Utilisation":
    st.header("Guide d'Utilisation")
    st.markdown("---")
    
    st.subheader("🎯 Comment Utiliser l'Application ?")
    
    st.write("""
    ### 1. Prédiction Simple
    - Entrez les détails d'une seule transaction
    - Ajustez les valeurs des 30 variables avec les sliders
    - Cliquez sur "Analyser la Transaction" pour obtenir la prédiction
    - Recevez une réponse instantanée avec le niveau de confiance
    
    ### 2. Prédiction Batch
    - Préparez un fichier CSV avec vos transactions
    - Le fichier doit contenir les colonnes : V1-V28, Time, Amount
    - Téléchargez le fichier
    - Obtenez un rapport détaillé avec tous les résultats
    - Téléchargez les prédictions en CSV
    
    ### 3. Informations du Modèle
    - Consultez les performances du modèle
    - Comprenez les métriques utilisées
    - Explorez les variables du modèle
    """)
    
    st.markdown("---")
    st.subheader("📊 Qu'est-ce que le Modèle ?")
    
    st.write(f"""
    Ce modèle utilise l'algorithme **{model_name}** entraîné sur un dataset 
    de **284,807 transactions bancaires réelles** contenant :
    - 284,315 transactions non-frauduleuses
    - 492 transactions frauduleuses
    
    Le modèle a été entraîné à reconnaître les patterns des transactions frauduleuses
    et peut prédire si une nouvelle transaction est potentiellement frauduleuse.
    """)
    
    st.markdown("---")
    st.subheader("⚠️ Limitations et Recommandations")
    
    st.warning("""
    - Ce modèle n'est à titre **informatif et pédagogique**
    - Il doit être utilisé **en complément** et non en remplacement des systèmes de détection professionnels
    - Les résultats dépendent de la qualité des données d'entrée
    - En cas de doute, contactez toujours votre banque
    - Le modèle peut avoir des faux positifs et des faux négatifs
    """)
    
    st.markdown("---")
    st.subheader("🔧 Spécifications Techniques")
    
    st.write(f"""
    - **Modèle** : {model_name}
    - **Nombre de variables** : {len(feature_names)}
    - **Normalisation** : StandardScaler
    - **Ensemble de test** : 30% des données (85,442 transactions)
    - **Framework** : scikit-learn, Streamlit
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <small>Modèle de Détection de Fraude à la Carte de Crédit | Projet d'Introduction à l'IA</small>
    <br>
    <small>© 2026 - Tous droits réservés</small>
</div>
""", unsafe_allow_html=True)
