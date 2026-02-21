"""
Application Streamlit - Détection de Fraude Bancaire
Version Ultra-Optimisée (Robuste et Simple)
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Fraude", page_icon="🏦", layout="wide")

st.title("🏦 Détection de Fraude à la Carte de Crédit")

# Vérifier si le modèle existe
import os
import pickle

model_file = 'fraud_detection_model.pkl'

if not os.path.exists(model_file):
    st.error(f"❌ Le fichier {model_file} n'existe pas!")
    st.info("""
    Solution: Exécutez d'abord le script d'entraînement
    ```
    python train_simple.py
    ```
    Cela créera le fichier fraud_detection_model.pkl
    """)
    st.stop()

# Charger le modèle
try:
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']
    model_name = model_data['model_name']
    feature_names = model_data['feature_names']
    metrics = model_data['metrics']
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
    st.stop()

# Navigation
st.sidebar.title("📋 Menu")
page = st.sidebar.radio("", ["Prédiction", "Batch", "Info", "Guide"])

# ============================================================================
# PAGE 1: PRÉDICTION SIMPLE
# ============================================================================

if page == "Prédiction":
    st.header("Prédiction Simple")
    st.write("Entrez les détails d'une transaction")
    
    col1, col2 = st.columns(2)
    input_data = {}
    
    with col1:
        st.subheader("V1-V14")
        for i in range(1, 15):
            input_data[f'V{i}'] = st.slider(f"V{i}", -10.0, 10.0, 0.0)
    
    with col2:
        st.subheader("V15-V28")
        for i in range(15, 29):
            input_data[f'V{i}'] = st.slider(f"V{i}", -10.0, 10.0, 0.0)
    
    col3, col4 = st.columns(2)
    with col3:
        input_data['Time'] = st.number_input("Time", 0, 172800, 0)
    with col4:
        input_data['Amount'] = st.number_input("Amount (€)", 0.0, 25691.0, 100.0)
    
    if st.button("Analyser", use_container_width=True):
        try:
            X = pd.DataFrame([input_data])[feature_names]
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0]
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if pred == 0:
                    st.metric("Verdict", "✅ OK", "Non-Fraude")
                else:
                    st.metric("Verdict", "🚨 ALERTE", "Fraude")
            
            with col2:
                st.metric("Non-Fraude", f"{prob[0]*100:.1f}%")
            
            with col3:
                st.metric("Fraude", f"{prob[1]*100:.1f}%")
        except Exception as e:
            st.error(f"Erreur: {str(e)}")

# ============================================================================
# PAGE 2: BATCH
# ============================================================================

elif page == "Batch":
    st.header("Analyse Batch")
    st.write("Téléchargez un CSV avec vos transactions")
    
    file = st.file_uploader("CSV", type=['csv'])
    
    if file:
        try:
            df = pd.read_csv(file)
            st.write(f"✓ {len(df)} lignes chargées")
            
            # Vérifier les colonnes
            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                st.error(f"Colonnes manquantes: {missing}")
            else:
                if st.button("Analyser Batch", use_container_width=True):
                    X = df[feature_names]
                    X_scaled = scaler.transform(X)
                    
                    preds = model.predict(X_scaled)
                    probs = model.predict_proba(X_scaled)[:, 1]
                    
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total", len(df))
                    with col2:
                        st.metric("Fraudes", (preds == 1).sum())
                    with col3:
                        st.metric("Taux", f"{(preds==1).sum()/len(df)*100:.1f}%")
                    
                    # Résultats
                    results = df.copy()
                    results['Fraude'] = preds
                    results['Probabilité'] = probs
                    
                    st.dataframe(results[['Time', 'Amount', 'Fraude', 'Probabilité']])
                    
                    # Télécharger
                    csv = results.to_csv(index=False)
                    st.download_button("Télécharger CSV", csv, "resultats.csv")
        except Exception as e:
            st.error(f"Erreur: {str(e)}")

# ============================================================================
# PAGE 3: INFO
# ============================================================================

elif page == "Info":
    st.header("Informations du Modèle")
    
    st.write(f"**Modèle**: {model_name}")
    st.write(f"**Variables**: {len(feature_names)}")
    
    st.subheader("Performances")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
    with col2:
        st.metric("Precision", f"{metrics['precision']*100:.1f}%")
    with col3:
        st.metric("Recall", f"{metrics['recall']*100:.1f}%")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("F1-Score", f"{metrics['f1']*100:.1f}%")
    with col2:
        st.metric("AUC", f"{metrics['auc']*100:.1f}%")
    
    st.subheader("Dataset")
    st.write("""
    - Transactions: 284,807
    - Non-fraude: 284,315 (99.83%)
    - Fraude: 492 (0.17%)
    - Entraînement: 70% | Test: 30%
    """)

# ============================================================================
# PAGE 4: GUIDE
# ============================================================================

elif page == "Guide":
    st.header("Guide d'Utilisation")
    
    st.subheader("📊 Prédiction Simple")
    st.write("""
    1. Entrez les 30 variables avec les sliders
    2. Cliquez sur "Analyser"
    3. Obtenez le résultat instantanément
    """)
    
    st.subheader("📈 Analyse Batch")
    st.write("""
    1. Préparez un CSV avec colonnes: V1-V28, Time, Amount
    2. Téléchargez le fichier
    3. Cliquez "Analyser Batch"
    4. Téléchargez les résultats
    """)
    
    st.subheader("ℹ️ À propos")
    st.write("""
    **Modèle**: Random Forest Classifier
    **Dataset**: Credit Card Fraud Detection
    **Normalisé**: StandardScaler
    **Évaluation**: 5 métriques (Accuracy, Precision, Recall, F1, AUC)
    """)
    
    st.subheader("⚠️ Important")
    st.warning("""
    ⚠️ Projet PÉDAGOGIQUE - À titre informatif
    - Ne remplace pas les systèmes de sécurité professionnels
    - Toujours vérifier avec votre banque
    - Utilisez en complément, pas en substitution
    """)

st.markdown("---")
st.markdown("<small>Projet IA - Détection de Fraude 2026</small>", unsafe_allow_html=True)
