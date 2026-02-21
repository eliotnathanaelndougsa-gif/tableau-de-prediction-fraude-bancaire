"""
Application Streamlit - Détection de Fraude Bancaire
Version Optimisée pour Streamlit Cloud (sans erreurs DOM)
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

# Configuration
st.set_page_config(
    page_title="Détection de Fraude",
    page_icon="🏦",
    layout="wide"
)

# Thème personnalisé
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Charger le modèle
def load_model():
    """Charger le modèle sans cache pour éviter les erreurs DOM"""
    try:
        with open('fraud_detection_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model_data = load_model()

if model_data is None:
    st.error("❌ Modèle non trouvé")
    st.info("Le fichier 'fraud_detection_model.pkl' est manquant.")
    st.stop()

model = model_data['model']
scaler = model_data['scaler']
model_name = model_data['model_name']
feature_names = model_data['feature_names']
metrics = model_data['metrics']

# Navigation
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio("Sélectionnez une page:", 
    ["📊 Prédiction", "📈 Batch", "📋 Info", "📚 Guide"])

# ============================================================================
# PAGE 1 : PRÉDICTION SIMPLE
# ============================================================================

if page == "📊 Prédiction":
    st.title("🏦 Détection de Fraude - Prédiction Simple")
    st.markdown("Entrez les détails d'une transaction")
    
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    with col1:
        st.subheader("Variables V1-V14")
        for i in range(1, 15):
            feature = f'V{i}'
            input_data[feature] = st.slider(f"{feature}", -10.0, 10.0, 0.0, key=f"slider_{feature}")
    
    with col2:
        st.subheader("Variables V15-V28")
        for i in range(15, 29):
            feature = f'V{i}'
            input_data[feature] = st.slider(f"{feature}", -10.0, 10.0, 0.0, key=f"slider_{feature}")
    
    st.subheader("Autres Variables")
    col3, col4 = st.columns(2)
    with col3:
        input_data['Time'] = st.number_input("Time", min_value=0, max_value=172800, value=0)
    with col4:
        input_data['Amount'] = st.number_input("Amount", min_value=0.0, max_value=25691.0, value=100.0)
    
    if st.button("🔍 Analyser", use_container_width=True):
        X_input = pd.DataFrame([input_data])[feature_names]
        X_scaled = scaler.transform(X_input)
        
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0]
        
        st.markdown("---")
        st.subheader("Résultats")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if pred == 0:
                st.metric("Prédiction", "✅ Non-Fraude")
            else:
                st.metric("Prédiction", "🚨 FRAUDE")
        with col2:
            st.metric("Non-Fraude", f"{prob[0]*100:.2f}%")
        with col3:
            st.metric("Fraude", f"{prob[1]*100:.2f}%")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(['Non-Fraude', 'Fraude'], prob * 100, color=['#2ecc71', '#e74c3c'])
        ax.set_ylabel('Probabilité (%)')
        ax.set_ylim([0, 100])
        st.pyplot(fig, use_container_width=True)
        
        if prob[1] > 0.7:
            st.warning("⚠️ RISQUE ÉLEVÉ de fraude")
        elif prob[1] > 0.4:
            st.info("⚠️ Vérifier cette transaction")
        else:
            st.success("✅ Transaction sûre")

# ============================================================================
# PAGE 2 : PRÉDICTION BATCH
# ============================================================================

elif page == "📈 Batch":
    st.title("🏦 Détection de Fraude - Analyse Batch")
    st.markdown("Téléchargez un CSV avec vos transactions")
    
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"✓ {len(df)} transactions chargées")
            
            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                st.error(f"Colonnes manquantes: {missing}")
            else:
                if st.button("Analyser", use_container_width=True):
                    X_batch = df[feature_names]
                    X_scaled = scaler.transform(X_batch)
                    
                    preds = model.predict(X_scaled)
                    probs = model.predict_proba(X_scaled)[:, 1]
                    
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total", len(df))
                    with col2:
                        st.metric("Fraudes", (preds == 1).sum())
                    with col3:
                        st.metric("Taux", f"{(preds == 1).sum()/len(df)*100:.2f}%")
                    
                    df['Fraude'] = preds
                    df['Prob'] = probs
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(probs * 100, bins=30, color='skyblue')
                    ax.set_xlabel('Probabilité de Fraude (%)')
                    ax.set_ylabel('Nombre')
                    st.pyplot(fig, use_container_width=True)
                    
                    csv = df.to_csv(index=False)
                    st.download_button("📥 Télécharger", csv, "resultats.csv")
        except Exception as e:
            st.error(f"Erreur: {str(e)}")

# ============================================================================
# PAGE 3 : INFORMATIONS
# ============================================================================

elif page == "📋 Info":
    st.title("Informations du Modèle")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Modèle")
        st.info(f"**{model_name}**")
    
    with col2:
        st.subheader("Performances")
    
    metrics_display = {
        'Accuracy': f"{metrics['accuracy']*100:.2f}%",
        'Precision': f"{metrics['precision']*100:.2f}%",
        'Recall': f"{metrics['recall']*100:.2f}%",
        'F1-Score': f"{metrics['f1']*100:.2f}%",
        'AUC-ROC': f"{metrics['auc']*100:.2f}%"
    }
    
    for name, value in metrics_display.items():
        st.write(f"**{name}**: {value}")
    
    st.markdown("---")
    st.subheader("Variables du Modèle")
    st.write(f"**{len(feature_names)} variables** utilisées:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("V1-V14: Variables PCA")
    with col2:
        st.write("V15-V28: Variables PCA")
    
    st.write("**Time**: Secondes depuis la 1ère transaction")
    st.write("**Amount**: Montant en euros")

# ============================================================================
# PAGE 4 : GUIDE
# ============================================================================

elif page == "📚 Guide":
    st.title("Guide d'Utilisation")
    
    st.subheader("🎯 Comment utiliser ?")
    st.write("""
    **Page 1 - Prédiction Simple**
    - Entrez les 30 variables avec les sliders
    - Cliquez sur "Analyser"
    - Obtenez la prédiction instantanée
    
    **Page 2 - Analyse Batch**
    - Téléchargez un CSV
    - Le fichier doit avoir les colonnes V1-V28, Time, Amount
    - Obtenez un rapport complet
    - Téléchargez les résultats
    """)
    
    st.subheader("📊 Méthodologie")
    st.write(f"""
    **Modèle**: {model_name}
    
    **Dataset**: 284,807 transactions
    - Non-fraude: 284,315 (99.83%)
    - Fraude: 492 (0.17%)
    
    **Entraînement**:
    - Normalisation: StandardScaler
    - Train/Test: 70/30
    - Évaluation: 5 métriques
    """)
    
    st.subheader("⚠️ Limitations")
    st.warning("""
    ⚠️ Projet PÉDAGOGIQUE
    - À titre informatif
    - Ne remplace pas les systèmes professionnels
    - Toujours vérifier avec la banque
    """)

# Footer
st.markdown("---")
st.markdown("<small>Détection de Fraude - Cours IA 2026</small>", unsafe_allow_html=True)
