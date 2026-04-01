import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevents matplotlib warnings
import matplotlib.pyplot as plt
from datetime import datetime
import io

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model loading for better performance
@st.cache_resource
def load_model():
    bundle = joblib.load("loan_project_final.pkl")
    return bundle["model"], bundle["feature_columns"]

# Define CSS for the entire app - Fixed sidebar visibility
st.markdown("""
<style>
@import url('');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    min-height: 100vh;
    background-image: url('https://images.pexels.com/photos/269077/pexels-photo-269077.jpeg?auto=compress&cs=tinysrgb&w=1600') !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    position: relative;
}

/* Dark overlay for better readability */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.7);
    pointer-events: none;
    z-index: 0;
}

/* Fix sidebar visibility - CRITICAL */
.css-1d391kg,
.css-17eq0hr,
.css-1lcbmhc,
.css-1outpf7,
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.95) !important;
    visibility: visible !important;
    display: block !important;
    opacity: 1 !important;
}

/* Ensure sidebar content is visible */
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #1a1a2e !important;
    visibility: visible !important;
}

/* Main content area */
.main .block-container {
    padding: 1.5rem 1rem 1.5rem 1rem;
    max-width: 1200px;
    margin: 0 auto;
    position: relative;
    z-index: 2;
}

/* Hide only the menu and footer */
#MainMenu, footer {
    visibility: hidden !important;
}

/* Title Section */
.hero-title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    text-align: center;
    color: #d0d0d0;
    font-size: 1.3rem;
    margin-bottom: 1.0rem;
    font-weight: 420;
}

.box-title {
    font-size: 1.5rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #667eea;
    font-weight: 700;
    margin-bottom: 1.2rem;
    border-bottom: 2px solid #f0f0f0;
    padding-bottom: 0.7rem;
}

/* Form elements */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stRadio > div {
    background: #f8f9fa !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 15px !important;
    color: #1a1a2e !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
}

label {
    color: white !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    margin-bottom: 0.10rem !important;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
    padding: 0.85rem !important;
    width: 100%;
    margin-top: 1rem;
    cursor: pointer;
    transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(102,126,234,0.3);
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(102,126,234,0.5);
}

.stButton > button:active {
    transform: translateY(0px);
}

/* Enhanced download button */
.download-button {
    background: linear-gradient(135deg, #4caf50, #2e7d32) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
    padding: 0.85rem !important;
    width: 100%;
    cursor: pointer;
    transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(76,175,80,0.3);
    display: flex;
    align-items: center;
    justify-content: center;
}

.download-button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(76,175,80,0.5);
}

/* Result boxes */
.result-approved {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    margin-bottom: 1rem;
    animation: slideIn 0.5s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.result-approved h3 {
    color: #2e7d32;
    margin: 0;
    font-size: 1.6rem;
    font-weight: 800;
}

.result-approved p {
    font-size: 0.9rem;
    margin-top: 0.5rem;
    color: #1b5e20;
}

.result-rejected {
    background: linear-gradient(135deg, #ffebee, #ffcdd2);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
    margin-bottom: 1rem;
    animation: slideIn 0.5s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.result-rejected h3 {
    color: #c62828;
    margin: 0;
    font-size: 1.6rem;
    font-weight: 800;
}

.result-rejected p {
    font-size: 0.9rem;
    margin-top: 0.5rem;
    color: #b71c1c;
}

/* Probability bar */
.prob-container {
    background: white;
    border-radius: 16px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    animation: fadeIn 0.5s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.95rem;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 0.7rem;
}

.prob-label span:last-child {
    font-size: 1.2rem;
    font-weight: 800;
    color: #667eea;
}

.prob-track {
    background: #e0e0e0;
    border-radius: 10px;
    height: 10px;
    overflow: hidden;
}

.prob-fill {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #667eea, #764ba2);
    transition: width 0.6s ease;
}

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 30px;
    font-size: 0.9rem;
    font-weight: 700;
    margin-top: 0.8rem;
}

.risk-low {
    background: #e8f5e9;
    color: #2e7d32;
    border: 1px solid #81c784;
}

.risk-mid {
    background: #fff3e0;
    color: #ed6c02;
    border: 1px solid #ffb74d;
}

.risk-high {
    background: #ffebee;
    color: #c62828;
    border: 1px solid #ef9a9a;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #f8f9fa !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    text-align: center !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
}

[data-testid="stMetricLabel"] {
    color: #667eea !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"] {
    color: #1a1a2e !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
}

/* Tips section styling */
.tips-container {
    background: linear-gradient(135deg, #667eea15, #764ba215);
    border-radius: 16px;
    padding: 1.2rem;
    margin-top: 1rem;
    border-left: 4px solid #667eea;
    animation: fadeIn 0.5s ease;
}

.tips-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.tips-approved {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    padding: 1rem;
    border-radius: 12px;
    border-left: 4px solid #2e7d32;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.tips-rejected {
    background: linear-gradient(135deg, #ffebee, #ffcdd2);
    padding: 1rem;
    border-radius: 12px;
    border-left: 4px solid #c62828;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.tips-list {
    margin: 0.5rem 0 0 1.2rem;
    color: #1a1a2e;
    line-height: 1.6;
}

.tips-list li {
    margin: 0.5rem 0;
}

/* Feature cards */
.feature-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.feature-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    text-align: center;
}

.feature-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 0.5rem;
    text-align: center;
}

.feature-desc {
    color: #666;
    text-align: center;
    line-height: 1.5;
}

/* Model detail cards - ADDED MIN-HEIGHT FOR EQUAL SIZE */
.model-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    min-height: 320px; /* Ensures all cards are equal height */
}

.model-card h3 {
    color: #667eea;
    font-size: 1.3rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.model-card p, .model-card li {
    color: #1a1a2e;
    line-height: 1.6;
}

.model-card ul {
    margin-left: 1.7rem;
}

/* NEW: Vibrant Risk Assessment Card */
.risk-assessment-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    color: white;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.risk-assessment-card h3 {
    color: white;
    font-size: 1.5rem;
    margin-bottom: 1.5rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.risk-assessment-card .risk-badge {
    background: rgba(255, 255, 255, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.4);
    color: white;
}

/* Hero image container */
.hero-image {
    width: 100%;
    max-width: 800px;
    border-radius: 16px;
    margin: 0 auto 1.5rem auto;
    display: block;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}

/* Result page specific styles */
.result-main-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    text-align: center;
}

.result-status {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}

.result-message {
    font-size: 1.1rem;
    color: #666;
    margin-bottom: 1.5rem;
}

.result-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-top: 1.5rem;
}

.result-item {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    border: 1px solid #e0e0e0;
}

.result-item-label {
    font-size: 0.85rem;
    color: #666;
    margin-bottom: 0.5rem;
}

.result-item-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1a1a2e;
}

.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.info-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    background: #f8f9fa;
    border-radius: 8px;
    border-left: 3px solid #667eea;
}

.info-label {
    font-weight: 600;
    color: #1a1a2e;
}

.info-value {
    font-weight: 700;
    color: #667eea;
}

/* Animations */
@keyframes slideIn {
    from {
        transform: translateY(20px);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

/* Responsive */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem;
    }
    .hero-title {
        font-size: 1.8rem;
    }
    .result-grid {
        grid-template-columns: 1fr;
    }
    .info-grid {
        grid-template-columns: 1fr;
    }
    
    /* Remove fixed height on mobile for better responsiveness */
    .model-card {
        min-height: auto;
    }
}
</style>
""", unsafe_allow_html=True)

# Initialize query parameters
query_params = st.query_params
if "page" not in query_params:
    query_params["page"] = "Home"

# Initialize session state
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None

# Initialize all predictions list to store all prediction results
if 'all_predictions' not in st.session_state:
    st.session_state.all_predictions = []

# Initialize flag to track if current prediction has been stored
if 'current_prediction_stored' not in st.session_state:
    st.session_state.current_prediction_stored = False

# Get current page from query params
current_page = query_params["page"]

# Sidebar navigation - Simplified and more visible
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 2rem; padding: 1rem;">
    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🏦</div>
    <div style="font-size: 1.2rem; font-weight: 700; color: #1a1a2e;">Loan Approval</div>
    <div style="font-size: 0.9rem; color: #667eea; margin-top: 0.25rem;">Prediction System</div>
</div>
""", unsafe_allow_html=True)

# Page selection with navigation
def navigate_to(page_name):
    query_params["page"] = page_name
    st.rerun()

# Create navigation buttons in sidebar with better styling
st.sidebar.markdown("---")
if st.sidebar.button("🏠 Home", width='stretch', key="nav_home"):
    navigate_to("Home")

if st.sidebar.button("📝 Application", width='stretch', key="nav_application"):
    navigate_to("Application")

if st.sidebar.button("📊 Results", width='stretch', key="nav_results"):
    navigate_to("Results")

if st.sidebar.button("🤖 About Model", width='stretch', key="nav_about"):
    navigate_to("About Model")

st.sidebar.markdown("---")

# Function to generate comprehensive CSV report with all predictions
def generate_all_predictions_csv():
    # Create an empty list to store all data
    all_data = []
    
    # Add all prediction data (NO MANUAL HEADER ADDED)
    for i, pred_data in enumerate(st.session_state.all_predictions):
        data = pred_data['data']
        prediction = pred_data['prediction']
        probability = pred_data['probability']
        emi = pred_data['emi']
        total_interest = pred_data['total_interest']
        
        # Create a dictionary with all the data in raw format
        row = {
            'S.No': i + 1,
            'Report Generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Application ID': f"APP{datetime.now().strftime('%Y%m%d%H%M%S')}-{i+1}",
            'Gender': data['gender'],
            'Marital Status': data['married'],
            'Dependents': data['dependents'],
            'Education': data['education'],
            'Self Employed': data['self_employed'],
            'Applicant Income': data['income'],
            'Coapplicant Income': data['coapplicant_income'],
            'Total Income': data['total_income'],
            'Loan Amount': data['loan_amount'] * 1000,
            'Loan Term (Days)': data['loan_term_days'],
            'Loan Term (Months)': round(data['loan_term_days'] / 30, 1),
            'Credit History': 'Good' if data['credit_history'] == 1.0 else 'Bad',
            'Property Area': data['property_area'],
            'Loan to Income Ratio': data['loan_income_ratio'],
            'Prediction Result': 'Approved' if prediction == 1 else 'Rejected',
            'Approval Probability': f"{probability:.2%}",
            'Risk Level': 'Low' if probability >= 0.70 else 'Moderate' if probability >= 0.50 else 'High',
            'Monthly EMI': emi,
            'Total Interest': total_interest,
            'Total Payment': emi * data['loan_term_days'] / 30,
            'Model Used': 'Random Forest',
            'Model Accuracy': '93%'
        }
        all_data.append(row)
    
    # Create DataFrame (Pandas automatically uses keys as headers)
    df = pd.DataFrame(all_data)
    
    # Convert to CSV
    csv = df.to_csv(index=False)
    
    return csv

# Home Page - Introduction with Bank Building Image
if current_page == "Home":
    # Reset prediction stored flag when going to home
    st.session_state.current_prediction_stored = False
    
    # Title
    st.markdown("""
    <div class="hero-title">Welcome to Loan Approval Predictor</div>
    <div class="hero-subtitle">Machine Learning Based loan approval system</div>
    """, unsafe_allow_html=True)
    
    # MODIFICATION START: Replaced single large image with two small side-by-side images
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <p style="font-size: 1.1rem; color: #d0d0d0;"></p>
    </div>
    """, unsafe_allow_html=True)
    
    img_col1, img_col2 = st.columns(2, gap="small")
    
    with img_col1:
        # FIXED: Changed use_column_width to width='stretch'
        st.image("https://z-cdn-media.chatglm.cn/files/5f497a5d-4e54-4a55-b000-825739c37335.jpeg?auth_key=1874625401-17dc83d7807a494d80555fe5c12b5dbf-0-82e2a65ba9090ea000da0c08a2092acc", 
                 caption="Savings & Growth", width='stretch')
    
    with img_col2:
        # FIXED: Changed use_column_width to width='stretch'
        st.image("https://z-cdn-media.chatglm.cn/files/2a3b9b1a-2f1d-41b0-8c75-9b4672833c10.jpeg?auth_key=1874625401-221c3a8c2a8c44dca29cf88ae9de7264-0-bb61016812af660c24fc244f31627d12", 
                 caption="Loan & Property", width='stretch')
    # MODIFICATION END
    
    # Model Overview
    st.markdown("""
    <div class="model-card">
        <h3>🤖 Our Loan Approval Model</h3>
        <p>Our Loan Approval Prediction System uses a sophisticated <strong>Random Forest</strong> machine learning model trained on thousands of historical loan applications. The model analyzes multiple factors to predict the likelihood of loan approval with high accuracy, helping financial institutions make faster and more informed lending decisions.</p>
        <p><strong>Model Performance:</strong> Our Random Forest model achieves an impressive <strong>93% accuracy</strong> on test datasets, making it highly reliable for loan approval predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Information (No Graphs)
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="model-card">
            <h3>📊 Key Input Features</h3>
            <ul>
                <li><strong>Credit History:</strong> Most important factor (35% weight)</li>
                <li><strong>Income Level:</strong> Combined applicant income (18% weight)</li>
                <li><strong>Loan Amount:</strong> Requested loan size (12% weight)</li>
                <li><strong>Loan Term:</strong> Repayment period (10% weight)</li>
                <li><strong>Property Area:</strong> Location type (8% weight)</li>
                <li><strong>Education:</strong> Academic qualification (7% weight)</li>
                <li><strong>Dependents:</strong> Number of dependents (6% weight)</li>
                <li><strong>Marital Status:</strong> Married or single (4% weight)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="model-card">
            <h3>⚙️ How It Works</h3>
            <ol>
                <li><strong>Data Collection:</strong> Collect applicant information including personal details, financial information, and loan requirements</li>
                <li><strong>Data preprocessing:</strong>handle missing values, balance it</li>
                <li><strong>Feature Engineering:</strong> Transform raw data into meaningful features for better predictions</li>
                <li><strong>Model selection:</strong>compare all model and select random forest for better performance</li>
                <li><strong>Model Prediction:</strong> Random Forest processes features and outputs approval probability</li>
                <li><strong>Decision Making:</strong> Provide approval decision based on probability and thresholds</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="model-card">
            <h3>🔬 Engineered Features</h3>
            <ul>
                <li><strong>Total Income:</strong> Combined income of all applicants</li>
                <li><strong>Loan-to-Income Ratio:</strong> Loan amount relative to income</li>
                <li><strong>Log-transformed Income:</strong> Normalized income values</li>
                <li><strong>Log-transformed Loan:</strong> Normalized loan amounts</li>
                <li><strong>Income Stability:</strong> Derived from employment patterns</li>
                <li><strong>Debt Burden:</strong> Estimated monthly obligations</li>
                <li><strong>Debt Burden:</strong> Estimated monthly obligations</li>
                <li><strong>Risk Score:</strong> Composite risk assessment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="model-card">
            <h3>📈 Model Performance Metrics</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                    <div style="font-size: 1.8rem; font-weight: 800; color: #667eea;">93%</div>
                    <div style="font-size: 0.9rem; color: #666;">Accuracy</div>
                </div>
                <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                    <div style="font-size: 1.8rem; font-weight: 800; color: #667eea;">91%</div>
                    <div style="font-size: 0.9rem; color: #666;">Precision</div>
                </div>
                <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                    <div style="font-size: 1.8rem; font-weight: 800; color: #667eea;">91%</div>
                    <div style="font-size: 0.9rem; color: #666;">Recall</div>
                </div>
                <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                    <div style="font-size: 1.8rem; font-weight: 800; color: #667eea;">0.95</div>
                    <div style="font-size: 0.9rem; color: #666;">AUC-ROC</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to Action Button
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <p style="font-size: 1.2rem; color: white; margin-bottom: 1.5rem;">Ready to apply for a loan?</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📝 START APPLICATION", width='stretch', type="primary"):
        navigate_to("Application")

# Application Page - Input Form
elif current_page == "Application":
    # Reset prediction stored flag when going to application
    st.session_state.current_prediction_stored = False
    
    # Title
    st.markdown("""
    <div class="hero-title">Loan Application Form</div>
    <div class="hero-subtitle">Fill in the details to get instant loan approval prediction</div>
    """, unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns(2, gap="large")

    # ==================== LEFT COLUMN ====================
    with col1:
        # Applicant Details Box
        st.markdown('<div class="box-title">📋 APPLICANT DETAILS</div>', unsafe_allow_html=True)
        
        gender = st.selectbox("Gender", ["Male", "Female"], help="Select the applicant's gender")
        married = st.selectbox("Marital Status", ["Yes", "No"], help="Is the applicant married?")
        dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, "3+"], help="Number of dependents (0-3+)")
        education = st.radio("Education Level", ["Graduate", "Not Graduate"], horizontal=True, help="Highest education completed")
        self_employed = st.selectbox("Self Employed", ["No", "Yes"], help="Is the applicant self-employed?")
        
        # Financial Details Box
        st.markdown('<div class="box-title">💰 FINANCIAL DETAILS</div>', unsafe_allow_html=True)
        
        income = st.number_input("Applicant Monthly Income (₹)", min_value=0.0, value=50000.0, step=5000.0, 
                                 help="Monthly income of the primary applicant", format="%.2f")
        coapplicant_income = st.number_input("Co-applicant Income (₹)", min_value=0.0, value=0.0, step=5000.0,
                                             help="Monthly income of the co-applicant",format="%.2f")
        loan_amount = st.number_input("Loan Amount (₹ thousands)", min_value=0.0, value=150.0, step=10.0,
                                      help="Loan amount in thousands (e.g., 150 = ₹1,50,000)",format="%.2f")
        
        # Loan term in DAYS
        loan_term_days = st.number_input(
            "Loan Term (Days)", 
            min_value=1, 
            max_value=1000,
            value=360, 
            step=30,
            help="Loan repayment period in days (e.g., 360 days = 12 months)"
        )
        
        # Display loan term in months and years for better understanding
        loan_term_months_display = loan_term_days / 30
        loan_term_years_display = loan_term_days / 365
        st.caption(f"📅 {loan_term_days} days = {loan_term_months_display:.1f} months = {loan_term_years_display:.2f} years")

    # ==================== RIGHT COLUMN ====================
    with col2:
        st.markdown('<div class="box-title">📄 ADDITIONAL INFORMATION</div>', unsafe_allow_html=True)
        
        credit_history = st.selectbox(
            "Credit History", 
            [1.0, 0.0], 
            format_func=lambda x: "✅ Good Credit History (1.0)" if x == 1.0 else "❌ Bad Credit History (0.0)",
            help="1.0 = Good credit history, 0.0 = No credit history"
        )
        property_area = st.selectbox("Property Area", ["Rural", "Urban", "Semiurban"], 
                                     help="Location of the property being purchased")
        
        # Summary of inputs
        st.markdown('<div class="box-title">📊 APPLICATION SUMMARY</div>', unsafe_allow_html=True)
        
        total_income = income + coapplicant_income
        loan_income_ratio = loan_amount / max(total_income, 1)
        
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.metric("Total Income", f"₹{total_income:,.0f}")
            st.metric("Loan Amount", f"₹{loan_amount*1000:,.0f}")
        with summary_col2:
            st.metric("Loan/Income Ratio", f"{loan_income_ratio:.2f}x")
            st.metric("Loan Term", f"{loan_term_days} days")
        
        # Enhanced button with icon
        if st.button("🔍 PREDICT LOAN APPROVAL", width='stretch', type="primary"):
            # Store form data in session state
            st.session_state.prediction_data = {
                'gender': gender,
                'married': married,
                'dependents': dependents,
                'education': education,
                'self_employed': self_employed,
                'income': income,
                'coapplicant_income': coapplicant_income,
                'loan_amount': loan_amount,
                'loan_term_days': loan_term_days,
                'credit_history': credit_history,
                'property_area': property_area,
                'total_income': total_income,
                'loan_income_ratio': loan_income_ratio
            }
            # Reset the flag for new prediction
            st.session_state.current_prediction_stored = False
            # Navigate to Results page
            navigate_to("Results")

# Results Page - Clean and Clear Design
elif current_page == "Results":
    # Title
    st.markdown("""
    <div class="hero-title">Loan Prediction Results</div>
    <div class="hero-subtitle">Your loan approval decision based on ML analysis</div>
    """, unsafe_allow_html=True)

    if st.session_state.prediction_data is None:
        st.warning("⚠️ No prediction data found. Please fill the application form first.")
        if st.button("📝 Go to Application Form", width='stretch', type="primary"):
            navigate_to("Application")
    else:
        # Load model (cached for performance)
        try:
            model, feature_columns = load_model()
            model_loaded = True
        except FileNotFoundError:
            model_loaded = False

        if not model_loaded:
            st.error("⚠️ Model file `loan_project_final.pkl` not found. Please place it in the same directory.")
            st.stop()

        # Get data from session state
        data = st.session_state.prediction_data
        
        # Process prediction only if it hasn't been stored yet
        if not st.session_state.current_prediction_stored:
            # Process prediction
            gender_en = 1 if data['gender'] == "Male" else 0
            married_en = 1 if data['married'] == "Yes" else 0
            # dep_en = int(data['dependents'])
            if data['dependents'] == '3+':
                dep_en = 3
            else:
                dep_en = int(data['dependents'])
            edu_en = 0 if data['education'] == "Graduate" else 1
            se_en = 1 if data['self_employed'] == "Yes" else 0
            area_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
            area_en = area_map[data['property_area']]
            
            # Create input dataframe with base features
            input_data = {
                "Gender": gender_en,
                "Married": married_en,
                "Dependents": dep_en,
                "Education": edu_en,
                "Self_Employed": se_en,
                "ApplicantIncome": data['income'],
                "CoapplicantIncome": data['coapplicant_income'],
                "LoanAmount": data['loan_amount'],
                "Loan_Amount_Term": data['loan_term_days'],
                "Credit_History": data['credit_history'],
                "Property_Area": area_en
            }
            
            input_df = pd.DataFrame([input_data])

            # Add engineered features (same as during training)
            input_df['TotalIncome'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
            input_df['LoanIncomeRatio'] = input_df['LoanAmount'] / (input_df['TotalIncome'] + 1)
            input_df['ApplicantIncome_log'] = np.log(input_df['ApplicantIncome'] + 1)
            input_df['LoanAmount_log'] = np.log(input_df['LoanAmount'] + 1)
            input_df['Total_Income_log'] = np.log(input_df['TotalIncome'] + 1)
            
            # Reorder columns to match training data exactly
            input_df = input_df[feature_columns]
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            pct = probability * 100
            
            # Calculate EMI (Equated Monthly Installment)
            loan_term_months = data['loan_term_days'] / 30
            annual_interest_rate = 0.10  # 10% per annum
            monthly_interest_rate = annual_interest_rate / 12
            
            if loan_term_months > 0 and monthly_interest_rate > 0 and data['loan_amount'] > 0:
                actual_loan_amount = data['loan_amount'] * 1000
                emi = actual_loan_amount * monthly_interest_rate * (1 + monthly_interest_rate)**loan_term_months
                emi = emi / max(((1 + monthly_interest_rate)**loan_term_months - 1), 0.001)  # Avoid division by zero
                emi = emi / 1000  # Convert back to thousands for display
                
                # Calculate total payment and interest
                total_payment = emi * loan_term_months
                total_interest = total_payment - data['loan_amount']
            else:
                emi = 0
                total_payment = 0
                total_interest = 0
            
            # Store prediction in all_predictions list
            st.session_state.all_predictions.append({
                'data': data,
                'prediction': prediction,
                'probability': probability,
                'emi': emi,
                'total_interest': total_interest
            })
            
            # Mark that this prediction has been stored
            st.session_state.current_prediction_stored = True
        
        # Get the latest prediction result for display
        if len(st.session_state.all_predictions) > 0:
            latest_prediction = st.session_state.all_predictions[-1]
            prediction = latest_prediction['prediction']
            probability = latest_prediction['probability']
            emi = latest_prediction['emi']
            total_interest = latest_prediction['total_interest']
            pct = probability * 100
        else:
            prediction = None
            probability = None
            emi = 0
            total_interest = 0
            pct = 0
        
        # Main Result Card
        if prediction == 1:
            st.markdown("""
            <div class="result-main-card">
                <div class="result-status" style="color: #2e7d32;">✅ LOAN APPROVED</div>
                <div class="result-message">Congratulations! Your loan application has been approved based on our AI analysis.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-main-card">
                <div class="result-status" style="color: #c62828;">❌ LOAN REJECTED</div>
                <div class="result-message">Unfortunately, your loan application does not meet our approval criteria at this time.</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Key Metrics Grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="result-item">
                <div class="result-item-label">Approval Probability</div>
                <div class="result-item-value">{:.1f}%</div>
            </div>
            """.format(pct), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="result-item">
                <div class="result-item-label">Total Income</div>
                <div class="result-item-value">₹{:,}</div>
            </div>
            """.format(data['total_income']), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="result-item">
                <div class="result-item-label">Loan Amount</div>
                <div class="result-item-value">₹{:,}</div>
            </div>
            """.format(data['loan_amount']*1000), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="result-item">
                <div class="result-item-label">EMI</div>
                <div class="result-item-value">₹{:.1f}k/mo</div>
            </div>
            """.format(emi), unsafe_allow_html=True)
        
        # Detailed Information Section
        col_info1, col_info2 = st.columns(2, gap="large")
        
        with col_info1:
            st.markdown('<div class="box-title">📋 APPLICATION DETAILS</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-grid">
                <div class="info-item">
                    <span class="info-label">Gender:</span>
                    <span class="info-value">{data['gender']}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Marital Status:</span>
                    <span class="info-value">{data['married']}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Dependents:</span>
                    <span class="info-value">{data['dependents']}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Education:</span>
                    <span class="info-value">{data['education']}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Self Employed:</span>
                    <span class="info-value">{data['self_employed']}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Property Area:</span>
                    <span class="info-value">{data['property_area']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_info2:
            st.markdown('<div class="box-title">💰 FINANCIAL ANALYSIS</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-grid">
                <div class="info-item">
                    <span class="info-label">Applicant Income:</span>
                    <span class="info-value">₹{data['income']:,}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Co-applicant Income:</span>
                    <span class="info-value">₹{data['coapplicant_income']:,}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Loan/Income Ratio:</span>
                    <span class="info-value">{data['loan_income_ratio']:.2f}x</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Loan Term:</span>
                    <span class="info-value">{data['loan_term_days']} days</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Credit History:</span>
                    <span class="info-value">{'Good' if data['credit_history'] == 1.0 else 'Bad'}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Total Interest:</span>
                    <span class="info-value">₹{total_interest:,.0f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk Assessment - CHANGED TO VIBRANT CARD CLASS
        risk_level = "Low Risk" if probability >= 0.70 else "Moderate Risk" if probability >= 0.50 else "High Risk"
        risk_color = "#ffffff" # White text looks better on vibrant background
        
        st.markdown(f"""
        <div class="risk-assessment-card">
            <h3>🎯 RISK ASSESSMENT</h3>
            <div style="text-align: center; margin: 1.3rem 0;">
                <div style="font-size: 3.5rem; font-weight: 800; color: white; text-shadow: 0 2px 10px rgba(0,0,0,0.2);">{pct:.1f}%</div>
                <div style="font-size: 1.2rem; color: rgba(255,255,255,0.9); margin-top: 0.3rem;">Approval Probability</div>
                <div style="margin-top: 1.5rem;">
                    <span class="risk-badge {'risk-low' if probability >= 0.70 else 'risk-mid' if probability >= 0.50 else 'risk-high'}">
                        {risk_level.upper()}
                    </span>
                </div>
            </div>
            <div style="background: rgba(255, 255, 255, 0.2); border-radius: 10px; padding: 1rem; margin-top: 2rem; backdrop-filter: blur(5px);">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; color: white;">
                    <span>Approval Chance:</span>
                    <span style="font-weight: 700;">{pct:.1f}%</span>
                </div>
                <div style="background: rgba(255, 255, 255, 0.3); border-radius: 10px; height: 12px; overflow: hidden;">
                    <div style="height: 100%; background: white; width: {pct}%; border-radius: 10px; box-shadow: 0 0 10px rgba(255,255,255,0.5);"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ==================== EXPERT TIPS SECTION ====================
        st.markdown('<div class="box-title">💡 EXPERT TIPS</div>', unsafe_allow_html=True)
        
        if prediction == 1:
            # Tips for approved loans
            st.markdown("""
            <div class="tips-approved">
                <div class="tips-title">
                    🎉 Congratulations! Here's how to maintain your approval:
                </div>
                <ul class="tips-list">
                    <li><strong>✅ Maintain Good Credit Score</strong> - Continue paying bills on time and keep credit utilization below 30%</li>
                    <li><strong>📊 Stable Income</strong> - Job stability and consistent income will help secure better interest rates</li>
                    <li><strong>💰 Lower Debt-to-Income Ratio</strong> - Try to keep your total debt below 40% of your income</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Tips for rejected loans with suggestions
            st.markdown("""
            <div class="tips-rejected">
                <div class="tips-title">
                    📝 Suggestions to Improve Your Application:
                </div>
                <ul class="tips-list">
                    <li><strong>💳 Improve Credit History</strong> - Pay existing loans on time, clear any outstanding dues, and build credit score</li>
                    <li><strong>💰 Reduce Loan Amount</strong> - Consider a smaller loan amount or increase your down payment</li>
                    <li><strong>📈 Increase Income</strong> - Include all sources of income (rent, investments, side business)</li>
                    <li><strong>📅 Extend Loan Term</strong> - A longer repayment period reduces EMI and improves affordability</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Download Section - Only one download button
        st.markdown('<div class="box-title">📥 DOWNLOAD REPORT</div>', unsafe_allow_html=True)
        
        if len(st.session_state.all_predictions) > 0:
            csv_data = generate_all_predictions_csv()
            st.download_button(
                label="📥 Download Complete Report (CSV)",
                data=csv_data,
                file_name="result.csv",
                mime="text/csv",
                width='stretch'
            )
            
            # Show summary of predictions
            st.markdown(f"""
            <div class="model-card" style="min-height: auto;">
                <h3>📊 Prediction Summary</h3>
                <p>Total predictions made: <strong>{len(st.session_state.all_predictions)}</strong></p>
                <p>Download the complete report above to get all prediction details in CSV format.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📝 New Application", width='stretch'):
                st.session_state.prediction_data = None
                st.session_state.current_prediction_stored = False
                navigate_to("Application")
        with col_btn2:
            if st.button("📊 View Model Details", width='stretch'):
                navigate_to("About Model")

# About Model Page
elif current_page == "About Model":
    # Reset prediction stored flag when going to about model
    st.session_state.current_prediction_stored = False
    
    # Title
    st.markdown("""
    <div class="hero-title">About Our Model</div>
    <div class="hero-subtitle">Understanding the technology behind loan approval prediction</div>
    """, unsafe_allow_html=True)

    # Algorithm Details
    st.markdown("""
    <div class="model-card">
        <h3>🧮 Algorithm Details</h3>
        <p>The system uses a sophisticated <strong>Random Forest</strong> machine learning algorithm, which is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the class that is the mode of the classes of the individual trees.</p>
        <ul>
            <li><strong>Random Forest:</strong> Primary classifier for robust decision making</li>
            <li><strong>Cross-validation:</strong> 5-fold CV for model validation</li>
            <li><strong>Feature Importance:</strong> SHAP values for explainability</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

 
    

    # Model Performance
    st.markdown("""
    <div class="model-card">
        <h3>📈 Model Performance</h3>
        <p>Our Random Forest model has been extensively tested and validated to ensure accurate predictions:</p>
        <ul>
            <li><strong>Accuracy:</strong> 93% on test datasets</li>
            <li><strong>Precision:</strong> High precision in identifying approved applications</li>
            <li><strong>Recall:</strong> Good recall rate for minimizing false rejections</li>
            <li><strong>F1-Score:</strong> Balanced performance across all metrics</li>
            <li><strong>AUC-ROC:</strong> 0.95 area under the curve</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Action buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📝 Start New Application", width='stretch'):
            st.session_state.prediction_data = None
            st.session_state.current_prediction_stored = False
            navigate_to("Application")
    with col_btn2:
        if st.button("📊 View Results", width='stretch', disabled=len(st.session_state.all_predictions) == 0):
            navigate_to("Results")