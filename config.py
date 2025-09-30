import streamlit as st
import os
import json
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), 'config', '.env')
load_dotenv(env_path)

# Data directories
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PROPOSALS_DIR = os.path.join(DATA_DIR, 'proposals')
DATA_FILE = os.path.join(DATA_DIR, 'proposals.json')

# Create directories if they don't exist
os.makedirs(PROPOSALS_DIR, exist_ok=True)

def save_proposals():
    """Save current proposals to file."""
    try:
        if 'data_manager' in st.session_state:
            proposals = st.session_state.data_manager.get_all_proposals()
            with open(DATA_FILE, 'w') as f:
                json.dump(proposals, f, indent=2)
    except Exception as e:
        st.error(f"Could not save proposals: {e}")

def load_proposals():
    """Load proposals from file."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                saved_proposals = json.load(f)
            return saved_proposals
        except Exception as e:
            st.warning(f"Could not load previous proposals: {e}")
            return []
    return []

def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 2rem;
            color: #1f77b4;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
    </style>
    """, unsafe_allow_html=True)