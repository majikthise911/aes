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
REPORTS_DIR = os.path.join(DATA_DIR, 'reports')
DATA_FILE = os.path.join(DATA_DIR, 'proposals.json')
REPORTS_FILE = os.path.join(DATA_DIR, 'analysis_reports.json')

# Create directories if they don't exist
os.makedirs(PROPOSALS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

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


def save_analysis_report(report_data: dict):
    """Save an analysis report with timestamp."""
    from datetime import datetime

    try:
        # Load existing reports
        reports = load_analysis_reports()

        # Add timestamp and ID
        report_data['timestamp'] = datetime.now().isoformat()
        report_data['id'] = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Add to beginning of list (newest first)
        reports.insert(0, report_data)

        # Keep only last 20 reports to avoid file bloat
        reports = reports[:20]

        # Save
        with open(REPORTS_FILE, 'w') as f:
            json.dump(reports, f, indent=2)

        return report_data['id']
    except Exception as e:
        st.error(f"Could not save report: {e}")
        return None


def load_analysis_reports():
    """Load all saved analysis reports."""
    if os.path.exists(REPORTS_FILE):
        try:
            with open(REPORTS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            return []
    return []


def get_report_by_id(report_id: str):
    """Get a specific report by ID."""
    reports = load_analysis_reports()
    for report in reports:
        if report.get('id') == report_id:
            return report
    return None


def delete_report(report_id: str):
    """Delete a report by ID."""
    try:
        reports = load_analysis_reports()
        reports = [r for r in reports if r.get('id') != report_id]
        with open(REPORTS_FILE, 'w') as f:
            json.dump(reports, f, indent=2)
        return True
    except Exception:
        return False

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