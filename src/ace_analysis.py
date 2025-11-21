"""
ACE Analysis module for processing ACE logs and generating clarification logs.
"""
import pandas as pd
from typing import Dict, List, Optional
from io import BytesIO


def get_risk_color(risk_value: str) -> str:
    """Return color based on risk value."""
    if pd.isna(risk_value):
        return ""
    risk = str(risk_value).strip().upper()
    if risk == "YES":
        return "background-color: #ffcccc"  # Red
    elif risk == "TBD":
        return "background-color: #fff3cd"  # Yellow
    elif risk == "NO":
        return "background-color: #d4edda"  # Green
    return ""


def style_ace_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply conditional styling to ACE dataframe."""
    def highlight_risk(row):
        risk = str(row.get('Risk?', '')).strip().upper() if pd.notna(row.get('Risk?')) else ''
        if risk == "YES":
            return ['background-color: #ffcccc'] * len(row)
        elif risk == "TBD":
            return ['background-color: #fff3cd'] * len(row)
        return [''] * len(row)

    return df.style.apply(highlight_risk, axis=1)


def calculate_ace_summary(df: pd.DataFrame) -> Dict:
    """Calculate summary statistics from ACE log."""
    summary = {
        'total_items': len(df),
        'risk_yes': 0,
        'risk_tbd': 0,
        'risk_no': 0,
        'risk_blank': 0,
        'categories': {},
        'needs_clarification': 0
    }

    if 'Risk?' in df.columns:
        for val in df['Risk?']:
            if pd.isna(val) or str(val).strip() == '':
                summary['risk_blank'] += 1
            elif str(val).strip().upper() == 'YES':
                summary['risk_yes'] += 1
            elif str(val).strip().upper() == 'TBD':
                summary['risk_tbd'] += 1
            elif str(val).strip().upper() == 'NO':
                summary['risk_no'] += 1

        summary['needs_clarification'] = summary['risk_yes'] + summary['risk_tbd']

    # Count by scope category
    if 'Scope' in df.columns:
        for scope in df['Scope'].dropna():
            scope_str = str(scope).strip()
            if scope_str:
                summary['categories'][scope_str] = summary['categories'].get(scope_str, 0) + 1

    return summary


def filter_for_clarification(df: pd.DataFrame) -> pd.DataFrame:
    """Filter ACE log to items requiring clarification (Risk = Yes or TBD)."""
    if 'Risk?' not in df.columns:
        return df

    def needs_clarification(val):
        if pd.isna(val):
            return False
        return str(val).strip().upper() in ['YES', 'TBD']

    return df[df['Risk?'].apply(needs_clarification)].copy()


def generate_clarification_log(df: pd.DataFrame, project_name: str = "Project",
                                epc_name: str = "EPC") -> pd.DataFrame:
    """Generate clarification log from filtered ACE items."""
    filtered = filter_for_clarification(df)

    if filtered.empty:
        return pd.DataFrame()

    # Create clarification log structure
    clar_log = pd.DataFrame({
        'No.': range(1, len(filtered) + 1),
        'Proposal Section Reference': filtered['Scope'].values if 'Scope' in filtered.columns else '',
        'Assumption/Exclusion': filtered['ACE Item'].values if 'ACE Item' in filtered.columns else '',
        'AES Position': filtered['Response to EPC'].fillna('').values if 'Response to EPC' in filtered.columns else '',
        'AES Response': '',
        'EPC Response': ''
    })

    return clar_log


def export_to_excel(df: pd.DataFrame, project_name: str = "Project",
                    epc_name: str = "EPC", log_type: str = "ACE") -> BytesIO:
    """Export dataframe to Excel file."""
    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if log_type == "clarification":
            # Write header info
            header_df = pd.DataFrame({
                'A': ['', 'Assumptions & Exclusions Clarification Log', '', 'Project', 'EPC'],
                'B': ['', '', '', project_name, epc_name]
            })
            header_df.to_excel(writer, sheet_name='Clarification Log',
                             index=False, header=False, startrow=0)

            # Write actual data
            df.to_excel(writer, sheet_name='Clarification Log',
                       index=False, startrow=6)
        else:
            df.to_excel(writer, sheet_name='ACE Log', index=False)

    output.seek(0)
    return output


ACE_COLUMNS = [
    'ACE Item',
    'Scope',
    'Risk?',
    'AES SME',
    'Point Person',
    'Internal Note',
    'Response to EPC',
    'Change Log'
]

SCOPE_CATEGORIES = [
    'AC Electrical Systems',
    'Civil Works',
    'DC Electrical Systems',
    'General & Administrative',
    'Mechanical',
    'Substation',
    'Interconnection',
    'Energy Storage',
    'Other'
]

RISK_OPTIONS = ['Yes', 'TBD', 'No', '']
