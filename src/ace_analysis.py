"""
ACE Analysis module for processing ACE logs and generating clarification logs.
Generates ACE logs from uploaded proposal scope data.
"""
import pandas as pd
import json
from typing import Dict, List, Optional, Callable
from io import BytesIO


def generate_ace_log_from_proposals(proposals: List[Dict]) -> pd.DataFrame:
    """
    Generate ACE log from uploaded proposals.
    Extracts assumptions, exclusions, clarifications from each proposal's scope data.
    """
    ace_items = []

    for proposal in proposals:
        epc_name = proposal.get('epc_contractor', {}).get('company_name', 'Unknown EPC')
        scope = proposal.get('scope', {})

        # Extract assumptions
        for item in scope.get('assumptions', []):
            if item and str(item).strip():
                ace_items.append({
                    'ACE Item': str(item).strip(),
                    'Type': 'Assumption',
                    'Scope': categorize_scope_item(str(item)),
                    'EPC': epc_name,
                    'Risk?': '',
                    'AES SME': '',
                    'Point Person': '',
                    'Internal Note': '',
                    'Response to EPC': ''
                })

        # Extract exclusions
        for item in scope.get('exclusions', []):
            if item and str(item).strip():
                ace_items.append({
                    'ACE Item': str(item).strip(),
                    'Type': 'Exclusion',
                    'Scope': categorize_scope_item(str(item)),
                    'EPC': epc_name,
                    'Risk?': '',
                    'AES SME': '',
                    'Point Person': '',
                    'Internal Note': '',
                    'Response to EPC': ''
                })

        # Extract clarifications
        for item in scope.get('clarifications', []):
            if item and str(item).strip():
                ace_items.append({
                    'ACE Item': str(item).strip(),
                    'Type': 'Clarification',
                    'Scope': categorize_scope_item(str(item)),
                    'EPC': epc_name,
                    'Risk?': '',
                    'AES SME': '',
                    'Point Person': '',
                    'Internal Note': '',
                    'Response to EPC': ''
                })

    if not ace_items:
        return pd.DataFrame(columns=['ACE Item', 'Type', 'Scope', 'EPC', 'Risk?',
                                     'AES SME', 'Point Person', 'Internal Note', 'Response to EPC'])

    return pd.DataFrame(ace_items)


def categorize_scope_item(item_text: str) -> str:
    """Auto-categorize an ACE item based on keywords."""
    text_lower = item_text.lower()

    if any(kw in text_lower for kw in ['inverter', 'transformer', 'switchgear', 'medium voltage', 'mv ', 'hv ', 'high voltage', 'ac cable', 'ac electrical']):
        return 'AC Electrical Systems'
    elif any(kw in text_lower for kw in ['module', 'panel', 'dc cable', 'string', 'combiner', 'dc electrical']):
        return 'DC Electrical Systems'
    elif any(kw in text_lower for kw in ['grading', 'civil', 'road', 'fence', 'erosion', 'drainage', 'excavation', 'foundation', 'pile', 'concrete']):
        return 'Civil Works'
    elif any(kw in text_lower for kw in ['tracker', 'racking', 'torque tube', 'mechanical']):
        return 'Mechanical'
    elif any(kw in text_lower for kw in ['substation', 'relay', 'protection', 'metering']):
        return 'Substation'
    elif any(kw in text_lower for kw in ['interconnect', 'gen-tie', 'transmission', 'utility']):
        return 'Interconnection'
    elif any(kw in text_lower for kw in ['battery', 'bess', 'storage', 'energy storage']):
        return 'Energy Storage'
    elif any(kw in text_lower for kw in ['permit', 'insurance', 'bond', 'tax', 'general', 'overhead', 'mobilization']):
        return 'General & Administrative'
    else:
        return 'Other'


def assess_ace_items_with_ai(df: pd.DataFrame, ai_client, progress_callback: Callable = None) -> pd.DataFrame:
    """
    Use AI to assess risk and generate internal notes for ACE items.
    Processes items in batches for efficiency.
    """
    if df.empty:
        return df

    # Process in batches of 10 items
    batch_size = 10
    total_items = len(df)
    results = []

    for i in range(0, total_items, batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_items = []

        for _, row in batch.iterrows():
            batch_items.append({
                'index': row.name,
                'item': row['ACE Item'],
                'type': row['Type'],
                'scope': row['Scope'],
                'epc': row['EPC']
            })

        # Build prompt for batch
        items_text = "\n".join([
            f"{idx+1}. [{item['type']}] {item['item']}"
            for idx, item in enumerate(batch_items)
        ])

        prompt = f"""You are an expert solar/renewable energy project analyst reviewing EPC proposal scope items.

For each item below, assess:
1. RISK: Is this item a potential risk to the project owner (AES)? Answer: Yes, TBD (need more info), or No
2. INTERNAL NOTE: Brief note (1-2 sentences) explaining your risk assessment and what AES should consider

Items to assess:
{items_text}

Respond in JSON format:
[
  {{"index": 1, "risk": "Yes/TBD/No", "note": "Your assessment note"}},
  ...
]

Risk Guidelines:
- YES: Item clearly transfers risk/cost to owner, excludes important scope, or has ambiguous language that could be exploited
- TBD: Item needs clarification or more context to assess properly
- NO: Standard industry practice, reasonable assumption, or clearly defined with no hidden risk"""

        try:
            response = ai_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model_tier="fast",
                temperature=0.2,
                max_tokens=2000,
                system_prompt="You are an expert EPC contract analyst. Respond only with valid JSON."
            )

            # Parse JSON response
            # Clean up response - find JSON array
            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()

            assessments = json.loads(response_text)

            for assessment in assessments:
                idx = assessment.get('index', 0) - 1
                if 0 <= idx < len(batch_items):
                    results.append({
                        'df_index': batch_items[idx]['index'],
                        'risk': assessment.get('risk', 'TBD'),
                        'note': assessment.get('note', '')
                    })

        except Exception as e:
            # On error, mark batch as TBD
            for item in batch_items:
                results.append({
                    'df_index': item['index'],
                    'risk': 'TBD',
                    'note': f'AI assessment pending - please review manually'
                })

        # Progress callback
        if progress_callback:
            progress_callback(min(i + batch_size, total_items), total_items)

    # Apply results to dataframe
    df_copy = df.copy()
    for result in results:
        df_copy.loc[result['df_index'], 'Risk?'] = result['risk']
        df_copy.loc[result['df_index'], 'Internal Note'] = result['note']

    return df_copy


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
