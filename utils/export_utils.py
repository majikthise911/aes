import pandas as pd
import json
from io import BytesIO
from typing import Dict, List
import plotly.express as px
import plotly.graph_objects as go

class ExportUtils:
    """Utilities for exporting dashboard data and visualizations."""

    @staticmethod
    def create_excel_export(proposals: List[Dict]) -> BytesIO:
        """Create a comprehensive Excel export with multiple sheets."""
        output = BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for proposal in proposals:
                row = {
                    'Project Name': proposal.get('project_info', {}).get('project_name', 'N/A'),
                    'Location': f"{proposal.get('project_info', {}).get('location', {}).get('city', 'N/A')}, {proposal.get('project_info', {}).get('location', {}).get('state', 'N/A')}",
                    'County': proposal.get('project_info', {}).get('location', {}).get('county', 'N/A'),
                    'Technology': proposal.get('technology', {}).get('type', 'N/A'),
                    'AC Capacity (MW)': proposal.get('capacity', {}).get('ac_mw', 'N/A'),
                    'DC Capacity (MW)': proposal.get('capacity', {}).get('dc_mw', 'N/A'),
                    'Storage (MWh)': proposal.get('capacity', {}).get('storage_mwh', 'N/A'),
                    'Total Cost ($)': proposal.get('costs', {}).get('total_project_cost', 'N/A'),
                    'Cost per Watt DC ($/W)': proposal.get('costs', {}).get('cost_per_watt_dc', 'N/A'),
                }
                summary_data.append(row)

            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

            # Equipment sheet
            equipment_data = []
            for proposal in proposals:
                equipment = proposal.get('equipment', {})
                row = {
                    'Project Name': proposal.get('project_info', {}).get('project_name', 'N/A'),
                    'Module Manufacturer': equipment.get('modules', {}).get('manufacturer', 'N/A'),
                    'Module Model': equipment.get('modules', {}).get('model', 'N/A'),
                    'Module Wattage': equipment.get('modules', {}).get('wattage', 'N/A'),
                    'Inverter Manufacturer': equipment.get('inverters', {}).get('manufacturer', 'N/A'),
                    'Inverter Model': equipment.get('inverters', {}).get('model', 'N/A'),
                    'Racking Manufacturer': equipment.get('racking', {}).get('manufacturer', 'N/A'),
                    'Racking Type': equipment.get('racking', {}).get('type', 'N/A'),
                    'Battery Manufacturer': equipment.get('batteries', {}).get('manufacturer', 'N/A'),
                    'Battery Model': equipment.get('batteries', {}).get('model', 'N/A'),
                }
                equipment_data.append(row)

            pd.DataFrame(equipment_data).to_excel(writer, sheet_name='Equipment', index=False)

            # Costs sheet
            cost_data = []
            for proposal in proposals:
                costs = proposal.get('costs', {})
                breakdown = costs.get('cost_breakdown', {})
                row = {
                    'Project Name': proposal.get('project_info', {}).get('project_name', 'N/A'),
                    'Total Cost ($)': costs.get('total_project_cost', 'N/A'),
                    'Equipment Cost ($)': breakdown.get('equipment', 'N/A'),
                    'Labor Cost ($)': breakdown.get('labor', 'N/A'),
                    'Materials Cost ($)': breakdown.get('materials', 'N/A'),
                    'Development Cost ($)': breakdown.get('development', 'N/A'),
                    'Other Cost ($)': breakdown.get('other', 'N/A'),
                    'Cost per Watt DC ($/W)': costs.get('cost_per_watt_dc', 'N/A'),
                    'Cost per Watt AC ($/W)': costs.get('cost_per_watt_ac', 'N/A'),
                }
                cost_data.append(row)

            pd.DataFrame(cost_data).to_excel(writer, sheet_name='Costs', index=False)

        output.seek(0)
        return output

    @staticmethod
    def create_pdf_report(proposals: List[Dict]) -> str:
        """Create a PDF report summary (placeholder for future implementation)."""
        # This would require reportlab or similar library
        return "PDF export feature coming soon!"

    @staticmethod
    def export_chart_as_image(fig, filename: str) -> bytes:
        """Export Plotly chart as image."""
        return fig.to_image(format="png", width=1200, height=600)