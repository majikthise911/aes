import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from dotenv import load_dotenv
from src.pdf_processor import PDFProcessor
from src.gpt_extractor import GPTExtractor
from src.data_manager import DataManager

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), 'config', '.env')
load_dotenv(env_path)

# Page configuration
st.set_page_config(
    page_title="EPC Proposal Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()

if 'pdf_processor' not in st.session_state:
    st.session_state.pdf_processor = PDFProcessor()

if 'gpt_extractor' not in st.session_state:
    st.session_state.gpt_extractor = GPTExtractor()

# Custom CSS
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

def main():
    st.markdown('<div class="main-header">⚡ EPC Proposal Dashboard</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Proposals")

        # File upload section
        uploaded_files = st.file_uploader(
            "Upload EPC Proposals",
            type=['pdf', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload PDF proposals and Excel schedule of values"
        )

        if uploaded_files:
            process_uploaded_files(uploaded_files)

        st.divider()

        # Filters section
        st.header("🔍 Filters")
        apply_filters()

        st.divider()

        # Actions section
        st.header("⚙️ Actions")
        if st.button("Clear All Data", type="secondary"):
            st.session_state.data_manager.clear_all_proposals()
            st.rerun()

    # Main content area
    proposals = st.session_state.data_manager.get_all_proposals()

    if not proposals:
        st.info("👆 Upload EPC proposals using the sidebar to get started")
        return

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🗺️ Map View", "💰 Cost Analysis", "📋 Comparison", "📝 AI Report"])

    with tab1:
        show_overview(proposals)

    with tab2:
        show_map_view(proposals)

    with tab3:
        show_cost_analysis(proposals)

    with tab4:
        show_comparison_view(proposals)

    with tab5:
        show_ai_report(proposals)

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF and Excel files."""
    with st.spinner("Processing uploaded files..."):
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.type == "application/pdf":
                    # Process PDF
                    text = st.session_state.pdf_processor.extract_text_from_pdf(uploaded_file)
                    clean_text = st.session_state.pdf_processor.clean_text(text)

                    # Extract data using GPT
                    proposal_data = st.session_state.gpt_extractor.extract_project_data(clean_text)

                    # Add to data manager
                    was_added = st.session_state.data_manager.add_proposal(proposal_data, uploaded_file.name)
                    if was_added:
                        st.success(f"Successfully processed: {uploaded_file.name}")
                    else:
                        st.info(f"File '{uploaded_file.name}' already processed (skipping duplicate)")

                elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
                    # Process Excel - could enhance this to extract cost data
                    excel_data = st.session_state.pdf_processor.extract_excel_data(uploaded_file)
                    st.success(f"Excel file '{uploaded_file.name}' uploaded successfully")
                    # For now, just store the filename - could integrate cost data extraction later

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

def apply_filters():
    """Apply filters to proposals."""
    proposals = st.session_state.data_manager.get_all_proposals()

    if not proposals:
        return

    # Technology filter
    tech_options = list(set([p.get('technology', {}).get('type', 'Unknown') for p in proposals]))
    selected_tech = st.selectbox("Technology Type", ["All"] + tech_options)

    # Capacity range filter
    capacities = [p.get('capacity', {}).get('ac_mw') for p in proposals if p.get('capacity', {}).get('ac_mw')]
    if capacities:
        min_capacity = float(min(capacities))
        max_capacity = float(max(capacities))

        # Only show slider if we have a range
        if min_capacity != max_capacity:
            min_cap, max_cap = st.slider(
                "AC Capacity Range (MW)",
                min_value=min_capacity,
                max_value=max_capacity,
                value=(min_capacity, max_capacity)
            )
        else:
            st.info(f"All proposals have the same capacity: {min_capacity} MW")

    # State filter
    states = list(set([p.get('project_info', {}).get('location', {}).get('state', 'Unknown') for p in proposals]))
    selected_state = st.selectbox("State", ["All"] + states)

    # Apply filters (implementation would go here)
    # For now, this is a placeholder for the filtering logic

def show_overview(proposals):
    """Display overview dashboard."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Proposals", len(proposals))

    with col2:
        total_capacity = sum([p.get('capacity', {}).get('ac_mw', 0) for p in proposals if p.get('capacity', {}).get('ac_mw')])
        st.metric("Total AC Capacity", f"{total_capacity:.1f} MW")

    with col3:
        tech_types = len(set([p.get('technology', {}).get('type', 'Unknown') for p in proposals]))
        st.metric("Technology Types", tech_types)

    with col4:
        avg_cost = sum([p.get('costs', {}).get('cost_per_watt_dc', 0) for p in proposals if p.get('costs', {}).get('cost_per_watt_dc')]) / len([p for p in proposals if p.get('costs', {}).get('cost_per_watt_dc')])
        if avg_cost > 0:
            st.metric("Avg Cost/W DC", f"${avg_cost:.2f}")
        else:
            st.metric("Avg Cost/W DC", "N/A")

    # Technology breakdown chart
    st.subheader("Technology Breakdown")
    tech_data = {}
    for proposal in proposals:
        tech = proposal.get('technology', {}).get('type', 'Unknown')
        tech_data[tech] = tech_data.get(tech, 0) + 1

    if tech_data:
        fig = px.pie(values=list(tech_data.values()), names=list(tech_data.keys()), title="Projects by Technology")
        st.plotly_chart(fig, use_container_width=True)

    # Recent proposals table
    st.subheader("Recent Proposals")
    df = st.session_state.data_manager.create_comparison_dataframe()
    if not df.empty:
        st.dataframe(df.head(10), use_container_width=True)

def show_map_view(proposals):
    """Display map view of projects."""
    st.subheader("🗺️ Project Locations")

    # Create map centered on US
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

    markers_added = 0
    projects_without_coords = []

    # Add markers for each proposal
    for i, proposal in enumerate(proposals):
        location = proposal.get('project_info', {}).get('location', {})
        coords = location.get('coordinates', {})
        project_name = proposal.get('project_info', {}).get('project_name', f'Unknown Project {i+1}')

        if coords and coords.get('lat') and coords.get('lon'):
            popup_text = f"""
            <b>{project_name}</b><br>
            Technology: {proposal.get('technology', {}).get('type', 'Unknown')}<br>
            Capacity: {proposal.get('capacity', {}).get('ac_mw', 'N/A')} MW AC<br>
            Location: {location.get('city', 'N/A')}, {location.get('state', 'N/A')}
            """

            folium.Marker(
                [coords['lat'], coords['lon']],
                popup=popup_text,
                tooltip=project_name,
                icon=folium.Icon(color='green', icon='bolt')
            ).add_to(m)
            markers_added += 1

        else:
            # Try to extract coordinates from address if available
            address_parts = []
            if location.get('city'): address_parts.append(location['city'])
            if location.get('state'): address_parts.append(location['state'])
            if location.get('address'): address_parts.append(location['address'])

            address_string = ', '.join(address_parts)

            if address_string and address_string != ', ':
                # Try to get coordinates from GPT
                try:
                    extracted_coords = st.session_state.gpt_extractor.extract_location_coordinates(address_string)
                    if extracted_coords:
                        popup_text = f"""
                        <b>{project_name}</b><br>
                        Technology: {proposal.get('technology', {}).get('type', 'Unknown')}<br>
                        Capacity: {proposal.get('capacity', {}).get('ac_mw', 'N/A')} MW AC<br>
                        Location: {address_string}<br>
                        <i>(Coordinates estimated from address)</i>
                        """

                        folium.Marker(
                            [extracted_coords['lat'], extracted_coords['lon']],
                            popup=popup_text,
                            tooltip=f"{project_name} (estimated location)",
                            icon=folium.Icon(color='orange', icon='bolt')
                        ).add_to(m)
                        markers_added += 1
                    else:
                        projects_without_coords.append((project_name, address_string))
                except:
                    projects_without_coords.append((project_name, address_string))
            else:
                projects_without_coords.append((project_name, "No location data"))

    # Display map
    map_data = st_folium(m, width=700, height=500)

    # Show info about markers
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Projects Mapped", markers_added)
    with col2:
        st.metric("Projects Without Coordinates", len(projects_without_coords))

    # Show projects without coordinates
    if projects_without_coords:
        st.subheader("Projects Without Map Coordinates")
        for project_name, location_info in projects_without_coords:
            st.write(f"• **{project_name}**: {location_info}")
        st.info("💡 Try including more specific location information (coordinates, full addresses) in your proposals for better map visualization.")

def show_cost_analysis(proposals):
    """Display EPC cost analysis dashboard."""
    st.subheader("💰 EPC Cost Analysis")

    # Filter proposals with cost data
    cost_proposals = [p for p in proposals if p.get('costs', {}).get('total_project_cost')]

    if not cost_proposals:
        st.warning("No cost data available in uploaded proposals")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Total cost comparison by EPC
        costs = []
        epc_names = []
        colors = []

        for proposal in cost_proposals:
            costs.append(proposal['costs']['total_project_cost'])
            epc_name = proposal.get('epc_contractor', {}).get('company_name', 'Unknown EPC')
            epc_names.append(epc_name)
            # Color code: lowest cost in green, others in blue/red gradient

        # Sort by cost for better visualization
        sorted_data = sorted(zip(costs, epc_names))
        costs, epc_names = zip(*sorted_data)

        # Color gradient: green for lowest, red for highest
        colors = ['#2ca02c' if i == 0 else '#ff7f0e' if i == len(costs)-1 else '#1f77b4'
                 for i in range(len(costs))]

        fig = px.bar(x=epc_names, y=costs, title="Total Cost by EPC Contractor")
        fig.update_traces(marker_color=colors)
        fig.update_layout(xaxis_title="EPC Contractor", yaxis_title="Total Cost ($)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Cost per watt comparison by EPC
        cpw_data = []
        cpw_epc_names = []
        for proposal in cost_proposals:
            cpw = proposal.get('costs', {}).get('cost_per_watt_dc')
            if cpw:
                cpw_data.append(cpw)
                epc_name = proposal.get('epc_contractor', {}).get('company_name', 'Unknown EPC')
                cpw_epc_names.append(epc_name)

        if cpw_data:
            # Sort by cost per watt
            sorted_cpw_data = sorted(zip(cpw_data, cpw_epc_names))
            cpw_data, cpw_epc_names = zip(*sorted_cpw_data)

            colors = ['#2ca02c' if i == 0 else '#ff7f0e' if i == len(cpw_data)-1 else '#1f77b4'
                     for i in range(len(cpw_data))]

            fig = px.bar(x=cpw_epc_names, y=cpw_data, title="Cost per Watt DC by EPC")
            fig.update_traces(marker_color=colors)
            fig.update_layout(xaxis_title="EPC Contractor", yaxis_title="$/W DC")
            st.plotly_chart(fig, use_container_width=True)

    # EPC comparison metrics
    st.subheader("EPC Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)

    total_costs = [p['costs']['total_project_cost'] for p in cost_proposals]
    cpw_costs = [p.get('costs', {}).get('cost_per_watt_dc') for p in cost_proposals if p.get('costs', {}).get('cost_per_watt_dc')]

    with col1:
        st.metric("Lowest Total Cost", f"${min(total_costs):,.0f}")
    with col2:
        st.metric("Highest Total Cost", f"${max(total_costs):,.0f}")
    with col3:
        st.metric("Cost Spread", f"${max(total_costs) - min(total_costs):,.0f}")
    with col4:
        if cpw_costs:
            st.metric("Best $/W DC", f"${min(cpw_costs):.3f}")

    # Detailed cost breakdown table
    st.subheader("EPC Cost Breakdown Details")
    cost_df = []
    for proposal in cost_proposals:
        epc = proposal.get('epc_contractor', {})
        cost_info = {
            'EPC Contractor': epc.get('company_name', 'Unknown EPC'),
            'Total Cost ($)': f"${proposal['costs']['total_project_cost']:,.0f}",
            'Cost/W DC ($/W)': f"${proposal.get('costs', {}).get('cost_per_watt_dc', 0):.3f}" if proposal.get('costs', {}).get('cost_per_watt_dc') else 'N/A',
            'Equipment ($)': f"${proposal.get('costs', {}).get('cost_breakdown', {}).get('equipment', 0):,.0f}" if proposal.get('costs', {}).get('cost_breakdown', {}).get('equipment') else 'N/A',
            'Labor ($)': f"${proposal.get('costs', {}).get('cost_breakdown', {}).get('labor', 0):,.0f}" if proposal.get('costs', {}).get('cost_breakdown', {}).get('labor') else 'N/A',
            'Materials ($)': f"${proposal.get('costs', {}).get('cost_breakdown', {}).get('materials', 0):,.0f}" if proposal.get('costs', {}).get('cost_breakdown', {}).get('materials') else 'N/A',
            'Contact': epc.get('contact_person', 'N/A'),
            'Proposal Date': epc.get('proposal_date', 'N/A')
        }
        cost_df.append(cost_info)

    if cost_df:
        # Sort by total cost (convert back to numeric for sorting)
        cost_df_sorted = sorted(cost_df, key=lambda x: float(x['Total Cost ($)'].replace('$', '').replace(',', '')))
        st.dataframe(pd.DataFrame(cost_df_sorted), use_container_width=True)

        # Highlight the best option
        st.success(f"💡 **Recommended**: {cost_df_sorted[0]['EPC Contractor']} offers the lowest total cost at {cost_df_sorted[0]['Total Cost ($)']}")

def show_comparison_view(proposals):
    """Display EPC contractor comparison and selection dashboard."""
    st.subheader("📋 EPC Contractor Comparison & Selection")

    if not proposals:
        st.warning("No proposals available for comparison")
        return

    # Create tabs for different comparison views
    tab1, tab2, tab3 = st.tabs(["🏆 EPC Rankings", "📊 Detailed Comparison", "📈 Cost Breakdown"])

    with tab1:
        show_epc_rankings(proposals)

    with tab2:
        show_detailed_comparison(proposals)

    with tab3:
        show_cost_breakdown_by_epc(proposals)

def show_epc_rankings(proposals):
    """Show EPC contractors ranked by cost and other factors."""
    st.subheader("EPC Contractor Rankings")

    # Create ranking dataframe
    df = st.session_state.data_manager.create_epc_ranking_dataframe()

    if df.empty:
        st.warning("No EPC data available for ranking")
        return

    # Add ranking indicators
    st.write("### 🥇 Cost-Based Rankings (Lowest to Highest)")

    # Display with highlighting for best options
    st.dataframe(
        df.style.format({
            'Total Cost ($)': lambda x: f"${x:,.0f}" if pd.notnull(x) and x != 'N/A' else x,
            'Cost/Watt DC ($/W)': lambda x: f"${x:.3f}" if pd.notnull(x) and x != 'N/A' else x,
            'Equipment Cost ($)': lambda x: f"${x:,.0f}" if pd.notnull(x) and x != 'N/A' else x,
            'Labor Cost ($)': lambda x: f"${x:,.0f}" if pd.notnull(x) and x != 'N/A' else x
        }),
        use_container_width=True
    )

    # Show key insights
    cost_data = [p.get('costs', {}).get('total_project_cost') for p in proposals if p.get('costs', {}).get('total_project_cost')]
    if cost_data:
        lowest_cost = min(cost_data)
        highest_cost = max(cost_data)
        savings = highest_cost - lowest_cost

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Lowest Bid", f"${lowest_cost:,.0f}")
        with col2:
            st.metric("Highest Bid", f"${highest_cost:,.0f}")
        with col3:
            st.metric("Potential Savings", f"${savings:,.0f}")

def show_detailed_comparison(proposals):
    """Show detailed side-by-side comparison."""
    st.subheader("Detailed EPC Comparison")

    df = st.session_state.data_manager.create_comparison_dataframe()

    if df.empty:
        st.warning("No comparison data available")
        return

    st.dataframe(df, use_container_width=True)

    # Export functionality
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Comparison to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="epc_detailed_comparison.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("Export to Excel"):
            st.info("Excel export feature coming soon!")

def show_cost_breakdown_by_epc(proposals):
    """Show cost breakdown analysis by EPC contractor."""
    st.subheader("Cost Breakdown by EPC Contractor")

    # Group by EPC contractor
    epc_costs = {}
    for proposal in proposals:
        epc_name = proposal.get('epc_contractor', {}).get('company_name', 'Unknown EPC')
        costs = proposal.get('costs', {})

        if costs.get('total_project_cost'):
            epc_costs[epc_name] = {
                'total': costs.get('total_project_cost', 0),
                'equipment': costs.get('cost_breakdown', {}).get('equipment', 0),
                'labor': costs.get('cost_breakdown', {}).get('labor', 0),
                'materials': costs.get('cost_breakdown', {}).get('materials', 0),
                'development': costs.get('cost_breakdown', {}).get('development', 0),
                'other': costs.get('cost_breakdown', {}).get('other', 0)
            }

    if not epc_costs:
        st.warning("No cost breakdown data available")
        return

    # Create stacked bar chart
    categories = ['Equipment', 'Labor', 'Materials', 'Development', 'Other']
    epc_names = list(epc_costs.keys())

    fig = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, category in enumerate(categories):
        values = [epc_costs[epc].get(category.lower(), 0) for epc in epc_names]
        fig.add_trace(go.Bar(
            name=category,
            x=epc_names,
            y=values,
            marker_color=colors[i]
        ))

    fig.update_layout(
        barmode='stack',
        title='Cost Breakdown by EPC Contractor',
        xaxis_title='EPC Contractor',
        yaxis_title='Cost ($)',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show cost breakdown table
    st.write("### Cost Breakdown Table")
    breakdown_data = []
    for epc_name, costs in epc_costs.items():
        breakdown_data.append({
            'EPC Contractor': epc_name,
            'Total Cost': f"${costs['total']:,.0f}",
            'Equipment': f"${costs['equipment']:,.0f}" if costs['equipment'] else 'N/A',
            'Labor': f"${costs['labor']:,.0f}" if costs['labor'] else 'N/A',
            'Materials': f"${costs['materials']:,.0f}" if costs['materials'] else 'N/A',
            'Development': f"${costs['development']:,.0f}" if costs['development'] else 'N/A',
            'Other': f"${costs['other']:,.0f}" if costs['other'] else 'N/A'
        })

    st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True)

def show_ai_report(proposals):
    """Display AI-generated EPC recommendation report."""
    st.subheader("🤖 AI-Powered EPC Recommendation Report")

    if len(proposals) < 2:
        st.warning("⚠️ You need at least 2 proposals to generate a meaningful comparison report.")
        st.info("Upload more EPC proposals to get AI-powered recommendations and detailed analysis.")
        return

    # Check if we have cost data for meaningful analysis
    cost_proposals = [p for p in proposals if p.get('costs', {}).get('total_project_cost')]
    if len(cost_proposals) < 2:
        st.warning("⚠️ Need cost data from at least 2 proposals for comprehensive analysis.")
        st.info("Make sure your uploaded proposals contain pricing information for better AI analysis.")

    # Report generation section
    col1, col2 = st.columns([3, 1])

    with col1:
        st.write("### Generate Comprehensive EPC Analysis Report")
        st.write("Our AI consultant will analyze all uploaded proposals and provide detailed recommendations based on:")
        st.write("- **Cost Analysis** - Compare pricing, value proposition, and financial risks")
        st.write("- **Technical Evaluation** - Assess equipment quality, technology choices, and installation approach")
        st.write("- **Risk Assessment** - Identify potential risks and mitigation strategies")
        st.write("- **Strategic Recommendations** - Provide actionable insights for EPC selection")

    with col2:
        if st.button("🚀 Generate AI Report", type="primary", use_container_width=True):
            generate_report = True
        else:
            generate_report = False

    # Session state to store generated report
    if 'generated_report' not in st.session_state:
        st.session_state.generated_report = None
        st.session_state.report_timestamp = None

    # Generate report
    if generate_report:
        with st.spinner("🤖 AI Consultant analyzing proposals... This may take 30-60 seconds."):
            try:
                report = st.session_state.gpt_extractor.generate_epc_recommendation_report(proposals)
                st.session_state.generated_report = report
                st.session_state.report_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("✅ Report generated successfully!")
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")
                return

    # Display report if available
    if st.session_state.generated_report:
        st.divider()

        # Report header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"### 📄 EPC Recommendation Report")
            st.write(f"*Generated on {st.session_state.report_timestamp}*")

        with col2:
            # Export functionality
            if st.button("📥 Export Report", use_container_width=True):
                # Create downloadable report
                report_content = f"""
EPC CONTRACTOR RECOMMENDATION REPORT
Generated: {st.session_state.report_timestamp}
Project Analysis Dashboard

{st.session_state.generated_report}

---
Generated by AES EPC Proposal Dashboard
Powered by AI Analysis
"""
                st.download_button(
                    label="💾 Download Report as Text",
                    data=report_content,
                    file_name=f"epc_recommendation_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

        # Display the report content
        st.markdown(st.session_state.generated_report)

        st.divider()

        # Additional actions
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Regenerate Report"):
                st.session_state.generated_report = None
                st.session_state.report_timestamp = None
                st.rerun()

        with col2:
            if st.button("📊 View Cost Analysis"):
                st.info("💡 Switch to the 'Cost Analysis' tab to view detailed cost comparisons.")

        with col3:
            if st.button("📋 View Comparison"):
                st.info("💡 Switch to the 'Comparison' tab to view side-by-side EPC analysis.")

        # Report insights summary
        st.subheader("💡 Key Insights Summary")

        # Quick stats about the analysis
        epc_count = len(set([p.get('epc_contractor', {}).get('company_name', 'Unknown') for p in proposals]))
        cost_range = None
        if cost_proposals:
            costs = [p['costs']['total_project_cost'] for p in cost_proposals]
            cost_range = max(costs) - min(costs)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("EPC Contractors Analyzed", epc_count)

        with col2:
            st.metric("Proposals with Cost Data", len(cost_proposals))

        with col3:
            if cost_range:
                st.metric("Cost Spread", f"${cost_range:,.0f}")
            else:
                st.metric("Cost Spread", "N/A")

        with col4:
            tech_types = len(set([p.get('technology', {}).get('type', 'Unknown') for p in proposals]))
            st.metric("Technology Types", tech_types)

if __name__ == "__main__":
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        st.error("⚠️ OpenAI API key not found. Please set your OPENAI_API_KEY environment variable.")
        st.info("Create a .env file in the config folder with your API key.")
        st.stop()

    main()