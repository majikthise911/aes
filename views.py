import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import save_proposals
import os
import json

def extract_scope_for_existing_proposals(proposals_missing_scope):
    """Re-extract scope data for proposals that don't have it."""
    from config import save_proposals

    progress_bar = st.progress(0)
    status_text = st.empty()

    total = len(proposals_missing_scope)
    success_count = 0

    # Check if PDFs are stored
    proposals_with_pdfs = [p for p in proposals_missing_scope if p.get('metadata', {}).get('pdf_path')]
    proposals_without_pdfs = [p for p in proposals_missing_scope if not p.get('metadata', {}).get('pdf_path')]

    if proposals_without_pdfs:
        st.warning(f"⚠️ {len(proposals_without_pdfs)} proposal(s) don't have stored PDFs (uploaded before PDF storage was added).")
        st.info("💡 These proposals need to be re-uploaded. New uploads will automatically save PDFs and extract scope data.")

    if not proposals_with_pdfs:
        progress_bar.empty()
        status_text.empty()
        return

    for idx, proposal in enumerate(proposals_with_pdfs):
        filename = proposal.get('metadata', {}).get('filename', 'Unknown')
        epc_name = proposal.get('epc_contractor', {}).get('company_name', 'Unknown')
        pdf_path = proposal.get('metadata', {}).get('pdf_path')

        status_text.text(f"Extracting scope data for {epc_name} ({idx + 1}/{len(proposals_with_pdfs)})...")

        try:
            if not os.path.exists(pdf_path):
                st.warning(f"⚠️ PDF file not found: {filename}")
                continue

            # Re-read the PDF
            with open(pdf_path, 'rb') as f:
                text = st.session_state.pdf_processor.extract_text_from_pdf(f)
                clean_text = st.session_state.pdf_processor.clean_text(text)

            # Extract scope details
            scope_data = st.session_state.gpt_extractor.extract_scope_details(clean_text)

            # Update the proposal in the data manager
            for p in st.session_state.data_manager.proposals:
                if p.get('metadata', {}).get('filename') == filename:
                    p['scope'] = scope_data
                    success_count += 1
                    break

        except Exception as e:
            st.error(f"Error extracting scope for {filename}: {str(e)}")

        progress_bar.progress((idx + 1) / len(proposals_with_pdfs))

    status_text.empty()
    progress_bar.empty()

    # Save updated proposals
    save_proposals()

    if success_count > 0:
        st.success(f"✅ Successfully extracted scope data for {success_count} proposal(s)!")
        st.rerun()

def show_overview(proposals):
    """Display overview dashboard."""
    try:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Proposals", len(proposals))

        with col2:
            # Get project capacity (should be same across all proposals for same project)
            capacities = [p.get('capacity', {}).get('ac_mw', 0) for p in proposals if p.get('capacity', {}).get('ac_mw')]
            if capacities:
                # Use the first valid capacity or average if they vary slightly
                unique_capacities = list(set(capacities))
                if len(unique_capacities) == 1:
                    project_capacity = unique_capacities[0]
                else:
                    # If capacities differ, show average and indicate variation
                    project_capacity = sum(capacities) / len(capacities)
                st.metric("Project AC Capacity", f"{project_capacity:.1f} MW")
            else:
                st.metric("Project AC Capacity", "N/A")

        with col3:
            tech_types = len(set([p.get('technology', {}).get('type', 'Unknown') for p in proposals]))
            st.metric("Technology Types", tech_types)

        with col4:
            cost_proposals = [p for p in proposals if p.get('costs', {}).get('cost_per_watt_dc')]
            if cost_proposals:
                avg_cost = sum([p['costs']['cost_per_watt_dc'] for p in cost_proposals]) / len(cost_proposals)
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
            # Format the dataframe for display
            df_display = df.copy()

            # Format Total Cost column
            if 'Total Cost ($)' in df_display.columns:
                df_display['Total Cost ($)'] = df_display['Total Cost ($)'].apply(
                    lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x
                )

            # Format Cost per Watt column
            if 'Cost per Watt DC ($/W)' in df_display.columns:
                df_display['Cost per Watt DC ($/W)'] = df_display['Cost per Watt DC ($/W)'].apply(
                    lambda x: f"${x:.3f}" if isinstance(x, (int, float)) else x
                )

            st.dataframe(df_display.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying overview: {str(e)}")
        st.info("Please check your proposal data or try uploading again.")

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

        for proposal in cost_proposals:
            costs.append(proposal['costs']['total_project_cost'])
            epc_name = proposal.get('epc_contractor', {}).get('company_name', 'Unknown EPC')
            epc_names.append(epc_name)

        # Sort by cost for better visualization
        sorted_data = sorted(zip(costs, epc_names))
        costs, epc_names = zip(*sorted_data)

        # Color gradient: green for lowest, red for highest
        colors = ['#2ca02c' if i == 0 else '#ff7f0e' if i == len(costs)-1 else '#1f77b4'
                  for i in range(len(costs))]

        fig = px.bar(x=epc_names, y=costs, title="Total Cost by EPC Contractor")
        fig.update_traces(marker_color=colors)
        fig.update_layout(
            xaxis_title="EPC Contractor",
            yaxis_title="Total Cost ($)",
            yaxis=dict(tickformat="$,.0f")  # Format y-axis with dollar sign and commas
        )
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
            fig.update_layout(
                xaxis_title="EPC Contractor",
                yaxis_title="$/W DC",
                yaxis=dict(tickformat="$.3f")  # Format y-axis with dollar sign and 3 decimal places
            )
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
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 EPC Rankings", "📊 Detailed Comparison", "📈 Cost Breakdown", "📋 Scope Comparison"])

    with tab1:
        show_epc_rankings(proposals)

    with tab2:
        show_detailed_comparison(proposals)

    with tab3:
        show_cost_breakdown_by_epc(proposals)

    with tab4:
        show_scope_comparison(proposals)

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
        yaxis=dict(tickformat="$,.0f"),  # Format y-axis with dollar sign and commas
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

    # Session state to store generated report
    if 'generated_report' not in st.session_state:
        st.session_state.generated_report = None
        st.session_state.report_timestamp = None

    with col2:
        if st.button("🚀 Generate AI Report", type="primary", use_container_width=True):
            # Generate report immediately when button is clicked
            with st.spinner("🤖 AI Consultant analyzing proposals... This may take 30-60 seconds."):
                try:
                    report = st.session_state.gpt_extractor.generate_epc_recommendation_report(proposals)
                    st.session_state.generated_report = report
                    st.session_state.report_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success("✅ Report generated successfully!")
                    st.rerun()  # Rerun to display the report
                except Exception as e:
                    st.error(f"❌ Error generating report: {str(e)}")
                    st.exception(e)  # Show full traceback for debugging

    # Display report if available
    if st.session_state.generated_report:
        st.divider()

        # Report header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"### 📄 EPC Recommendation Report")
            st.write(f"*Generated on {st.session_state.report_timestamp}*")

        with col2:
            # Export executive summary
            if st.button("📥 Export Executive Summary", use_container_width=True):
                # Create executive summary with key points only
                with st.spinner("Creating executive summary..."):
                    try:
                        exec_summary = st.session_state.gpt_extractor.generate_executive_summary(
                            st.session_state.generated_report
                        )

                        summary_content = f"""
EXECUTIVE SUMMARY
EPC CONTRACTOR RECOMMENDATION
Generated: {st.session_state.report_timestamp}

{'=' * 80}
{exec_summary}
{'=' * 80}

NOTE: This is an executive summary.
For the complete detailed analysis, please refer to the
AI Analysis > Recommendation Report tab in the dashboard.

Generated by AES EPC Proposal Dashboard
"""
                        st.download_button(
                            label="💾 Download Summary",
                            data=summary_content,
                            file_name=f"epc_exec_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"Error creating summary: {str(e)}")

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

def show_chatbot(proposals):
    """Display AI chatbot for asking questions about proposals."""
    st.subheader("💬 Ask AI About Your Proposals")

    if not proposals:
        st.warning("No proposals uploaded yet. Upload proposals to start asking questions!")
        return

    st.write(f"Ask questions about the {len(proposals)} uploaded proposal(s). The AI has access to all proposal data including costs, equipment, scope, assumptions, and exclusions.")

    # Initialize chatbot in session state
    if 'chatbot' not in st.session_state:
        from src.chatbot import ProposalChatbot
        st.session_state.chatbot = ProposalChatbot()

    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    # Example questions
    with st.expander("💡 Example Questions"):
        st.write("- What are the main cost differences between the EPCs?")
        st.write("- Which EPC has the most comprehensive scope?")
        st.write("- What assumptions does Mortenson make in their proposal?")
        st.write("- What items are excluded from Blattner's proposal?")
        st.write("- Compare the equipment specifications across all proposals")
        st.write("- What are the payment terms and warranties offered?")
        st.write("- Which proposal has the fastest schedule?")

    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your proposals..."):
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.chat(prompt, proposals)
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

    # Clear chat button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.chatbot.clear_history()
            st.rerun()

def show_scope_comparison(proposals):
    """Display AI-powered scope comprehensiveness analysis."""
    st.subheader("📋 AI-Powered Scope Comprehensiveness Analysis")

    if not proposals:
        st.warning("No proposals available for scope comparison")
        return

    # Check if scope data exists for proposals
    proposals_missing_scope = [p for p in proposals if not p.get('scope') or not any(p.get('scope', {}).get(k) for k in ['assumptions', 'exclusions', 'clarifications', 'inclusions'])]

    if proposals_missing_scope:
        st.warning(f"⚠️ {len(proposals_missing_scope)} proposal(s) are missing scope data. These were likely uploaded before scope extraction was added.")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Click 'Extract Scope Data' to analyze existing proposals and extract assumptions, exclusions, and clarifications. This may take 1-2 minutes per proposal.")
        with col2:
            if st.button("🔄 Extract Scope Data", type="primary", use_container_width=True):
                extract_scope_for_existing_proposals(proposals_missing_scope)

        st.divider()

    # AI Analysis Section
    st.write("### 🤖 Multi-Chain Reasoning Analysis")
    st.write("Our AI performs expert-level scope evaluation using a 3-chain reasoning process:")
    st.write("- **Chain 1**: Qualitative assessment of inclusions, exclusions, assumptions, and clarifications (not just counting)")
    st.write("- **Chain 2**: Significance evaluation of what's excluded (critical vs minor items) and risk analysis")
    st.write("- **Chain 3**: Self-critique and validation to ensure logical, defensible conclusions")

    # Session state for AI analysis
    if 'scope_ai_analysis' not in st.session_state:
        st.session_state.scope_ai_analysis = None

    col1, col2 = st.columns([3, 1])

    with col1:
        st.info("💡 This analysis uses GPT-4o with multi-chain reasoning. Analysis takes ~30-60 seconds and evaluates all proposals comprehensively.")

    with col2:
        if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
            with st.spinner("🤖 AI analyzing scope comprehensiveness across all proposals... This may take up to 1 minute."):
                try:
                    # Run multi-chain analysis on all proposals
                    analysis_result = st.session_state.gpt_extractor.analyze_scope_comprehensiveness(proposals)
                    st.session_state.scope_ai_analysis = analysis_result
                    st.success("✅ Analysis complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error running AI analysis: {str(e)}")

    # Display AI analysis results if available
    if st.session_state.scope_ai_analysis:
        analysis = st.session_state.scope_ai_analysis

        if analysis.get('error'):
            st.error(analysis['error'])
        else:
            # Display results in expandable sections
            st.write("---")

            # Chain 1: Initial Analysis
            with st.expander("🔍 Chain 1: Initial Qualitative Analysis", expanded=True):
                st.markdown(analysis.get('initial_analysis', 'No analysis available'))

            # Chain 2: Significance Evaluation
            with st.expander("⚖️ Chain 2: Significance & Risk Evaluation", expanded=True):
                st.markdown(analysis.get('significance_evaluation', 'No evaluation available'))

            # Chain 3: Final Assessment (most important)
            with st.expander("✅ Chain 3: Final Validated Assessment & Recommendation", expanded=True):
                st.markdown(analysis.get('final_assessment', 'No assessment available'))

            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Re-run Analysis"):
                    st.session_state.scope_ai_analysis = None
                    st.rerun()

            with col2:
                if st.button("📥 Export Executive Summary"):
                    timestamp = analysis.get('timestamp', 'unknown_time')

                    # Create executive summary (extract key points from final assessment)
                    final_assessment = analysis.get('final_assessment', 'N/A')

                    exec_summary = f"""
EXECUTIVE SUMMARY
SCOPE COMPREHENSIVENESS ANALYSIS
Generated: {timestamp}

{'=' * 80}
FINAL RECOMMENDATION
{'=' * 80}
{final_assessment}

{'=' * 80}
NOTE: This is an executive summary.
For detailed analysis including all 3 chains of reasoning,
please refer to the AI Analysis tab in the dashboard.
{'=' * 80}

Generated by AES EPC Proposal Dashboard
"""
                    st.download_button(
                        label="💾 Download Summary",
                        data=exec_summary,
                        file_name=f"scope_summary_{timestamp.replace(':', '-').replace(' ', '_')}.txt" if timestamp != 'unknown_time' else "scope_summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

def show_database_query(proposals):
    """Display database query interface with AI and visual filters."""
    st.subheader("🗄️ Database Query & Export")

    if not proposals:
        st.warning("No proposals in database. Upload proposals to get started.")
        return

    st.write(f"**Database contains {len(proposals)} proposal(s)**")

    # Create tabs for different query methods
    query_tab1, query_tab2, query_tab3 = st.tabs(["🤖 AI Query", "🔧 Visual Filters", "📊 View All"])

    # Tab 1: AI Natural Language Query
    with query_tab1:
        st.write("### Ask Questions About Your Proposals")
        st.write("Ask in plain English and AI will query the database for you.")

        # Example queries
        with st.expander("💡 Example Queries"):
            st.write("- Show me all proposals under $500M")
            st.write("- Which EPCs have the lowest cost per watt?")
            st.write("- Find proposals with more than 10 exclusions")
            st.write("- Show me Mortenson and Blattner proposals only")
            st.write("- What proposals have batteries included?")
            st.write("- List proposals by cost from lowest to highest")

        # AI Query input
        ai_query = st.text_input(
            "Your question:",
            placeholder="e.g., Show me all proposals under $500M with solar + storage",
            key="db_ai_query"
        )

        if st.button("🔍 Search", type="primary", use_container_width=True):
            if ai_query:
                with st.spinner("🤖 AI processing your query..."):
                    try:
                        # Call AI to interpret query and filter data
                        filtered_proposals = st.session_state.gpt_extractor.query_proposals_with_ai(
                            ai_query, proposals
                        )

                        if filtered_proposals:
                            st.success(f"✅ Found {len(filtered_proposals)} matching proposal(s)")

                            # Display results
                            display_query_results(filtered_proposals)
                        else:
                            st.warning("No proposals match your query. Try adjusting your criteria.")

                    except Exception as e:
                        st.error(f"❌ Error processing query: {str(e)}")
            else:
                st.warning("Please enter a query")

    # Tab 2: Visual Filters
    with query_tab2:
        st.write("### Filter Proposals Visually")

        # Cost filters
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Cost Filters**")

            # Get min/max costs for range slider
            costs = [p.get('costs', {}).get('total_project_cost') for p in proposals if p.get('costs', {}).get('total_project_cost')]

            if costs:
                min_cost = min(costs)
                max_cost = max(costs)

                cost_range = st.slider(
                    "Total Project Cost ($M)",
                    min_value=float(min_cost) / 1_000_000,
                    max_value=float(max_cost) / 1_000_000,
                    value=(float(min_cost) / 1_000_000, float(max_cost) / 1_000_000),
                    step=1.0,
                    key="db_cost_filter"
                )

            # Cost per watt filter
            cpw_values = [p.get('costs', {}).get('cost_per_watt_dc') for p in proposals if p.get('costs', {}).get('cost_per_watt_dc')]
            if cpw_values:
                min_cpw = min(cpw_values)
                max_cpw = max(cpw_values)

                cpw_range = st.slider(
                    "Cost per Watt DC ($/W)",
                    min_value=float(min_cpw),
                    max_value=float(max_cpw),
                    value=(float(min_cpw), float(max_cpw)),
                    step=0.01,
                    key="db_cpw_filter"
                )

        with col2:
            st.write("**EPC & Technology Filters**")

            # EPC filter
            epc_names = list(set([p.get('epc_contractor', {}).get('company_name', 'Unknown') for p in proposals]))
            selected_epcs = st.multiselect(
                "EPC Contractors",
                options=epc_names,
                default=epc_names,
                key="db_epc_filter"
            )

            # Technology filter
            tech_types = list(set([p.get('technology', {}).get('type', 'Unknown') for p in proposals]))
            selected_tech = st.multiselect(
                "Technology Types",
                options=tech_types,
                default=tech_types,
                key="db_tech_filter"
            )

        # Scope filters
        st.write("**Scope Filters**")
        col1, col2 = st.columns(2)

        with col1:
            max_exclusions = st.number_input(
                "Max Exclusions",
                min_value=0,
                max_value=100,
                value=100,
                key="db_max_exclusions_filter"
            )

        with col2:
            min_inclusions = st.number_input(
                "Min Inclusions",
                min_value=0,
                max_value=100,
                value=0,
                key="db_min_inclusions_filter"
            )

        # Apply filters button
        if st.button("🔧 Apply Filters", type="primary", use_container_width=True):
            filtered = proposals.copy()

            # Apply cost filter
            if costs:
                filtered = [p for p in filtered if p.get('costs', {}).get('total_project_cost') and
                           cost_range[0] * 1_000_000 <= p['costs']['total_project_cost'] <= cost_range[1] * 1_000_000]

            # Apply CPW filter
            if cpw_values:
                filtered = [p for p in filtered if p.get('costs', {}).get('cost_per_watt_dc') and
                           cpw_range[0] <= p['costs']['cost_per_watt_dc'] <= cpw_range[1]]

            # Apply EPC filter
            filtered = [p for p in filtered if p.get('epc_contractor', {}).get('company_name') in selected_epcs]

            # Apply technology filter
            filtered = [p for p in filtered if p.get('technology', {}).get('type') in selected_tech]

            # Apply scope filters
            filtered = [p for p in filtered if
                       len(p.get('scope', {}).get('exclusions', [])) <= max_exclusions and
                       len(p.get('scope', {}).get('inclusions', [])) >= min_inclusions]

            st.success(f"✅ Found {len(filtered)} matching proposal(s)")
            display_query_results(filtered)

    # Tab 3: View All
    with query_tab3:
        st.write("### All Proposals in Database")
        display_query_results(proposals, show_export=True)

def display_query_results(proposals, show_export=True):
    """Display query results in a table with export options."""

    if not proposals:
        st.info("No proposals to display")
        return

    # Create DataFrame for display
    results_data = []
    for p in proposals:
        epc = p.get('epc_contractor', {})
        costs = p.get('costs', {})
        capacity = p.get('capacity', {})
        tech = p.get('technology', {})
        scope = p.get('scope', {})
        project = p.get('project_info', {})

        results_data.append({
            'EPC Contractor': epc.get('company_name', 'Unknown'),
            'Project': project.get('project_name', 'N/A'),
            'Technology': tech.get('type', 'N/A'),
            'Total Cost ($)': costs.get('total_project_cost', 0),
            'Cost/Watt DC ($/W)': costs.get('cost_per_watt_dc', 0),
            'AC Capacity (MW)': capacity.get('ac_mw', 0),
            'DC Capacity (MW)': capacity.get('dc_mw', 0),
            'Inclusions': len(scope.get('inclusions', [])),
            'Exclusions': len(scope.get('exclusions', [])),
            'Proposal Date': epc.get('proposal_date', 'N/A')
        })

    df = pd.DataFrame(results_data)

    # Format currency columns
    if 'Total Cost ($)' in df.columns:
        df['Total Cost ($)'] = df['Total Cost ($)'].apply(lambda x: f"${x:,.0f}" if x > 0 else 'N/A')

    if 'Cost/Watt DC ($/W)' in df.columns:
        df['Cost/Watt DC ($/W)'] = df['Cost/Watt DC ($/W)'].apply(lambda x: f"${x:.3f}" if x > 0 else 'N/A')

    # Display table
    st.dataframe(df, use_container_width=True, hide_index=True)

    if show_export:
        # Export options
        st.divider()
        st.write("### Export Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            # CSV Export
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"proposals_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            # JSON Export
            json_str = json.dumps(proposals, indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_str,
                file_name=f"proposals_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

        with col3:
            # Excel export note
            st.button(
                "📥 Download as Excel",
                help="Excel export coming soon!",
                disabled=True,
                use_container_width=True
            )

# ============================================================================
# CONSOLIDATED VIEWS FOR 4-TAB STRUCTURE
# ============================================================================

def show_dashboard(proposals):
    """Consolidated dashboard: Overview metrics + Map + Proposals table."""
    st.subheader("📊 Dashboard Overview")

    if not proposals:
        st.info("👆 Upload EPC proposals using the sidebar to get started")
        return

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Proposals", len(proposals))

    with col2:
        capacities = [p.get('capacity', {}).get('ac_mw', 0) for p in proposals if p.get('capacity', {}).get('ac_mw')]
        if capacities:
            unique_capacities = list(set(capacities))
            if len(unique_capacities) == 1:
                project_capacity = unique_capacities[0]
            else:
                project_capacity = sum(capacities) / len(capacities)
            st.metric("Project AC Capacity", f"{project_capacity:.1f} MW")
        else:
            st.metric("Project AC Capacity", "N/A")

    with col3:
        tech_types = len(set([p.get('technology', {}).get('type', 'Unknown') for p in proposals]))
        st.metric("Technology Types", tech_types)

    with col4:
        cost_proposals = [p for p in proposals if p.get('costs', {}).get('cost_per_watt_dc')]
        if cost_proposals:
            avg_cost = sum([p['costs']['cost_per_watt_dc'] for p in cost_proposals]) / len(cost_proposals)
            st.metric("Avg Cost/W DC", f"${avg_cost:.2f}")
        else:
            st.metric("Avg Cost/W DC", "N/A")

    st.divider()

    # Map and Technology Breakdown
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🗺️ Project Locations")
        # Map visualization
        m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
        markers_added = 0

        for i, proposal in enumerate(proposals):
            location = proposal.get('project_info', {}).get('location', {})
            coords = location.get('coordinates', {})
            project_name = proposal.get('project_info', {}).get('project_name', f'Unknown Project {i+1}')

            if coords and coords.get('lat') and coords.get('lon'):
                popup_text = f"""
                <b>{project_name}</b><br>
                Technology: {proposal.get('technology', {}).get('type', 'Unknown')}<br>
                Capacity: {proposal.get('capacity', {}).get('ac_mw', 'N/A')} MW AC
                """
                folium.Marker(
                    [coords['lat'], coords['lon']],
                    popup=popup_text,
                    tooltip=project_name,
                    icon=folium.Icon(color='green', icon='bolt')
                ).add_to(m)
                markers_added += 1

        st_folium(m, width=700, height=400)
        st.caption(f"📍 {markers_added} of {len(proposals)} proposals mapped")

    with col2:
        st.subheader("⚙️ Technology Mix")
        # Calculate capacity by technology (use average across proposals for same project)
        # Group proposals by project to avoid double counting
        project_capacities = {}

        for proposal in proposals:
            project_name = proposal.get('project_info', {}).get('project_name', 'Unknown')

            # Get PV DC capacity
            pv_dc_capacity = proposal.get('capacity', {}).get('dc_mw', 0)

            # Check if has BESS
            bess_power = 0
            batteries = proposal.get('equipment', {}).get('batteries')
            if batteries and isinstance(batteries, dict):
                bess_power = batteries.get('power_mw', 0)

            # Store capacities by project (will take average if multiple proposals)
            if project_name not in project_capacities:
                project_capacities[project_name] = {'pv': [], 'bess': []}

            project_capacities[project_name]['pv'].append(pv_dc_capacity)
            if bess_power > 0:
                project_capacities[project_name]['bess'].append(bess_power)

        # Calculate technology totals (average per project)
        tech_capacity = {}
        for project, caps in project_capacities.items():
            # Average PV capacity for this project
            avg_pv = sum(caps['pv']) / len(caps['pv']) if caps['pv'] else 0
            tech_capacity['PV (DC)'] = tech_capacity.get('PV (DC)', 0) + avg_pv

            # Average BESS capacity for this project
            if caps['bess']:
                avg_bess = sum(caps['bess']) / len(caps['bess'])
                tech_capacity['BESS'] = tech_capacity.get('BESS', 0) + avg_bess

        if tech_capacity:
            fig = px.pie(
                values=list(tech_capacity.values()),
                names=list(tech_capacity.keys()),
                title="Project Capacity by Technology (MW)"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Proposals Table
    st.subheader("📋 All Proposals")
    df = st.session_state.data_manager.create_comparison_dataframe()
    if not df.empty:
        df_display = df.copy()
        if 'Total Cost ($)' in df_display.columns:
            df_display['Total Cost ($)'] = df_display['Total Cost ($)'].apply(
                lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x
            )
        if 'Cost per Watt DC ($/W)' in df_display.columns:
            df_display['Cost per Watt DC ($/W)'] = df_display['Cost per Watt DC ($/W)'].apply(
                lambda x: f"${x:.3f}" if isinstance(x, (int, float)) else x
            )
        st.dataframe(df_display, use_container_width=True)

def show_cost_comparison(proposals):
    """Consolidated cost analysis and EPC comparison."""
    st.subheader("💰 Cost Analysis & EPC Comparison")

    if not proposals:
        st.warning("No proposals available")
        return

    # Filter proposals with cost data
    cost_proposals = [p for p in proposals if p.get('costs', {}).get('total_project_cost')]

    if not cost_proposals:
        st.warning("No cost data available in uploaded proposals")
        return

    # Cost comparison charts
    col1, col2 = st.columns(2)

    with col1:
        # Total cost comparison
        costs = []
        epc_names = []
        for proposal in cost_proposals:
            costs.append(proposal['costs']['total_project_cost'])
            epc_name = proposal.get('epc_contractor', {}).get('company_name', 'Unknown EPC')
            epc_names.append(epc_name)

        sorted_data = sorted(zip(costs, epc_names))
        costs, epc_names = zip(*sorted_data)
        colors = ['#2ca02c' if i == 0 else '#ff7f0e' if i == len(costs)-1 else '#1f77b4' for i in range(len(costs))]

        fig = px.bar(x=epc_names, y=costs, title="Total Cost by EPC Contractor")
        fig.update_traces(marker_color=colors)
        fig.update_layout(xaxis_title="EPC Contractor", yaxis_title="Total Cost ($)", yaxis=dict(tickformat="$,.0f"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Cost per watt comparison
        cpw_data = []
        cpw_epc_names = []
        for proposal in cost_proposals:
            cpw = proposal.get('costs', {}).get('cost_per_watt_dc')
            if cpw:
                cpw_data.append(cpw)
                epc_name = proposal.get('epc_contractor', {}).get('company_name', 'Unknown EPC')
                cpw_epc_names.append(epc_name)

        if cpw_data:
            sorted_cpw_data = sorted(zip(cpw_data, cpw_epc_names))
            cpw_data, cpw_epc_names = zip(*sorted_cpw_data)
            colors = ['#2ca02c' if i == 0 else '#ff7f0e' if i == len(cpw_data)-1 else '#1f77b4' for i in range(len(cpw_data))]

            fig = px.bar(x=cpw_epc_names, y=cpw_data, title="Cost per Watt DC by EPC")
            fig.update_traces(marker_color=colors)
            fig.update_layout(xaxis_title="EPC Contractor", yaxis_title="$/W DC", yaxis=dict(tickformat="$.3f"))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Key metrics
    st.subheader("📊 Key Metrics")
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

    st.divider()

    # Detailed comparison table
    st.subheader("📋 Detailed EPC Comparison")
    df = st.session_state.data_manager.create_epc_ranking_dataframe()
    if not df.empty:
        st.dataframe(
            df.style.format({
                'Total Cost ($)': lambda x: f"${x:,.0f}" if pd.notnull(x) and x != 'N/A' else x,
                'Cost/Watt DC ($/W)': lambda x: f"${x:.3f}" if pd.notnull(x) and x != 'N/A' else x,
                'Equipment Cost ($)': lambda x: f"${x:,.0f}" if pd.notnull(x) and x != 'N/A' else x,
                'Labor Cost ($)': lambda x: f"${x:,.0f}" if pd.notnull(x) and x != 'N/A' else x
            }),
            use_container_width=True
        )

def show_ai_analysis(proposals):
    """Consolidated AI features: Scope analysis, recommendation report, chatbot."""
    st.subheader("🤖 AI-Powered Analysis")

    if not proposals:
        st.warning("No proposals available for analysis")
        return

    # Sub-tabs for different AI features
    ai_tab1, ai_tab2, ai_tab3 = st.tabs(["🔍 Scope Analysis", "📝 Recommendation Report", "💬 Ask Questions"])

    # Tab 1: Scope Comprehensiveness Analysis
    with ai_tab1:
        show_scope_comparison(proposals)

    # Tab 2: AI Recommendation Report
    with ai_tab2:
        show_ai_report(proposals)

    # Tab 3: Chatbot Q&A
    with ai_tab3:
        show_chatbot(proposals)

def show_database(proposals):
    """Queryable database interface with export capabilities."""
    st.subheader("🗄️ Queryable Database")

    if not proposals:
        st.warning("No proposals in database. Upload proposals to get started.")
        return

    st.write(f"**Database contains {len(proposals)} proposal(s)**")
    st.write("Query your proposal database using natural language or view all data with export options.")

    # Database tabs
    db_tab1, db_tab2 = st.tabs(["🔍 Query Database", "📊 View & Export All"])

    # Tab 1: AI Query
    with db_tab1:
        st.write("### Natural Language Database Query")
        st.write("Ask questions in plain English to query and filter your proposal database.")

        with st.expander("💡 Example Queries"):
            st.write("- Show me all proposals under $500M")
            st.write("- Which EPCs have the lowest cost per watt?")
            st.write("- Find proposals with more than 10 exclusions")
            st.write("- Show me Mortenson and Blattner proposals only")
            st.write("- What proposals include battery storage?")
            st.write("- Compare proposals from the last 6 months")

        ai_query = st.text_input(
            "Query:",
            placeholder="e.g., Show me all proposals under $500M with solar + storage",
            key="db_query_input"
        )

        if st.button("🔍 Query Database", type="primary", use_container_width=True):
            if ai_query:
                with st.spinner("🤖 Querying database..."):
                    try:
                        filtered_proposals = st.session_state.gpt_extractor.query_proposals_with_ai(ai_query, proposals)
                        if filtered_proposals:
                            st.success(f"✅ Query returned {len(filtered_proposals)} result(s)")
                            display_query_results(filtered_proposals, show_export=True)
                        else:
                            st.info("No results found. Try adjusting your query.")
                    except Exception as e:
                        st.error(f"❌ Query error: {str(e)}")
            else:
                st.warning("Please enter a query")

    # Tab 2: View All & Export
    with db_tab2:
        st.write("### Database Contents")
        st.write("View all proposals in the database. Use sidebar filters to narrow results before exporting.")

        display_query_results(proposals, show_export=True)