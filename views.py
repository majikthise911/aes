import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import save_proposals
import os

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
    """Display scope comparison - assumptions, exclusions, clarifications between EPCs."""
    st.subheader("📋 Scope Comparison: Assumptions, Exclusions & Clarifications")

    if not proposals:
        st.warning("No proposals available for scope comparison")
        return

    st.write("Compare what each EPC contractor is including, excluding, and assuming in their proposals to understand which offers the most comprehensive scope.")

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

    # Create side-by-side comparison for each EPC
    epc_names = [p.get('epc_contractor', {}).get('company_name', f'EPC {i+1}') for i, p in enumerate(proposals)]

    # Display comparison selector
    st.write("### Select EPCs to Compare")
    col1, col2 = st.columns(2)

    with col1:
        epc1_idx = st.selectbox("First EPC", range(len(epc_names)), format_func=lambda x: epc_names[x], key="scope_epc1")

    with col2:
        epc2_options = [i for i in range(len(epc_names)) if i != epc1_idx]
        if epc2_options:
            epc2_idx = st.selectbox("Second EPC", epc2_options, format_func=lambda x: epc_names[x], key="scope_epc2")
        else:
            st.info("Need at least 2 proposals to compare")
            return

    st.divider()

    # Display side-by-side comparison
    proposal1 = proposals[epc1_idx]
    proposal2 = proposals[epc2_idx]

    scope1 = proposal1.get('scope', {})
    scope2 = proposal2.get('scope', {})

    # Inclusions comparison
    st.subheader("✅ Inclusions (What's Included in Scope)")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{epc_names[epc1_idx]}**")
        inclusions1 = scope1.get('inclusions', []) if scope1.get('inclusions') else []
        if inclusions1:
            for item in inclusions1:
                st.markdown(f"✓ {item}")
        else:
            st.info("No inclusions data extracted")

    with col2:
        st.markdown(f"**{epc_names[epc2_idx]}**")
        inclusions2 = scope2.get('inclusions', []) if scope2.get('inclusions') else []
        if inclusions2:
            for item in inclusions2:
                st.markdown(f"✓ {item}")
        else:
            st.info("No inclusions data extracted")

    st.divider()

    # Assumptions comparison
    st.subheader("📝 Assumptions")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{epc_names[epc1_idx]}**")
        assumptions1 = scope1.get('assumptions', []) if scope1.get('assumptions') else []
        if assumptions1:
            for item in assumptions1:
                st.markdown(f"• {item}")
        else:
            st.info("No assumptions data extracted")

    with col2:
        st.markdown(f"**{epc_names[epc2_idx]}**")
        assumptions2 = scope2.get('assumptions', []) if scope2.get('assumptions') else []
        if assumptions2:
            for item in assumptions2:
                st.markdown(f"• {item}")
        else:
            st.info("No assumptions data extracted")

    st.divider()

    # Exclusions comparison
    st.subheader("❌ Exclusions (Not Included)")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{epc_names[epc1_idx]}**")
        exclusions1 = scope1.get('exclusions', []) if scope1.get('exclusions') else []
        if exclusions1:
            for item in exclusions1:
                st.markdown(f"✗ {item}")
        else:
            st.info("No exclusions data extracted")

    with col2:
        st.markdown(f"**{epc_names[epc2_idx]}**")
        exclusions2 = scope2.get('exclusions', []) if scope2.get('exclusions') else []
        if exclusions2:
            for item in exclusions2:
                st.markdown(f"✗ {item}")
        else:
            st.info("No exclusions data extracted")

    st.divider()

    # Clarifications comparison
    st.subheader("💡 Clarifications & Special Conditions")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{epc_names[epc1_idx]}**")
        clarifications1 = scope1.get('clarifications', []) if scope1.get('clarifications') else []
        if clarifications1:
            for item in clarifications1:
                st.markdown(f"ℹ️ {item}")
        else:
            st.info("No clarifications data extracted")

    with col2:
        st.markdown(f"**{epc_names[epc2_idx]}**")
        clarifications2 = scope2.get('clarifications', []) if scope2.get('clarifications') else []
        if clarifications2:
            for item in clarifications2:
                st.markdown(f"ℹ️ {item}")
        else:
            st.info("No clarifications data extracted")

    st.divider()

    # Scope comprehensiveness analysis
    st.subheader("📊 Scope Comprehensiveness Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(f"{epc_names[epc1_idx]} - Inclusions", len(inclusions1))
        st.metric(f"{epc_names[epc1_idx]} - Exclusions", len(exclusions1))

    with col2:
        st.metric(f"{epc_names[epc2_idx]} - Inclusions", len(inclusions2))
        st.metric(f"{epc_names[epc2_idx]} - Exclusions", len(exclusions2))

    with col3:
        # Calculate comprehensiveness score (more inclusions = better, fewer exclusions = better)
        score1 = len(inclusions1) - (len(exclusions1) * 0.5)
        score2 = len(inclusions2) - (len(exclusions2) * 0.5)

        if score1 > score2:
            st.success(f"✅ {epc_names[epc1_idx]} appears more comprehensive")
        elif score2 > score1:
            st.success(f"✅ {epc_names[epc2_idx]} appears more comprehensive")
        else:
            st.info("Both proposals have similar scope comprehensiveness")

    # View all proposals summary
    st.divider()
    st.subheader("📋 All Proposals Scope Summary")

    summary_data = []
    for i, proposal in enumerate(proposals):
        epc_name = proposal.get('epc_contractor', {}).get('company_name', f'EPC {i+1}')
        scope = proposal.get('scope', {})

        inclusions = scope.get('inclusions', []) if scope.get('inclusions') else []
        exclusions = scope.get('exclusions', []) if scope.get('exclusions') else []
        assumptions = scope.get('assumptions', []) if scope.get('assumptions') else []
        clarifications = scope.get('clarifications', []) if scope.get('clarifications') else []

        summary_data.append({
            'EPC Contractor': epc_name,
            '# Inclusions': len(inclusions),
            '# Exclusions': len(exclusions),
            '# Assumptions': len(assumptions),
            '# Clarifications': len(clarifications),
            'Comprehensiveness Score': len(inclusions) - (len(exclusions) * 0.5)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Comprehensiveness Score', ascending=False)
    st.dataframe(summary_df, use_container_width=True)

    st.info("💡 **Comprehensiveness Score** = (# Inclusions) - (# Exclusions × 0.5). Higher is better, indicating more items included in scope.")

    if summary_data:
        best_epc = summary_df.iloc[0]['EPC Contractor']
        st.success(f"🏆 **Most Comprehensive Scope**: {best_epc}")