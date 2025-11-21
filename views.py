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
        df = st.session_state.data_manager.create_comparison_dataframe(proposals)
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
    df = st.session_state.data_manager.create_epc_ranking_dataframe(proposals)

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

    df = st.session_state.data_manager.create_comparison_dataframe(proposals)

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

    # Check if multiple projects are selected
    unique_projects = set([p.get('project_info', {}).get('project_name', 'Unknown') for p in proposals])
    if len(unique_projects) > 1:
        st.error("❌ Multiple projects detected")
        st.warning(f"You have {len(unique_projects)} different projects selected: {', '.join(unique_projects)}")
        st.info("👈 Please use the sidebar filters to select a **single project** before generating the recommendation report. EPC selection should compare contractors bidding on the same project.")
        # Clear any stale flags
        if 'proceed_without_scope' in st.session_state:
            del st.session_state.proceed_without_scope
        if 'run_scope_then_report' in st.session_state:
            del st.session_state.run_scope_then_report
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
        st.write("**Uses 3-chain reasoning with scope analysis integration:**")
        st.write("- **Chain 1: Initial Analysis** - Cost, technical specs, and scope evaluation")
        st.write("- **Chain 2: Deep Evaluation** - Trade-offs, risks, and value engineering")
        st.write("- **Chain 3: Final Recommendation** - Self-critique and executive summary")
        st.write("")
        st.write("The report incorporates comprehensive scope analysis and provides actionable EPC selection guidance.")

    # Session state to store generated report
    if 'generated_report' not in st.session_state:
        st.session_state.generated_report = None
        st.session_state.report_timestamp = None

    # Handle run scope then report flag
    if st.session_state.get('run_scope_then_report', False):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Run scope analysis
            status_text.text("📋 Running scope analysis first...")
            progress_bar.progress(10)

            analysis_result = st.session_state.gpt_extractor.analyze_scope_comprehensiveness(proposals)
            st.session_state.scope_ai_analysis = analysis_result
            progress_bar.progress(50)

            # Step 2: Generate recommendation report
            status_text.text("🤖 Now generating recommendation report...")

            # Prepare data summary
            summary = st.session_state.gpt_extractor._prepare_proposals_summary(proposals)

            # Get scope summary
            scope_summary = f"""

SCOPE ANALYSIS RESULTS:
{analysis_result.get('final_assessment', 'No scope analysis available')}
"""
            progress_bar.progress(60)

            # Run 3-chain recommendation
            status_text.text("⚙️ Chain 1: Running initial analysis...")
            chain1_result = st.session_state.gpt_extractor._chain1_initial_recommendation(summary, scope_summary)
            progress_bar.progress(70)

            status_text.text("⚙️ Chain 2: Conducting deep evaluation...")
            chain2_result = st.session_state.gpt_extractor._chain2_deep_evaluation(summary, scope_summary, chain1_result)
            progress_bar.progress(85)

            status_text.text("⚙️ Chain 3: Generating final recommendation...")
            chain3_result = st.session_state.gpt_extractor._chain3_final_recommendation(summary, scope_summary, chain1_result, chain2_result)
            progress_bar.progress(100)

            # Store results - save all chains for detailed report
            st.session_state.generated_report = chain3_result
            st.session_state.report_chain1 = chain1_result
            st.session_state.report_chain2 = chain2_result
            st.session_state.report_chain3 = chain3_result
            st.session_state.report_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

            # Clear flag
            del st.session_state.run_scope_then_report

            # Clean up
            status_text.empty()
            progress_bar.empty()

            st.success("✅ Scope analysis & recommendation report generated successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            status_text.empty()
            progress_bar.empty()
            if 'run_scope_then_report' in st.session_state:
                del st.session_state.run_scope_then_report

    # Check if scope analysis has been run for these specific proposals
    # Need to verify it's for the CURRENT set of proposals, not cached from different project
    scope_analysis_done = False
    if 'scope_ai_analysis' in st.session_state and st.session_state.scope_ai_analysis:
        analysis = st.session_state.scope_ai_analysis
        # Verify the analysis has valid data and matches current proposals
        if analysis.get('final_assessment') and not analysis.get('error'):
            # Get the proposal count from analysis to verify it matches current proposals
            analysis_proposal_count = analysis.get('proposal_count', 0)
            if analysis_proposal_count == len(proposals):
                scope_analysis_done = True
            else:
                # Analysis is for different set of proposals, clear it
                st.session_state.scope_ai_analysis = None
                # Also clear the proceed flag since analysis doesn't match
                if 'proceed_without_scope' in st.session_state:
                    del st.session_state.proceed_without_scope

    with col2:
        # Debug info (can remove later)
        st.caption(f"📊 {len(proposals)} proposals | Project: {list(unique_projects)[0] if len(unique_projects) == 1 else 'Multiple'}")

        # Show warning if scope analysis not done
        if not scope_analysis_done:
            st.error("❌ Scope analysis required")
            st.write("The recommendation report uses scope analysis for better insights.")
            st.divider()

            if st.button("🚀 Run Scope Analysis & Generate Report", type="primary", use_container_width=True):
                st.session_state.run_scope_then_report = True
                st.rerun()

            st.divider()

            if st.button("⚡ Skip Scope Analysis", type="secondary", use_container_width=True):
                st.session_state.proceed_without_scope = True
                st.rerun()
        else:
            st.success("✅ Scope analysis complete")

        # Show generate button if scope analysis is done OR user chose to proceed anyway
        if scope_analysis_done or st.session_state.get('proceed_without_scope', False):
            if scope_analysis_done:
                st.info("✅ Using existing scope analysis results")

            if st.button("🚀 Generate AI Report", type="primary", use_container_width=True):
                import time as time_module
                # Generate report immediately when button is clicked
                try:
                    # Reset usage stats and start timer
                    st.session_state.gpt_extractor.reset_usage_stats()
                    start_time = time_module.time()

                    # Get scope analysis if available
                    scope_analysis = None
                    if scope_analysis_done:
                        scope_analysis = st.session_state.scope_ai_analysis

                    # Create progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Prepare data summary
                    status_text.text("🤖 Preparing data for analysis...")
                    progress_bar.progress(10)

                    # Use parallel report generation
                    status_text.text("⚙️ Running parallel recommendation analysis...")
                    progress_bar.progress(20)

                    chain3_result = st.session_state.gpt_extractor.generate_epc_recommendation_report(
                        proposals, scope_analysis, parallel=True
                    )
                    progress_bar.progress(100)

                    # Get timing and usage stats
                    elapsed_time = time_module.time() - start_time
                    usage_stats = st.session_state.gpt_extractor.get_usage_stats()

                    # Store results
                    st.session_state.generated_report = chain3_result
                    st.session_state.report_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.report_usage = {
                        'elapsed_time': elapsed_time,
                        'input_tokens': usage_stats.input_tokens,
                        'output_tokens': usage_stats.output_tokens,
                        'total_tokens': usage_stats.total_tokens,
                        'provider': st.session_state.gpt_extractor.provider_name,
                        'used_scope_analysis': scope_analysis_done
                    }

                    # Clear the proceed flag
                    if 'proceed_without_scope' in st.session_state:
                        del st.session_state.proceed_without_scope

                    # Clean up progress indicators
                    status_text.empty()
                    progress_bar.empty()

                    st.success(f"✅ Report generated in {elapsed_time:.1f}s | {usage_stats.total_tokens:,} tokens")
                    st.rerun()  # Rerun to display the report
                except Exception as e:
                    st.error(f"❌ Error generating report: {str(e)}")
                    st.exception(e)  # Show full traceback for debugging

    # Display report if available
    if st.session_state.generated_report:
        st.divider()

        # Report header and export buttons
        st.write(f"### 📄 EPC Recommendation Report")
        st.write(f"*Generated on {st.session_state.report_timestamp}*")

        # Display usage stats if available
        if 'report_usage' in st.session_state and st.session_state.report_usage:
            usage = st.session_state.report_usage
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Duration", f"{usage.get('elapsed_time', 0):.1f}s")
            with col_s2:
                st.metric("Total Tokens", f"{usage.get('total_tokens', 0):,}")
            with col_s3:
                st.metric("Input Tokens", f"{usage.get('input_tokens', 0):,}")
            with col_s4:
                st.metric("Output Tokens", f"{usage.get('output_tokens', 0):,}")
            scope_note = "with scope analysis" if usage.get('used_scope_analysis') else "without scope analysis"
            st.caption(f"Provider: {usage.get('provider', 'N/A')} | Generated {scope_note}")

        # Export buttons
        col1, col2 = st.columns(2)
        with col1:
            # Export executive summary
            if st.button("📥 Executive Summary", use_container_width=True, key="recommendation_exec_summary"):
                # Create concise executive summary
                with st.spinner("Creating executive summary..."):
                    try:
                        # Extract only key sections: Recommendation + Cost + Next Steps
                        prompt = f"""Extract only the following sections from this EPC recommendation report:

1. Top Recommended EPC (name and 2-3 sentence justification)
2. Estimated Cost (total project cost comparison)
3. Key Risk (top 1-2 risks to mitigate)
4. Next Steps (top 2-3 action items)

Keep it extremely concise - maximum 250 words total.

FULL REPORT:
{st.session_state.generated_report}

Return only these 4 sections formatted clearly."""

                        response = st.session_state.gpt_extractor.client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You extract key points for executive summaries. Be extremely concise and direct."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.1,
                            max_tokens=400
                        )

                        exec_summary_content = response.choices[0].message.content

                        summary_content = f"""
EXECUTIVE SUMMARY
EPC CONTRACTOR RECOMMENDATION
Generated: {st.session_state.report_timestamp}

{'=' * 80}
{exec_summary_content}
{'=' * 80}

Generated by AES EPC Proposal Dashboard
"""
                        st.download_button(
                            label="💾 Download Executive Summary",
                            data=summary_content,
                            file_name=f"epc_exec_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error creating summary: {str(e)}")

        with col2:
            # Export detailed report with all chains
            if st.button("📄 Detailed Report", use_container_width=True, key="recommendation_detailed_report"):
                timestamp = st.session_state.report_timestamp

                # Create full detailed report with all 3 chains
                detailed_report = f"""
DETAILED EPC RECOMMENDATION REPORT
Multi-Chain Reasoning Analysis
Generated: {timestamp}

{'=' * 80}
CHAIN 1: INITIAL COMPREHENSIVE ANALYSIS
{'=' * 80}

{st.session_state.get('report_chain1', 'Chain 1 data not available')}

{'=' * 80}
CHAIN 2: DEEP EVALUATION & TRADE-OFF ANALYSIS
{'=' * 80}

{st.session_state.get('report_chain2', 'Chain 2 data not available')}

{'=' * 80}
CHAIN 3: FINAL RECOMMENDATION & SYNTHESIS
{'=' * 80}

{st.session_state.get('report_chain3', st.session_state.generated_report)}

{'=' * 80}
END OF REPORT
{'=' * 80}

Generated by AES EPC Proposal Dashboard
Multi-Chain AI Analysis System
"""
                st.download_button(
                    label="💾 Download Detailed Report",
                    data=detailed_report,
                    file_name=f"epc_detailed_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
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

    # Sync chatbot provider with extractor
    if hasattr(st.session_state, 'gpt_extractor') and hasattr(st.session_state.chatbot, 'set_provider'):
        if st.session_state.chatbot.provider_name != st.session_state.gpt_extractor.provider_name:
            st.session_state.chatbot.set_provider(st.session_state.gpt_extractor.provider_name)

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

    # Check if multiple projects are selected
    unique_projects = set([p.get('project_info', {}).get('project_name', 'Unknown') for p in proposals])
    if len(unique_projects) > 1:
        st.error("❌ Multiple projects detected")
        st.warning(f"You have {len(unique_projects)} different projects selected: {', '.join(unique_projects)}")
        st.info("👈 Please use the sidebar filters to select a **single project** before running scope analysis. Scope comparison should analyze EPCs bidding on the same project.")
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
            import time as time_module
            try:
                # Reset usage stats before analysis
                st.session_state.gpt_extractor.reset_usage_stats()
                start_time = time_module.time()

                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                timer_display = st.empty()

                # Prepare scope data
                status_text.text("📋 Preparing scope data for analysis...")
                progress_bar.progress(10)

                # Use the parallel analyze_scope_comprehensiveness method
                status_text.text("⚙️ Running parallel multi-chain analysis...")
                timer_display.text(f"⏱️ Elapsed: 0s")
                progress_bar.progress(20)

                analysis_result = st.session_state.gpt_extractor.analyze_scope_comprehensiveness(proposals, parallel=True)
                progress_bar.progress(100)

                # Get final timing and usage
                elapsed_time = time_module.time() - start_time
                usage_stats = st.session_state.gpt_extractor.get_usage_stats()

                # Add timing and usage to results
                analysis_result['elapsed_time'] = elapsed_time
                analysis_result['usage_stats'] = {
                    'input_tokens': usage_stats.input_tokens,
                    'output_tokens': usage_stats.output_tokens,
                    'total_tokens': usage_stats.total_tokens,
                    'provider': st.session_state.gpt_extractor.provider_name
                }

                st.session_state.scope_ai_analysis = analysis_result

                # Clean up progress indicators
                status_text.empty()
                progress_bar.empty()
                timer_display.empty()

                st.success(f"✅ Analysis complete in {elapsed_time:.1f}s | {usage_stats.total_tokens:,} tokens used")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error running AI analysis: {str(e)}")

    # Display AI analysis results if available
    if st.session_state.scope_ai_analysis:
        analysis = st.session_state.scope_ai_analysis

        if analysis.get('error'):
            st.error(analysis['error'])
        else:
            # Display usage stats
            st.write("---")
            usage = analysis.get('usage_stats', {})
            elapsed = analysis.get('elapsed_time', 0)
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Duration", f"{elapsed:.1f}s")
            with col_stat2:
                st.metric("Total Tokens", f"{usage.get('total_tokens', 0):,}")
            with col_stat3:
                st.metric("Input Tokens", f"{usage.get('input_tokens', 0):,}")
            with col_stat4:
                st.metric("Output Tokens", f"{usage.get('output_tokens', 0):,}")

            st.caption(f"Provider: {usage.get('provider', 'N/A')} | Generated: {analysis.get('timestamp', 'N/A')}")

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
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Re-run Analysis", use_container_width=True):
                    st.session_state.scope_ai_analysis = None
                    st.rerun()

            with col2:
                if st.button("📥 Executive Summary", use_container_width=True, key="scope_exec_summary"):
                    timestamp = analysis.get('timestamp', 'unknown_time')

                    # Extract only key sections from final assessment
                    final_assessment = analysis.get('final_assessment', 'N/A')

                    # Use GPT to create concise executive summary
                    with st.spinner("Creating executive summary..."):
                        try:
                            prompt = f"""Extract only the following sections from this scope analysis:

1. Identified Gaps (key missing items across proposals)
2. Final Ranking (which EPC has best/worst scope)
3. Recommendation (which to choose and why)

Keep it concise - maximum 300 words total.

FULL ANALYSIS:
{final_assessment}

Return only these 3 sections formatted clearly."""

                            response = st.session_state.gpt_extractor.client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "You extract key points for executive summaries. Be concise and direct."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.1,
                                max_tokens=500
                            )

                            exec_summary_content = response.choices[0].message.content

                            exec_summary = f"""
EXECUTIVE SUMMARY
SCOPE COMPREHENSIVENESS ANALYSIS
Generated: {timestamp}

{'=' * 80}
{exec_summary_content}
{'=' * 80}

Generated by AES EPC Proposal Dashboard
"""
                            st.download_button(
                                label="💾 Download Executive Summary",
                                data=exec_summary,
                                file_name=f"scope_exec_summary_{timestamp.replace(':', '-').replace(' ', '_')}.txt" if timestamp != 'unknown_time' else "scope_exec_summary.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error creating summary: {str(e)}")

            with col3:
                if st.button("📄 Detailed Report", use_container_width=True, key="scope_detailed_report"):
                    timestamp = analysis.get('timestamp', 'unknown_time')

                    # Create full detailed report with all chains
                    detailed_report = f"""
DETAILED SCOPE ANALYSIS REPORT
Multi-Chain Reasoning Analysis
Generated: {timestamp}

{'=' * 80}
CHAIN 1: INITIAL QUALITATIVE ANALYSIS
{'=' * 80}

{analysis.get('initial_analysis', 'No analysis available')}

{'=' * 80}
CHAIN 2: SIGNIFICANCE & RISK EVALUATION
{'=' * 80}

{analysis.get('significance_evaluation', 'No evaluation available')}

{'=' * 80}
CHAIN 3: FINAL VALIDATED ASSESSMENT & RECOMMENDATION
{'=' * 80}

{analysis.get('final_assessment', 'No assessment available')}

{'=' * 80}
END OF REPORT
{'=' * 80}

Generated by AES EPC Proposal Dashboard
Multi-Chain AI Analysis System
"""
                    st.download_button(
                        label="💾 Download Detailed Report",
                        data=detailed_report,
                        file_name=f"scope_detailed_report_{timestamp.replace(':', '-').replace(' ', '_')}.txt" if timestamp != 'unknown_time' else "scope_detailed_report.txt",
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

            # Check if has BESS - use storage capacity (MWh)
            bess_capacity = proposal.get('capacity', {}).get('storage_mwh', 0)

            # Store capacities by project (will take average if multiple proposals)
            if project_name not in project_capacities:
                project_capacities[project_name] = {'pv': [], 'bess': []}

            project_capacities[project_name]['pv'].append(pv_dc_capacity)
            if bess_capacity and bess_capacity > 0:
                project_capacities[project_name]['bess'].append(bess_capacity)

        # Calculate technology totals (average per project)
        tech_capacity = {}
        for project, caps in project_capacities.items():
            # Average PV capacity for this project
            avg_pv = sum(caps['pv']) / len(caps['pv']) if caps['pv'] else 0
            tech_capacity['PV (DC)'] = tech_capacity.get('PV (DC)', 0) + avg_pv

            # Average BESS capacity for this project
            if caps['bess']:
                avg_bess = sum(caps['bess']) / len(caps['bess'])
                tech_capacity['BESS (MWh)'] = tech_capacity.get('BESS (MWh)', 0) + avg_bess

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
    df = st.session_state.data_manager.create_comparison_dataframe(proposals)
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

        # Use data_editor to make Comments column editable
        edited_df = st.data_editor(
            df_display,
            use_container_width=True,
            disabled=[col for col in df_display.columns if col not in ['Comments']],
            hide_index=True,
            key="proposals_table_editor"
        )

        # Save comments if edited
        if edited_df is not None and not edited_df.equals(df_display):
            from config import save_proposals
            for idx, row in edited_df.iterrows():
                filename = row['Filename']
                comment = row.get('Comments', '')
                st.session_state.data_manager.update_comments(filename, comment)
            save_proposals()
            st.success("💾 Comments saved!")

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

    # High-level comparison table
    st.subheader("📋 High-Level EPC Summary")
    df = st.session_state.data_manager.create_epc_ranking_dataframe(proposals)
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

    st.divider()

    # Bid Assumptions Section
    st.subheader("📋 Bid Assumptions Comparison")

    # Check if any proposals have bid assumptions
    proposals_with_assumptions = [p for p in proposals if p.get('bid_assumptions')]

    if not proposals_with_assumptions:
        st.info("No bid assumptions data available.")
    else:
        show_bid_assumptions_comparison(proposals_with_assumptions)

    st.divider()

    # Detailed Schedule of Values (SOV) Section
    st.subheader("💰 Detailed Schedule of Values (SOV)")

    # Check if any proposals have detailed SOV data
    proposals_with_sov = [p for p in proposals if p.get('detailed_sov')]

    if not proposals_with_sov:
        st.info("No detailed SOV data available. Upload proposals with detailed cost breakdowns to see line-item analysis.")
    else:
        st.write(f"**{len(proposals_with_sov)} proposal(s) with detailed SOV data**")

        # Category selector
        sov_categories = [
            "All Categories",
            "Procurement",
            "Design",
            "Contractor General",
            "Civil Works",
            "PV Mechanical",
            "DC Electrical",
            "AC Electrical",
            "Controls & Communications",
            "Testing & Commissioning",
            "Substation",
            "Transmission"
        ]

        selected_category = st.selectbox("Select SOV Category to View:", sov_categories)

        # Build comparison table
        show_detailed_sov_comparison(proposals_with_sov, selected_category)

def show_bid_assumptions_comparison(proposals):
    """Display bid assumptions comparison table - items as rows, EPCs as columns."""
    import pandas as pd

    # Build data dictionary: {item_name: {epc_name: value}}
    comparison_dict = {}
    epc_columns = []

    for proposal in proposals:
        epc_name = proposal.get('epc_contractor', {}).get('company_name', 'Unknown')
        project_name = proposal.get('project_info', {}).get('project_name', 'Unknown')
        bid_assumptions = proposal.get('bid_assumptions', {})
        capacity = proposal.get('capacity', {})
        equipment = proposal.get('equipment', {})

        # Create column name as "EPC - Project"
        col_name = f"{epc_name}"
        epc_columns.append(col_name)

        # Add all assumption items
        items = {
            'Pricing Type': bid_assumptions.get('pricing_type', 'N/A'),
            'Bidder': epc_name,
            'Project Name': project_name,
            'Bid Date': bid_assumptions.get('bid_date', 'N/A'),
            'State': proposal.get('project_info', {}).get('location', {}).get('state', 'N/A'),
            'MWAC (Total)': capacity.get('ac_mw', 'N/A'),
            'MWAC (PV)': capacity.get('ac_mw', 'N/A'),
            'MWDC (PV)': capacity.get('dc_mw', 'N/A'),
            'Storage (MWh)': capacity.get('storage_mwh') if capacity.get('storage_mwh') else 'N/A',
            'PV Design Basis': bid_assumptions.get('pv_design_basis', 'N/A'),
            'HV Design Basis': bid_assumptions.get('hv_design_basis', 'N/A'),
            'Work Schedule': bid_assumptions.get('work_schedule', 'N/A'),
            'Labor Rate': bid_assumptions.get('labor_rate', 'N/A'),
            'Sales Tax Exemption': bid_assumptions.get('sales_tax_exemption', 'N/A'),
            'Module Type': equipment.get('modules', {}).get('manufacturer', 'N/A') + ' ' + equipment.get('modules', {}).get('model', ''),
            'Inverter Type (PV)': equipment.get('inverters', {}).get('manufacturer', 'N/A') + ' ' + equipment.get('inverters', {}).get('model', ''),
            'Racking Type': equipment.get('racking', {}).get('manufacturer', 'N/A') + ' ' + equipment.get('racking', {}).get('type', ''),
            'Battery Type': (equipment.get('batteries') or {}).get('manufacturer', 'N/A') + ' ' + (equipment.get('batteries') or {}).get('model', '') if (equipment.get('batteries') or {}).get('manufacturer') else 'N/A',
            'Battery Capacity (kWh)': (equipment.get('batteries') or {}).get('capacity_kwh', 'N/A') if (equipment.get('batteries') or {}).get('capacity_kwh') else 'N/A',
            'Pile Size': bid_assumptions.get('pile_size', 'N/A'),
            'Galvanized (Y/N and mil thickness)': bid_assumptions.get('pile_galvanized', 'N/A'),
            'Pre-drill (%)': bid_assumptions.get('predrill_percentage', 'N/A'),
            'Pre-drill $/Pile': f"${bid_assumptions.get('predrill_cost_per_pile'):,.2f}" if bid_assumptions.get('predrill_cost_per_pile') else 'N/A',
            'Pile Refusal % Assumed': bid_assumptions.get('pile_refusal_percentage', 'N/A'),
            'Racking Spares': bid_assumptions.get('racking_spares', 'N/A'),
            'IRS Domestic Content Steel': bid_assumptions.get('domestic_content_steel', 'N/A'),
            'IRS Domestic Content Tracker': bid_assumptions.get('domestic_content_tracker', 'N/A')
        }

        # Add each item to comparison dict
        for item_name, value in items.items():
            if item_name not in comparison_dict:
                comparison_dict[item_name] = {}
            comparison_dict[item_name][col_name] = value

    if comparison_dict:
        # Convert to DataFrame with items as rows, EPCs as columns
        df = pd.DataFrame(comparison_dict).T
        df.columns.name = 'CATEGORY'
        df.index.name = 'Assumption/Item'

        st.write(f"**{len(proposals)} proposal(s) comparison**")
        st.dataframe(df, use_container_width=True, height=800)

        # Download button
        st.download_button(
            label="📥 Download Bid Assumptions Comparison (CSV)",
            data=df.to_csv().encode('utf-8'),
            file_name=f"bid_assumptions_comparison_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


def show_detailed_sov_comparison(proposals, selected_category):
    """Display detailed SOV comparison table across EPCs."""
    import pandas as pd

    # Build comparison data
    comparison_data = []

    # Map display names to data keys
    category_map = {
        "Procurement": "procurement",
        "Design": "design",
        "Contractor General": "contractor_general",
        "Civil Works": "civil_works",
        "PV Mechanical": "pv_mechanical",
        "DC Electrical": "dc_electrical",
        "AC Electrical": "ac_electrical",
        "Controls & Communications": "controls_communications",
        "Testing & Commissioning": "testing_commissioning",
        "Substation": "substation",
        "Transmission": "transmission"
    }

    if selected_category == "All Categories":
        # Show category-level summary
        for proposal in proposals:
            epc_name = proposal.get('epc_contractor', {}).get('company_name', 'Unknown')
            detailed_sov = proposal.get('detailed_sov', {})

            for category_display, category_key in category_map.items():
                category_data = detailed_sov.get(category_key, {})
                if not category_data:
                    continue

                # Calculate category total
                total_cost = 0
                total_unit_cost = 0
                item_count = 0

                for item_name, item_data in category_data.items():
                    if isinstance(item_data, dict):
                        cost = item_data.get('cost')
                        unit_cost = item_data.get('unit_cost')
                        if cost and cost > 0:
                            total_cost += cost
                            if unit_cost:
                                total_unit_cost += unit_cost
                            item_count += 1
                    elif item_name == 'electrical' and isinstance(item_data, dict):
                        # Handle substation electrical subcategory
                        for sub_item_name, sub_item_data in item_data.items():
                            if isinstance(sub_item_data, dict):
                                cost = sub_item_data.get('cost')
                                unit_cost = sub_item_data.get('unit_cost')
                                if cost and cost > 0:
                                    total_cost += cost
                                    if unit_cost:
                                        total_unit_cost += unit_cost
                                    item_count += 1

                if total_cost > 0:
                    comparison_data.append({
                        'EPC': epc_name,
                        'Category': category_display,
                        'Total Cost': total_cost,
                        'Unit Cost ($/W)': total_unit_cost,
                        'Line Items': item_count
                    })

        if comparison_data:
            df = pd.DataFrame(comparison_data)
            # Pivot for better comparison
            pivot_df = df.pivot_table(
                index='Category',
                columns='EPC',
                values='Total Cost',
                aggfunc='sum'
            ).fillna(0)

            st.dataframe(
                pivot_df.style.format("${:,.0f}"),
                use_container_width=True
            )

            # Show breakdown chart with grouped bars by EPC
            st.write("### Cost Breakdown by Category")

            # Create plotly grouped bar chart
            import plotly.express as px

            # Sort categories by total cost (descending)
            category_totals = df.groupby('Category')['Total Cost'].sum().sort_values(ascending=False)
            df['Category'] = pd.Categorical(df['Category'], categories=category_totals.index, ordered=True)
            df_sorted = df.sort_values('Category')

            fig = px.bar(
                df_sorted,
                x='Category',
                y='Total Cost',
                color='EPC',
                barmode='group',
                title='SOV Category Costs by EPC',
                labels={'Total Cost': 'Total Cost ($)', 'Category': 'SOV Category'},
                text='Total Cost'
            )

            # Format text on bars
            fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')

            # Update layout
            fig.update_layout(
                xaxis_title="SOV Category",
                yaxis_title="Total Cost ($)",
                yaxis=dict(tickformat="$,.0f"),
                legend_title="EPC Contractor",
                height=500,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        # Show line-item detail for selected category
        category_key = category_map.get(selected_category)
        if not category_key:
            st.warning("Invalid category selected")
            return

        for proposal in proposals:
            epc_name = proposal.get('epc_contractor', {}).get('company_name', 'Unknown')
            detailed_sov = proposal.get('detailed_sov', {})
            category_data = detailed_sov.get(category_key, {})

            if not category_data:
                continue

            for item_name, item_data in category_data.items():
                if isinstance(item_data, dict) and 'cost' in item_data:
                    cost = item_data.get('cost')
                    unit_cost = item_data.get('unit_cost')

                    if cost and cost > 0:
                        comparison_data.append({
                            'EPC': epc_name,
                            'Line Item': item_name.replace('_', ' ').title(),
                            'Cost': cost,
                            'Unit Cost ($/W)': unit_cost if unit_cost else 0
                        })
                elif item_name == 'electrical' and isinstance(item_data, dict):
                    # Handle substation electrical subcategory
                    for sub_item_name, sub_item_data in item_data.items():
                        if isinstance(sub_item_data, dict):
                            cost = sub_item_data.get('cost')
                            unit_cost = sub_item_data.get('unit_cost')

                            if cost and cost > 0:
                                comparison_data.append({
                                    'EPC': epc_name,
                                    'Line Item': f"Electrical - {sub_item_name.replace('_', ' ').title()}",
                                    'Cost': cost,
                                    'Unit Cost ($/W)': unit_cost if unit_cost else 0
                                })

        if comparison_data:
            df = pd.DataFrame(comparison_data)

            # Create pivot table
            pivot_df = df.pivot_table(
                index='Line Item',
                columns='EPC',
                values='Cost',
                aggfunc='sum'
            ).fillna(0)

            # Add delta columns
            epc_columns = pivot_df.columns.tolist()
            if len(epc_columns) > 1:
                pivot_df['Lowest'] = pivot_df.min(axis=1)
                pivot_df['Highest'] = pivot_df.max(axis=1)
                pivot_df['Spread'] = pivot_df['Highest'] - pivot_df['Lowest']

            st.dataframe(
                pivot_df.style.format("${:,.0f}"),
                use_container_width=True
            )

            # Summary metrics
            st.write("### Summary")
            col1, col2, col3 = st.columns(3)

            total_by_epc = df.groupby('EPC')['Cost'].sum()
            with col1:
                st.metric("Lowest Category Total", f"${total_by_epc.min():,.0f}")
            with col2:
                st.metric("Highest Category Total", f"${total_by_epc.max():,.0f}")
            with col3:
                st.metric("Cost Spread", f"${total_by_epc.max() - total_by_epc.min():,.0f}")

        else:
            st.info(f"No cost data available for {selected_category} category")


def show_ai_analysis(proposals):
    """Consolidated AI features: Scope analysis, recommendation report, chatbot."""
    st.subheader("🤖 AI-Powered Analysis")

    if not proposals:
        st.warning("No proposals available for analysis")
        return

    # AI Provider Selection
    from src.ai_provider import get_available_providers, provider_from_name

    available_providers = get_available_providers()

    if not available_providers:
        st.error("No AI providers configured. Please add API keys to your Streamlit secrets.")
        st.info("Supported providers: OPENAI_API_KEY, ANTHROPIC_API_KEY, GROK_API_KEY (or XAI_API_KEY)")
        return

    # Provider selection in sidebar or compact UI
    col_provider, col_info = st.columns([2, 3])

    with col_provider:
        selected_provider = st.selectbox(
            "AI Provider",
            options=available_providers,
            index=0,
            key="ai_provider_select",
            help="Select which AI provider to use for analysis. Different providers may have varying speeds and capabilities."
        )

    with col_info:
        provider_info = {
            "OpenAI (GPT-4o)": "Fast, reliable, best for detailed analysis",
            "Anthropic (Claude)": "Excellent reasoning, nuanced analysis",
            "Grok (xAI)": "Fast responses, good for quick insights"
        }
        st.caption(f"**{selected_provider}**: {provider_info.get(selected_provider, 'AI-powered analysis')}")

    # Update the GPT extractor with selected provider
    if st.session_state.gpt_extractor.provider_name != selected_provider:
        try:
            st.session_state.gpt_extractor.set_provider(selected_provider)
            st.toast(f"Switched to {selected_provider}", icon="🔄")
        except Exception as e:
            st.error(f"Failed to switch provider: {str(e)}")

    st.divider()

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
    """PostgreSQL database client interface."""
    import time

    if not proposals:
        st.warning("No proposals in database. Upload proposals to get started.")
        return

    # Database connection header (PostgreSQL style)
    st.markdown("""
    <div style='background-color: #1e1e1e; padding: 10px; border-radius: 5px; font-family: monospace; color: #d4d4d4;'>
        <span style='color: #4ec9b0;'>postgres@localhost:5432</span>/<span style='color: #ce9178;'>proposals_db</span>
        <span style='color: #608b4e;'># Connected</span> |
        <span style='color: #9cdcfe;'>{} rows in proposals table</span>
    </div>
    """.format(len(proposals)), unsafe_allow_html=True)

    st.write("")

    # Two column layout: Schema on left, Query/Results on right
    col_schema, col_query = st.columns([1, 3])

    with col_schema:
        st.markdown("**📁 Schema Browser**")

        # Database tree
        with st.expander("🗄️ proposals_db", expanded=True):
            st.markdown("**📊 Tables**")
            with st.expander("└─ proposals", expanded=True):
                st.markdown("""
```
Columns (23):
├─ id                  int4
├─ epc_contractor      varchar
├─ project_name        varchar
├─ project_location    varchar
├─ technology          varchar
├─ total_cost          numeric
├─ cost_per_watt_dc    numeric
├─ ac_capacity_mw      numeric
├─ dc_capacity_mw      numeric
├─ modules             varchar
├─ inverters           varchar
├─ racking             varchar
├─ batteries           varchar
├─ proposal_date       date
├─ contact_person      varchar
├─ email               varchar
├─ assumptions_count   int4
├─ exclusions_count    int4
├─ inclusions_count    int4
├─ clarifications_cnt  int4
├─ comments            text
├─ filename            varchar
└─ uploaded_at         timestamp
```
                """)
                st.caption(f"Total rows: {len(proposals)}")

    with col_query:
        # Query Editor
        st.markdown("**📝 Query Editor**")

        # Track execution time
        if 'last_query_time' not in st.session_state:
            st.session_state.last_query_time = None

        query_input = st.text_area(
            "SQL Query / Natural Language:",
            value="SELECT * FROM proposals;",
            height=100,
            key="postgres_query",
            help="Enter SQL syntax or natural language (e.g., 'Show me proposals under $500M')"
        )

        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
        with col_btn1:
            execute_btn = st.button("▶️ Execute", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)

        if execute_btn:
            start_time = time.time()

            if query_input and query_input.strip() not in ["", "SELECT * FROM proposals;"]:
                try:
                    # First, generate SQL translation
                    with st.spinner("Translating to SQL..."):
                        sql_translation = st.session_state.gpt_extractor.client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You translate natural language queries into PostgreSQL SQL. Return ONLY the SQL query, no explanation."},
                                {"role": "user", "content": f"Translate this to SQL for a 'proposals' table:\n\n{query_input}\n\nReturn only the SQL query."}
                            ],
                            temperature=0.1,
                            max_tokens=200
                        )
                        generated_sql = sql_translation.choices[0].message.content.strip()
                        # Clean up SQL formatting
                        if generated_sql.startswith("```sql"):
                            generated_sql = generated_sql.split("```sql")[1].split("```")[0].strip()
                        elif generated_sql.startswith("```"):
                            generated_sql = generated_sql.split("```")[1].strip()

                    # Parse query with AI
                    filtered = st.session_state.gpt_extractor.query_proposals_with_ai(query_input, proposals)
                    execution_time = time.time() - start_time
                    st.session_state.db_query_results = filtered if filtered else []
                    st.session_state.last_query_time = execution_time
                    st.session_state.last_sql_query = generated_sql
                    st.rerun()
                except Exception as e:
                    st.error(f"ERROR: {str(e)}")
            else:
                # Default: show all
                execution_time = time.time() - start_time
                st.session_state.db_query_results = proposals
                st.session_state.last_query_time = execution_time
                st.session_state.last_sql_query = "SELECT * FROM proposals;"
                st.rerun()

        if clear_btn:
            st.session_state.db_query_results = None
            st.session_state.last_query_time = None
            st.session_state.last_sql_query = None
            st.rerun()

        # Show generated SQL query (if available)
        if 'last_sql_query' in st.session_state and st.session_state.last_sql_query:
            st.markdown("**📋 Executed SQL Query:**")
            st.code(st.session_state.last_sql_query, language="sql")
            st.caption("⚠️ Note: This is a simulated SQL representation. The actual query is executed via Python filtering on the JSON data store.")

        st.divider()

        # Results Panel
        results = st.session_state.get('db_query_results', proposals)

        if results is not None:
            # Query execution info
            exec_time = st.session_state.last_query_time
            time_str = f"{exec_time:.3f}s" if exec_time else "N/A"

            st.markdown(f"""
            <div style='background-color: #f0f0f0; padding: 8px; border-radius: 3px; font-family: monospace; font-size: 12px;'>
                <b>Query Result:</b> {len(results)} rows returned |
                <b>Execution time:</b> {time_str} |
                <b>Status:</b> <span style='color: green;'>SUCCESS</span>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            if results:
                # Convert to table
                table_data = []
                for idx, p in enumerate(results, 1):
                    scope = p.get('scope', {})
                    table_data.append({
                        'id': idx,
                        'epc_contractor': p.get('epc_contractor', {}).get('company_name', 'NULL'),
                        'project_name': p.get('project_info', {}).get('project_name', 'NULL'),
                        'project_location': f"{p.get('project_info', {}).get('location', {}).get('city', 'NULL')}, {p.get('project_info', {}).get('location', {}).get('state', 'NULL')}",
                        'technology': p.get('technology', {}).get('type', 'NULL'),
                        'total_cost': p.get('costs', {}).get('total_project_cost', 'NULL'),
                        'cost_per_watt_dc': p.get('costs', {}).get('cost_per_watt_dc', 'NULL'),
                        'ac_capacity_mw': p.get('capacity', {}).get('ac_mw', 'NULL'),
                        'dc_capacity_mw': p.get('capacity', {}).get('dc_mw', 'NULL'),
                        'modules': f"{p.get('equipment', {}).get('modules', {}).get('manufacturer', 'NULL')} {p.get('equipment', {}).get('modules', {}).get('model', '')}".strip(),
                        'inverters': f"{p.get('equipment', {}).get('inverters', {}).get('manufacturer', 'NULL')} {p.get('equipment', {}).get('inverters', {}).get('model', '')}".strip(),
                        'proposal_date': p.get('epc_contractor', {}).get('proposal_date', 'NULL'),
                        'contact_person': p.get('epc_contractor', {}).get('contact_person', 'NULL'),
                        'email': p.get('epc_contractor', {}).get('email', 'NULL'),
                        'assumptions_count': len(scope.get('assumptions', [])),
                        'exclusions_count': len(scope.get('exclusions', [])),
                        'inclusions_count': len(scope.get('inclusions', [])),
                        'comments': p.get('metadata', {}).get('comments', ''),
                        'filename': p.get('metadata', {}).get('filename', 'NULL'),
                    })

                df = pd.DataFrame(table_data)

                # Data table
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )

                st.write("")

                # Export toolbar
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📄 Export CSV",
                        data=csv,
                        file_name=f"query_result_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col2:
                    json_data = json.dumps([p for p in results], indent=2)
                    st.download_button(
                        label="📄 Export JSON",
                        data=json_data,
                        file_name=f"query_result_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )

                with col3:
                    st.metric("Rows", len(results), delta=None)

                with col4:
                    st.metric("Columns", len(df.columns), delta=None)

            else:
                st.info("Query returned 0 rows")
        else:
            st.info("💡 Enter a query and click Execute to view results")