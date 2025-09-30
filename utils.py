import streamlit as st
from config import save_proposals, PROPOSALS_DIR
import os
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def get_existing_pdfs():
    """Get list of PDFs already stored in data/proposals folder."""
    if not os.path.exists(PROPOSALS_DIR):
        return []

    pdf_files = [f for f in os.listdir(PROPOSALS_DIR) if f.lower().endswith('.pdf')]

    # Get file info
    pdf_info = []
    for filename in pdf_files:
        filepath = os.path.join(PROPOSALS_DIR, filename)
        file_size = os.path.getsize(filepath)
        file_size_mb = file_size / (1024 * 1024)
        modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))

        pdf_info.append({
            'filename': filename,
            'filepath': filepath,
            'size_mb': file_size_mb,
            'modified': modified_time
        })

    # Sort by modified time (newest first)
    pdf_info.sort(key=lambda x: x['modified'], reverse=True)
    return pdf_info

def process_existing_pdf(pdf_info):
    """Process a PDF that's already stored in the data folder."""
    filename = pdf_info['filename']
    filepath = pdf_info['filepath']

    try:
        # Check if already processed
        existing_proposals = st.session_state.data_manager.get_all_proposals()

        # Check by stored filename or regular filename
        for p in existing_proposals:
            stored_name = p.get('metadata', {}).get('stored_filename', '')
            original_name = p.get('metadata', {}).get('filename', '')
            if stored_name == filename or original_name == filename:
                return False, f"Already processed: {filename}"

        # Read and process the PDF
        with open(filepath, 'rb') as f:
            text = st.session_state.pdf_processor.extract_text_from_pdf(f)

        if not text or len(text.strip()) < 50:
            return False, f"PDF appears empty or has minimal text"

        clean_text = st.session_state.pdf_processor.clean_text(text)

        # Extract data using GPT
        proposal_data = st.session_state.gpt_extractor.extract_project_data(clean_text)

        if not proposal_data:
            return False, "Could not extract proposal data"

        # Extract scope details
        scope_data = st.session_state.gpt_extractor.extract_scope_details(clean_text)
        proposal_data['scope'] = scope_data

        # Add metadata
        proposal_data['metadata'] = {
            'filename': filename,
            'stored_filename': filename,
            'pdf_path': filepath,
            'uploaded_at': datetime.now().isoformat(),
            'file_size': pdf_info['size_mb'] * 1024 * 1024
        }

        # Add to data manager
        st.session_state.data_manager.proposals.append(proposal_data)

        # Build scope summary
        scope_summary = []
        if scope_data.get('assumptions'):
            scope_summary.append(f"{len(scope_data['assumptions'])} assumptions")
        if scope_data.get('exclusions'):
            scope_summary.append(f"{len(scope_data['exclusions'])} exclusions")
        if scope_data.get('clarifications'):
            scope_summary.append(f"{len(scope_data['clarifications'])} clarifications")
        if scope_data.get('inclusions'):
            scope_summary.append(f"{len(scope_data['inclusions'])} inclusions")

        scope_info = f" (Extracted: {', '.join(scope_summary)})" if scope_summary else ""

        return True, f"Successfully processed{scope_info}"

    except Exception as e:
        return False, f"Error: {str(e)}"

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF and Excel files."""
    with st.spinner("Processing uploaded files..."):
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.type == "application/pdf":
                    # Process PDF
                    try:
                        # Check if this file was already processed
                        existing_proposals = st.session_state.data_manager.get_all_proposals()
                        if any(p.get('metadata', {}).get('filename') == uploaded_file.name for p in existing_proposals):
                            st.info(f"ℹ️ File '{uploaded_file.name}' already processed (skipping duplicate)")
                            continue

                        # Save PDF to data folder with timestamp to avoid conflicts
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        safe_filename = f"{timestamp}_{uploaded_file.name}"
                        pdf_path = os.path.join(PROPOSALS_DIR, safe_filename)

                        with open(pdf_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # Extract text from PDF
                        text = st.session_state.pdf_processor.extract_text_from_pdf(uploaded_file)
                        if not text or len(text.strip()) < 50:
                            st.warning(f"⚠️ {uploaded_file.name}: PDF appears empty or has minimal text")
                            os.remove(pdf_path)  # Clean up saved file
                            continue

                        clean_text = st.session_state.pdf_processor.clean_text(text)

                        # Extract data using GPT
                        with st.spinner(f"Extracting proposal data from {uploaded_file.name}..."):
                            proposal_data = st.session_state.gpt_extractor.extract_project_data(clean_text)

                        if not proposal_data:
                            st.warning(f"⚠️ {uploaded_file.name}: Could not extract proposal data")
                            os.remove(pdf_path)  # Clean up saved file
                            continue

                        # Extract scope details separately for better accuracy
                        with st.spinner(f"Extracting scope details (assumptions, exclusions, clarifications)..."):
                            scope_data = st.session_state.gpt_extractor.extract_scope_details(clean_text)
                            proposal_data['scope'] = scope_data

                        # Add metadata including PDF path
                        proposal_data['metadata'] = {
                            'filename': uploaded_file.name,
                            'stored_filename': safe_filename,
                            'pdf_path': pdf_path,
                            'uploaded_at': datetime.now().isoformat(),
                            'file_size': uploaded_file.size
                        }

                        # Add to data manager (without duplicate check since we already checked)
                        st.session_state.data_manager.proposals.append(proposal_data)

                        # Show what was extracted
                        scope_summary = []
                        if scope_data.get('assumptions'):
                            scope_summary.append(f"{len(scope_data['assumptions'])} assumptions")
                        if scope_data.get('exclusions'):
                            scope_summary.append(f"{len(scope_data['exclusions'])} exclusions")
                        if scope_data.get('clarifications'):
                            scope_summary.append(f"{len(scope_data['clarifications'])} clarifications")
                        if scope_data.get('inclusions'):
                            scope_summary.append(f"{len(scope_data['inclusions'])} inclusions")

                        scope_info = f" (Extracted: {', '.join(scope_summary)})" if scope_summary else ""
                        st.success(f"✅ Successfully processed and saved: {uploaded_file.name}{scope_info}")

                    except Exception as pdf_error:
                        st.error(f"❌ Error processing PDF {uploaded_file.name}: {str(pdf_error)}")
                        # Clean up saved file if it exists
                        if 'pdf_path' in locals() and os.path.exists(pdf_path):
                            os.remove(pdf_path)
                        continue

                elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
                    # Process Excel - could enhance this to extract cost data
                    try:
                        excel_data = st.session_state.pdf_processor.extract_excel_data(uploaded_file)
                        st.success(f"✅ Excel file '{uploaded_file.name}' uploaded successfully")
                        # For now, just store the filename - could integrate cost data extraction later
                    except Exception as excel_error:
                        st.error(f"❌ Error processing Excel {uploaded_file.name}: {str(excel_error)}")
                        continue

                else:
                    st.warning(f"⚠️ Unsupported file type: {uploaded_file.name} ({uploaded_file.type})")

            except Exception as e:
                st.error(f"❌ Unexpected error processing {uploaded_file.name}: {str(e)}")

    # Save proposals after processing
    save_proposals()

def apply_filters():
    """Apply filters to proposals and store filtered list in session state."""
    all_proposals = st.session_state.data_manager.get_all_proposals()

    if not all_proposals:
        st.session_state.filtered_proposals = []
        return

    # EPC Contractor filter
    epc_options = list(set([p.get('epc_contractor', {}).get('company_name', 'Unknown') for p in all_proposals]))
    selected_epc = st.selectbox("EPC Contractor", ["All"] + sorted(epc_options), key="epc_filter")

    # Technology filter
    tech_options = list(set([p.get('technology', {}).get('type', 'Unknown') for p in all_proposals]))
    selected_tech = st.selectbox("Technology Type", ["All"] + sorted(tech_options), key="tech_filter")

    # Capacity range filter
    capacities = [p.get('capacity', {}).get('ac_mw') for p in all_proposals if p.get('capacity', {}).get('ac_mw')]
    min_cap, max_cap = None, None
    if capacities:
        min_capacity = float(min(capacities))
        max_capacity = float(max(capacities))

        # Only show slider if we have a range
        if min_capacity != max_capacity:
            min_cap, max_cap = st.slider(
                "AC Capacity Range (MW)",
                min_value=min_capacity,
                max_value=max_capacity,
                value=(min_capacity, max_capacity),
                key="capacity_filter"
            )
        else:
            st.info(f"All proposals have the same capacity: {min_capacity} MW")
            min_cap, max_cap = min_capacity, max_capacity

    # State filter
    states = list(set([p.get('project_info', {}).get('location', {}).get('state', 'Unknown') for p in all_proposals]))
    selected_state = st.selectbox("State", ["All"] + sorted(states), key="state_filter")

    # Apply filters
    filtered = all_proposals

    if selected_epc != "All":
        filtered = [p for p in filtered if p.get('epc_contractor', {}).get('company_name', 'Unknown') == selected_epc]

    if selected_tech != "All":
        filtered = [p for p in filtered if p.get('technology', {}).get('type', 'Unknown') == selected_tech]

    if min_cap is not None and max_cap is not None:
        filtered = [p for p in filtered if p.get('capacity', {}).get('ac_mw') and min_cap <= p['capacity']['ac_mw'] <= max_cap]

    if selected_state != "All":
        filtered = [p for p in filtered if p.get('project_info', {}).get('location', {}).get('state', 'Unknown') == selected_state]

    st.session_state.filtered_proposals = filtered

    # Show filter summary
    if len(filtered) != len(all_proposals):
        st.info(f"Showing {len(filtered)} of {len(all_proposals)} proposals (filters applied)")