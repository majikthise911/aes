import streamlit as st
import os
import time
from dotenv import load_dotenv

# Load environment variables first
env_path = os.path.join(os.path.dirname(__file__), 'config', '.env')
load_dotenv(env_path)

# Page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="EPC Proposal Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modular components
from config import save_proposals, load_proposals, apply_custom_css
from utils import process_uploaded_files, apply_filters, get_existing_pdfs, process_existing_pdf
from views import show_overview, show_map_view, show_cost_analysis, show_comparison_view, show_ai_report, show_chatbot

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables."""
    if 'initialized' not in st.session_state:
        try:
            from src.data_manager import DataManager
            from src.pdf_processor import PDFProcessor
            from src.gpt_extractor import GPTExtractor

            st.session_state.data_manager = DataManager()
            st.session_state.pdf_processor = PDFProcessor()
            st.session_state.gpt_extractor = GPTExtractor()

            # Load existing proposals
            saved_proposals = load_proposals()
            if saved_proposals:
                st.session_state.data_manager.proposals = saved_proposals
                st.session_state.proposals_loaded = True

            st.session_state.initialized = True
        except ImportError as e:
            st.error(f"⚠️ Error importing required modules: {e}")
            st.info("Make sure all dependencies are installed. Run: pip install -r requirements.txt")
            st.stop()
        except Exception as e:
            st.error(f"⚠️ Error initializing application: {e}")
            st.stop()

def main():
    st.markdown('<div class="main-header">⚡ EPC Proposal Dashboard</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        # Command Chat at the top
        st.header("💬 AI Assistant")

        # Initialize command AI
        if 'command_ai' not in st.session_state:
            from src.command_ai import CommandAI
            st.session_state.command_ai = CommandAI()

        if 'command_messages' not in st.session_state:
            st.session_state.command_messages = [
                {"role": "assistant", "content": "👋 Hi! I can help you manage your proposals. Try:\n- 'Add cost data from Excel to Blattner'\n- 'Show only proposals under $500M'\n- 'Delete a proposal'\n\nJust tell me what you need!"}
            ]

        # Show chat messages
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.command_messages[-3:]:  # Show last 3 messages
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**AI:** {msg['content']}")

        # Command input
        command_input = st.text_input("Command:", key="command_input", placeholder="What would you like to do?")

        # File upload for commands
        command_file = st.file_uploader("Upload file (optional)", type=['xlsx', 'xls', 'pdf'], key="command_file", label_visibility="collapsed")

        if st.button("Send", type="primary", use_container_width=True):
            if command_input:
                # Add user message
                st.session_state.command_messages.append({"role": "user", "content": command_input})

                # Interpret command
                all_proposals = st.session_state.data_manager.get_all_proposals()
                intent = st.session_state.command_ai.interpret_command(
                    command_input,
                    all_proposals,
                    has_file=command_file is not None
                )

                # Execute action
                response_msg = intent.get('response', 'Processing...')

                if intent['action'] == 'add_cost_data' and command_file:
                    # Parse Excel and update proposal
                    cost_data = st.session_state.command_ai.parse_excel_sov(command_file)

                    if cost_data:
                        target_epc = intent.get('target_epc', 'Unknown')
                        success, msg = st.session_state.command_ai.update_proposal_costs(
                            all_proposals,
                            target_epc,
                            cost_data
                        )

                        if success:
                            save_proposals()
                            response_msg = f"✅ {msg}\n\nTotal Cost: ${cost_data['total_project_cost']:,.2f}"
                        else:
                            response_msg = f"❌ {msg}"
                    else:
                        response_msg = "❌ Could not parse cost data from Excel file. Please check the file format."

                elif intent['action'] == 'add_cost_data' and not command_file:
                    response_msg = "📎 Please upload an Excel file with cost data, then send your command again."

                elif intent['action'] == 'question':
                    response_msg = "💡 For questions, please use the 'Ask AI' tab in the main area where I can provide detailed answers!"

                else:
                    response_msg = intent.get('response', 'I understand, but this action is not yet implemented.')

                st.session_state.command_messages.append({"role": "assistant", "content": response_msg})
                st.rerun()

        st.divider()

        st.header("📁 Manage Proposals")

        # Create tabs for upload methods
        upload_tab, existing_tab = st.tabs(["Upload New", "Load Existing"])

        with upload_tab:
            # File upload section
            uploaded_files = st.file_uploader(
                "Upload EPC Proposals",
                type=['pdf', 'xlsx', 'xls'],
                accept_multiple_files=True,
                help="Upload PDF proposals and Excel schedule of values"
            )

            if uploaded_files:
                process_uploaded_files(uploaded_files)

        with existing_tab:
            st.write("Load previously stored PDFs from data folder")

            # Get existing PDFs
            existing_pdfs = get_existing_pdfs()

            if not existing_pdfs:
                st.info("No PDFs found in data/proposals folder")
            else:
                st.write(f"**{len(existing_pdfs)} PDF(s) available:**")

                # Show PDFs with checkboxes
                if 'selected_pdfs' not in st.session_state:
                    st.session_state.selected_pdfs = set()

                for pdf in existing_pdfs:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        is_checked = st.checkbox(
                            pdf['filename'],
                            key=f"pdf_{pdf['filename']}",
                            help=f"{pdf['size_mb']:.1f} MB • Modified: {pdf['modified'].strftime('%Y-%m-%d %H:%M')}"
                        )
                        if is_checked:
                            st.session_state.selected_pdfs.add(pdf['filename'])
                        elif pdf['filename'] in st.session_state.selected_pdfs:
                            st.session_state.selected_pdfs.remove(pdf['filename'])

                st.divider()

                # Process selected PDFs
                if st.session_state.selected_pdfs:
                    if st.button(f"📥 Process {len(st.session_state.selected_pdfs)} Selected PDF(s)", type="primary", use_container_width=True):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        start_time = time.time()

                        selected_pdf_list = [pdf for pdf in existing_pdfs if pdf['filename'] in st.session_state.selected_pdfs]
                        total = len(selected_pdf_list)
                        success_count = 0

                        for idx, pdf in enumerate(selected_pdf_list):
                            elapsed = time.time() - start_time
                            avg_time = elapsed / (idx + 1) if idx > 0 else 0
                            remaining = (total - idx - 1) * avg_time

                            status_text.text(f"Processing {idx + 1}/{total}: {pdf['filename']} (Est. {int(remaining)}s remaining)")

                            success, message = process_existing_pdf(pdf)

                            if success:
                                success_count += 1
                                st.success(f"✅ {pdf['filename']}: {message}")
                            else:
                                st.warning(f"⚠️ {pdf['filename']}: {message}")

                            progress_bar.progress((idx + 1) / total)

                        elapsed = time.time() - start_time
                        status_text.empty()
                        progress_bar.empty()

                        # Save and refresh
                        save_proposals()
                        st.session_state.selected_pdfs.clear()
                        st.success(f"✅ Processed {success_count}/{total} PDF(s) in {int(elapsed)}s!")
                        st.rerun()

        st.divider()

        # Filters section
        st.header("🔍 Filters")
        apply_filters()

        st.divider()

        # Actions section
        st.header("⚙️ Actions")
        if st.button("Clear All Data", type="secondary"):
            st.session_state.data_manager.clear_all_proposals()
            save_proposals()
            st.rerun()

    # Main content area
    # Use filtered proposals if available, otherwise all proposals
    if 'filtered_proposals' in st.session_state:
        proposals = st.session_state.filtered_proposals
    else:
        proposals = st.session_state.data_manager.get_all_proposals()

    if not proposals:
        st.info("👆 Upload EPC proposals using the sidebar to get started")
        return

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "🗺️ Map View", "💰 Cost Analysis", "📋 Comparison", "📝 AI Report", "💬 Ask AI"])

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

    with tab6:
        show_chatbot(proposals)

if __name__ == "__main__":
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        st.error("⚠️ OpenAI API key not found. Please set your OPENAI_API_KEY environment variable.")
        st.info("Create a .env file in the config folder with your API key.")
        st.stop()

    # Initialize session state and apply styling
    initialize_session_state()
    apply_custom_css()

    # Show one-time message for loaded proposals
    if st.session_state.get('proposals_loaded') and not st.session_state.get('load_message_shown'):
        proposal_count = len(st.session_state.data_manager.get_all_proposals())
        if proposal_count > 0:
            st.toast(f"✅ Loaded {proposal_count} previous proposals", icon="📁")
        st.session_state.load_message_shown = True

    main()