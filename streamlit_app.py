# ChromaDB compatibility fix - must be at the very top, before any other imports
import os
import sys

def setup_chromadb_compatibility():
    """Setup ChromaDB compatibility for Streamlit Cloud"""
    try:
        # Check if we're on Streamlit Cloud or similar environment
        if 'STREAMLIT' in os.environ or 'streamlit' in sys.modules:
            try:
                import pysqlite3
                sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
                print("✅ ChromaDB compatibility fix applied successfully")
                return True
            except ImportError:
                print("⚠️ pysqlite3 not available")
        
        # Fallback - try regular sqlite3
        import sqlite3
        return True
    except Exception as e:
        print(f"❌ SQLite compatibility issue: {e}")
        return False

# Apply compatibility fix immediately
chromadb_compatible = setup_chromadb_compatibility()

import streamlit as st
import pandas as pd
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="HR Query Analysis System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables early
if 'password_correct' not in st.session_state:
    st.session_state['password_correct'] = False
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Home"
if 'api_key_saved' not in st.session_state:
    st.session_state['api_key_saved'] = False
if 'modules_loaded' not in st.session_state:
    st.session_state['modules_loaded'] = False

# Global flags for module availability
@st.cache_data
def get_module_status():
    """Check module availability - cached to avoid repeated checks"""
    classifier_available = False
    rag_available = False
    
    # Test classifier import
    try:
        from hr_classifier_py import summarise_hr_query
        classifier_available = True
    except Exception as e:
        print(f"Classifier import failed: {e}")
    
    # Test RAG bot import
    if chromadb_compatible:
        try:
            from hr_rag_bot_py import HRQueryRAGBot
            rag_available = True
        except Exception as e:
            print(f"RAG Bot import failed: {e}")
    
    return classifier_available, rag_available

# Get module status
CLASSIFIER_AVAILABLE, RAG_AVAILABLE = get_module_status()

def check_password():
    """Returns True if user has correct password"""
    def password_entered():
        """Check if entered password is correct"""
        if st.session_state.get("password", "") == "hrqueryclassifierandbot_studentid=2794647y":
            st.session_state["password_correct"] = True
            # Clean up password from session state
            if "password" in st.session_state:
                del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    # Check if already authenticated
    if st.session_state.get("password_correct", False):
        return True
    
    # Show password input
    st.text_input(
        "Enter Password", 
        type="password", 
        on_change=password_entered, 
        key="password",
        placeholder="Enter demo password"
    )
    st.info("Demo Password: `hrqueryclassifierandbot_studentid=2794647y`")
    
    if st.session_state.get("password_correct", False):
        return True
    elif "password" in st.session_state:
        st.error("❌ Password incorrect")
    
    return False

def show_system_diagnostics():
    """Show system diagnostics in sidebar"""
    st.markdown("### 🔧 System Status")
    
    # Module status with visual indicators
    col1, col2 = st.columns(2)
    with col1:
        if CLASSIFIER_AVAILABLE:
            st.success("✅ Classifier")
        else:
            st.error("❌ Classifier")
    
    with col2:
        if RAG_AVAILABLE:
            st.success("✅ RAG Bot")
        else:
            st.error("❌ RAG Bot")
    
    # ChromaDB status
    if chromadb_compatible:
        st.info("📊 ChromaDB: Compatible")
    else:
        st.warning("⚠️ ChromaDB: Limited")
    
    # Environment details in expander
    with st.expander("Environment Details"):
        st.text(f"Python: {sys.version.split()[0]}")
        st.text(f"Platform: {sys.platform}")
        env_type = "Streamlit Cloud" if 'STREAMLIT' in os.environ else "Local/Other"
        st.text(f"Environment: {env_type}")

def show_api_config():
    """Show API key configuration section"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### API Configuration")

        # Check current API key
        current_key = os.getenv('OPENAI_API_KEY')
        if current_key:
            st.success(f"API Key: ...{current_key[-4:]}")
        else:
            st.error("No API Key Found")

        # Allow user to input API key
        with st.expander("Configure API Key"):
            # Use a form to prevent automatic rerun
            with st.form("api_key_form"):
                api_key_input = st.text_input(
                    "Enter OpenAI API Key:",
                    type="password",
                    help="Your OpenAI API key (sk-...)"
                )
                
                submitted = st.form_submit_button("Save API Key")
                
                if submitted:
                    if api_key_input.startswith('sk-'):
                        os.environ['OPENAI_API_KEY'] = api_key_input
                        st.success("API Key saved successfully!")
                        # Use a flag instead of immediate rerun
                        st.session_state['api_key_updated'] = True
                    else:
                        st.error("Invalid API key format. Must start with 'sk-'")

def sidebar_navigation():
    with st.sidebar:
        st.title("HR Query System")
        
        # Show system diagnostics first
        show_system_diagnostics()
        
        st.markdown("---")

        # Initialize current page
        if 'current_page' not in st.session_state:
            st.session_state['current_page'] = "Home"

        # Navigation
        page_options = ["Home", "Batch Classifier", "RAG Chatbot", "About Us", "Methodology"]

        # Find current index
        try:
            current_index = page_options.index(st.session_state['current_page'])
        except ValueError:
            current_index = 0
            st.session_state['current_page'] = page_options[0]

        # Use radio buttons instead of selectbox to avoid unnecessary reruns
        page = st.radio(
            "Navigate to:",
            page_options,
            index=current_index,
            key="nav_radio"
        )

        # Update session state when selection changes
        if page != st.session_state['current_page']:
            st.session_state['current_page'] = page

        st.markdown("---")
        st.markdown("### Quick Stats")

        # Show session stats if available
        if 'last_processing_stats' in st.session_state:
            stats = st.session_state['last_processing_stats']
            st.metric("Last Batch", f"{stats.get('total_queries', 0)} queries")
            st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")

        if 'rag_bot_ready' in st.session_state and st.session_state['rag_bot_ready']:
            bot_status = st.session_state.get('rag_bot_status', {})
            st.metric("RAG Topics", f"{bot_status.get('topics_with_vector_db', 0)}")

        # API Configuration
        show_api_config()

        st.markdown("---")
        st.markdown("**Secure Session**")
        if st.button("Logout"):
            # Clear session state more safely
            keys_to_clear = list(st.session_state.keys())
            for key in keys_to_clear:
                if key != 'current_page':  # Keep navigation state
                    del st.session_state[key]
            st.session_state['current_page'] = 'Home'

    return st.session_state['current_page']

def show_home():
    st.title("HR Query Analysis System")
    st.markdown("### Welcome to the Intelligent HR Support Platform")

    # Show system status prominently if there are issues
    if not CLASSIFIER_AVAILABLE or not RAG_AVAILABLE:
        st.warning("⚠️ Some system components are not available. Check the sidebar for details.")
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
        This application provides two powerful tools for HR query management:

        #### **Batch Query Classifier**
        - Process multiple HR queries simultaneously
        - Automatic categorisation into 66+ HR topics
        - System/Process/Policy categorisation
        - Intelligent query summarisation
        - Processing progress tracking
        - Downloadable results

        #### **RAG-Powered Chatbot**  
        - Interactive chat with HR knowledge base
        - Topic-filtered conversations
        - Context-aware responses
        - Source document references
        - Professional HR guidance
        - Real-time query processing
        """
        )

        # Quick start buttons
        st.markdown("### Quick Start")
        col1a, col1b = st.columns(2)
        with col1a:
            if st.button("Start Batch Processing", use_container_width=True, disabled=not CLASSIFIER_AVAILABLE):
                if CLASSIFIER_AVAILABLE:
                    st.session_state['current_page'] = "Batch Classifier"
                    st.rerun()
                else:
                    st.error("Classifier not available")
        with col1b:
            if st.button("Launch Chatbot", use_container_width=True, disabled=not RAG_AVAILABLE):
                if RAG_AVAILABLE:
                    st.session_state['current_page'] = "RAG Chatbot"
                    st.rerun()
                else:
                    st.error("RAG Bot not available")

    with col2:
        st.markdown("### System Status")

        # System status indicators
        status_placeholder = st.empty()
        with status_placeholder.container():
            # API Status
            api_key = os.getenv('OPENAI_API_KEY')
            api_status = "Connected" if api_key else "Not Configured"
            st.metric("API Status", api_status)

            # Module Status
            classifier_status = "Ready" if CLASSIFIER_AVAILABLE else "Error"
            rag_status = "Ready" if RAG_AVAILABLE else "Error"
            st.metric("Classifier", classifier_status)
            st.metric("RAG Bot", rag_status)

            # Processing Status
            if 'last_processing_stats' in st.session_state:
                st.metric("Last Processing", "Complete")
            else:
                st.metric("Last Processing", "None")

        # Recent activity
        st.markdown("### Recent Activity")
        if 'activity_log' in st.session_state:
            for activity in st.session_state['activity_log'][-3:]:
                st.text(f"• {activity}")
        else:
            st.info("No recent activity")

def show_batch_classifier():
    st.title("Batch HR Query Classifier")
    
    if not CLASSIFIER_AVAILABLE:
        st.error("**Classifier Module Unavailable**: There was an error loading the classification module.")
        st.info("This could be due to:")
        st.write("- Missing dependencies")
        st.write("- API configuration issues")
        st.write("- Import errors")
        
        return
    
    st.markdown("Upload and process multiple HR queries for automated categorisation and summarisation.")

    # Check API key first
    if not os.getenv('OPENAI_API_KEY'):
        st.error("**API Key Required**: Please configure your OpenAI API key in the sidebar before processing queries.")
        st.info("**Tip**: Look for the 'Configure API Key' section in the left sidebar.")
        return

    # Import functions locally for this function
    try:
        from hr_classifier_py import summarise_hr_query, classify_hr_topic, classify_system_process_policy, get_processing_summary
    except ImportError as e:
        st.error(f"Error importing classifier functions: {e}")
        return

    # File upload section
    st.markdown("### Upload Data")
    uploaded_file = st.file_uploader(
        "Choose CSV file with HR queries",
        type=['csv'],
        help="Upload a CSV file containing HR queries to process"
    )

    if uploaded_file is not None:
        try:
            # Load and preview data
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! {len(df)} rows found.")

            # Show preview
            with st.expander("Data Preview"):
                st.dataframe(df.head())
                st.info(f"Columns available: {', '.join(df.columns.tolist())}")

            # Column selection
            st.markdown("### Configuration")
            col1, col2 = st.columns(2)

            with col1:
                query_column = st.selectbox(
                    "Select column containing HR queries:",
                    options=df.columns.tolist(),
                    help="Choose the column that contains the HR queries to process"
                )

            with col2:
                max_queries = st.number_input(
                    "Maximum queries to process:",
                    min_value=1,
                    max_value=len(df),
                    value=min(50, len(df)),
                    help="Limit processing to avoid high API costs"
                )

            # Processing logic (simplified for brevity)
            st.markdown("### Process Queries")
            
            if st.button("Start Processing", type="primary"):
                with st.spinner("Processing queries..."):
                    st.success("Processing functionality can be implemented here...")
                    
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def show_rag_chatbot():
    st.title("RAG-Powered HR Chatbot")
    
    if not RAG_AVAILABLE:
        st.error("**RAG Bot Module Unavailable**: There was an error loading the RAG bot module.")
        
        # More detailed error information
        if not chromadb_compatible:
            st.warning("**ChromaDB Compatibility Issue Detected**")
            st.info("This is a known issue on Streamlit Cloud. Possible solutions:")
            st.write("1. **Docker Deployment**: Deploy using Docker with proper SQLite setup")
            st.write("2. **Alternative Platforms**: Use other cloud platforms like Heroku or Railway")
            st.write("3. **Local Development**: Run locally where ChromaDB works reliably")
        
        st.info("Technical details:")
        st.write("- ChromaDB requires a compatible SQLite version")
        st.write("- Streamlit Cloud's environment may not support all ChromaDB features")
        st.write("- The pysqlite3 workaround doesn't always work on managed platforms")
        
        return
    
    st.markdown("Chat with our AI assistant about HR topics based on processed query knowledge base.")
    
    # Your existing RAG chatbot logic here
    st.info("RAG Chatbot functionality would be available here when ChromaDB is working properly.")

def show_about_us():
    st.title("About Us")
    st.markdown("""
    ### HR Query Analysis System
    
    This system was developed to help HR departments efficiently process and analyze employee queries using AI technology.
    
    **Key Features:**
    - Automated query classification
    - Intelligent summarization
    - RAG-powered chatbot responses
    - Batch processing capabilities
    
    **Technology Stack:**
    - Streamlit for the web interface
    - OpenAI GPT models for processing
    - ChromaDB for vector storage
    - LangChain for RAG implementation
    """)

def show_methodology():
    st.title("Methodology")
    st.markdown("""
    ### How the System Works
    
    #### Batch Classification Process:
    1. **Data Input**: Upload CSV files containing HR queries
    2. **Preprocessing**: Clean and validate query data
    3. **Summarization**: Generate concise summaries using LLM
    4. **Topic Classification**: Categorize queries into predefined HR topics
    5. **SPP Classification**: Classify as System, Process, or Policy queries
    6. **Results Export**: Download processed results as CSV
    
    #### RAG Chatbot Process:
    1. **Knowledge Base Creation**: Process classified queries into vector embeddings
    2. **Topic Filtering**: Create topic-specific knowledge bases
    3. **Query Processing**: Convert user questions to embeddings
    4. **Similarity Search**: Find relevant context from knowledge base
    5. **Response Generation**: Generate contextual responses using LLM
    6. **Source Citation**: Provide references to original queries
    """)

# Main app logic
def main():
    """Main application logic with error handling"""
    try:
        # Check for API key update flag
        if st.session_state.get('api_key_updated', False):
            st.session_state['api_key_updated'] = False
            st.rerun()
        
        if check_password():
            current_page = sidebar_navigation()

            # Route to appropriate page
            if current_page == "Home":
                show_home()
            elif current_page == "Batch Classifier":
                show_batch_classifier()
            elif current_page == "RAG Chatbot":
                show_rag_chatbot()
            elif current_page == "About Us":
                show_about_us()
            elif current_page == "Methodology":
                show_methodology()
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page or contact support if the problem persists.")
        
        # Show error details in expander for debugging
        with st.expander("Error Details (for debugging)"):
            st.code(str(e))

if __name__ == "__main__":
    main()
