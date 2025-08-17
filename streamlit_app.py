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

def test_classifier_functions():
    """Test if the classifier functions work with a simple query"""
    st.write("🔍 Testing Classifier Functions...")
    
    # Test API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("❌ No API key found")
        return False
    else:
        st.success(f"✅ API Key found: ...{api_key[-4:]}")
    
    # Test imports
    try:
        from hr_classifier_py import summarise_hr_query, classify_hr_topic, classify_system_process_policy
        st.success("✅ All functions imported successfully")
    except Exception as e:
        st.error(f"❌ Import error: {str(e)}")
        return False
    
    # Test a simple query
    test_query = "I need help with my medical leave application"
    
    try:
        st.write("Testing summarization...")
        summary = summarise_hr_query(test_query)
        st.success(f"✅ Summary: {summary}")
    except Exception as e:
        st.error(f"❌ Summarization failed: {str(e)}")
        return False
    
    try:
        st.write("Testing topic classification...")
        topics = ['Medical Leave', 'Other HR Matters']  # Simplified list
        topic = classify_hr_topic(test_query, topics)
        st.success(f"✅ Topic: {topic}")
    except Exception as e:
        st.error(f"❌ Topic classification failed: {str(e)}")
        return False
    
    try:
        st.write("Testing SPP classification...")
        spp_def = "Policy: Related to criteria, policy clarifications\nProcess: Related to procedures, requests\nSystem: Related to technical issues"
        spp = classify_system_process_policy(test_query, spp_def)
        st.success(f"✅ SPP: {spp}")
    except Exception as e:
        st.error(f"❌ SPP classification failed: {str(e)}")
        return False
    
    st.success("🎉 All tests passed!")
    return True

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

        # ADD THE DEBUG TEST HERE
        st.markdown("---")
        st.markdown("### 🔧 Debug Tools")
        if st.button("🔧 Run Classifier Test", use_container_width=True):
            test_classifier_functions()

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

def process_queries_with_live_updates(df_process, query_column, total_queries, progress_container, status_container):
    """Process queries with real-time progress updates"""

    if df_process is None:
        st.error("Error: The input DataFrame is empty or not provided.")
        return
    if query_column is None:
        st.error("Error: The query column name is not provided.")
        return

    # Import here to avoid circular imports
    from hr_classifier_py import summarise_hr_query, classify_hr_topic, classify_system_process_policy

    # Topic categories
    topic_categories = [
        'Annual Variable Component and CONNECT Plan', 'Recruitment', 'Posting',
        'Other HR Matters', 'Medical Leave', 'Performance Appraisal',
        'Professional Development', 'Casual Employment Scheme',
        'Pro-Family Full Pay Leave', 'Medical and Dental Benefits',
        'Leaving Service', 'No-Pay Leave', 'Awards',
        'Letter of Employment, Testimonial',
        'IT Matters (HRP, HR Online, AM Assist)',
        'Updating of Personal Particulars and Educational Qualifications',
        'Pre-NIE/NIE Admission Matters', 'PS and CS Cards', 'Discipline',
        'Urgent Private Affairs Leave/Vacation Leave/Sabbatical Leave',
        'Insurance', 'Scholarships and Talent Management', 'Salary Matters',
        'Confirmation after probation', 'Bond and Liquidated Damages',
        'Injuries During Course of Work', 'Performance Reward',
        'Annual Declaration', 'Flexi Work Arrangement', 'Other Leave', 'HRP GRC',
        'Compensation Policy', 'HRP Payroll', 'HRP Employee Service',
        'Working Hours / Workload', 'Election Manpower Matters', 'Starting Salary',
        "Employees requested/CSC provided HR's contact", 'Claims - LDS, Others',
        'MKE Commitment Payment', 'HRP Others',
        'Participation in External Activities',
        'HRG – Organisational Development & Psychology',
        'Transfer of service/scheme', 'Contractual Matters and Contract Renewal',
        'HRP Deploy', 'Training-Related (PSIP, Induction Courses)',
        'Change of Subject Proficiency', 'Contributions to Charity',
        'Salary Review',
        'Key Personnel/Senior, Lead, Master And Principal Master Teacher Matters',
        'Career Development', 'HRP Perform', 'SIUE', 'Emplacement Matters',
        'HRP Attract', 'Emplacement Salary', 'Other Benefits', 'CPF and Income Tax',
        'Annual Exercise', 'HRG – Carpark Charges Masterlist',
        'HRP Data Management', 'Returning to Service', 'Deceased',
        'Students - Teacher Ratio Matters', 'Benefits Policy',
        'IT Matters (iCON, CES Email, TRAISI)',
        'MK Matters: Internship and Work Attachment',
        'Performance Appraisal (MKEs)', 'Review of Scheme of Service',
        'Outstanding Contribution Award', 'Posting Policy', 'HRP Develop',
        'Career Management', 'Senior Key Personnel (KP) Rotation'
    ]

    spp_definitions = """
    Policy: Related to criteria, policy clarifications
    Process: Related to procedures, requests, timeline, submission of documents
    System: Related to technical issues
    """

    # Initialise result dataframe
    result_df = df_process.copy()
    result_df['summary'] = ''
    result_df['topic_classification'] = ''
    result_df['spp_classification'] = ''
    result_df['processing_status'] = ''

    # Process each query with live updates
    processed_count = 0
    success_count = 0
    error_count = 0

    for index, row in result_df.iterrows():
        query = row[query_column]
        processed_count += 1

        # Calculate progress
        progress_percentage = (processed_count / total_queries) * 100

        # Update session_state for live values
        st.session_state['processing_progress'] = {
            'processed': processed_count,
            'total': total_queries,
            'successful': success_count,
            'errors': error_count,
            'percentage': progress_percentage
        }

        # Update progress display in real-time
        with progress_container.container():
            st.progress(progress_percentage / 100)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Processed", f"{processed_count}/{total_queries}")
            with col2:
                st.metric("Successful", success_count)
            with col3:
                st.metric("Errors", error_count)
            with col4:
                st.metric("Progress", f"{progress_percentage:.1f}%")

        # Update status with current query
        with status_container.container():
            if pd.isna(query) or str(query).strip() == '':
                st.info(f"Skipping empty query {processed_count}/{total_queries}")
            else:
                st.info(f"Processing query {processed_count}/{total_queries}: {str(query)[:100]}...")

        if pd.isna(query) or str(query).strip() == '':
            result_df.at[index, 'processing_status'] = 'SKIPPED - Empty query'
            continue

        try:
            # Update status for summarisation
            with status_container.container():
                st.info(f"Summarising query {processed_count}...")

            # Generate summary
            summary = summarise_hr_query(query)
            result_df.at[index, 'summary'] = summary
            time.sleep(0.1)

            # Update status for topic classification
            with status_container.container():
                st.info(f"Categorising topic for query {processed_count}...")

            # Classify topic
            topic = classify_hr_topic(query, topic_categories)
            result_df.at[index, 'topic_classification'] = topic
            time.sleep(0.1)

            # Update status for SPP classification
            with status_container.container():
                st.info(f"Categorising SPP for query {processed_count}...")

            # Classify SPP
            spp_class = classify_system_process_policy(query, spp_definitions)
            result_df.at[index, 'spp_classification'] = spp_class

            result_df.at[index, 'processing_status'] = 'SUCCESS'
            success_count += 1

            # Show success for current query
            with status_container.container():
                st.success(f"Completed query {processed_count}: {topic} | {spp_class}")

            time.sleep(0.2)

        except Exception as e:
            result_df.at[index, 'processing_status'] = f'ERROR: {str(e)}'
            error_count += 1

            # Show error for current query
            with status_container.container():
                st.error(f"Error processing query {processed_count}: {str(e)}")

        # Update session state after processing each query
        st.session_state['processing_progress'] = {
            'processed': processed_count,
            'total': total_queries,
            'successful': success_count,
            'errors': error_count,
            'percentage': (processed_count / total_queries) * 100
        }

    # Final progress update
    with progress_container.container():
        st.progress(1.0)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Processed", f"{processed_count}/{total_queries}")
        with col2:
            st.metric("Successful", success_count)
        with col3:
            st.metric("Errors", error_count)
        with col4:
            st.metric("Progress", "100.0%")

    with status_container.container():
        st.success(f"Processing complete! {success_count} successful, {error_count} errors")

    return result_df

def show_batch_classifier():
    st.title("Batch HR Query Classifier")
    st.markdown("Upload and process multiple HR queries for automated categorisation and summarisation.")

    # Check if classifier is available
    if not CLASSIFIER_AVAILABLE:
        st.error("**Classifier Module Unavailable**: Please check the system diagnostics in the sidebar.")
        return

    # Check API key first
    if not os.getenv('OPENAI_API_KEY'):
        st.error("**API Key Required**: Please configure your OpenAI API key in the sidebar before processing queries.")
        st.info("**Tip**: Look for the 'Configure API Key' section in the left sidebar.")
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

            st.markdown("### Process Queries")

            # Initialize processing state
            if 'processing_state' not in st.session_state:
                st.session_state['processing_state'] = 'ready'

            # Create placeholders for live updates
            progress_container = st.empty()
            status_container = st.empty()
            
            # Create placeholder for main action button
            button_placeholder = st.empty()

            # Logic to render correct button and handle state transitions
            if st.session_state['processing_state'] == 'ready':
                if button_placeholder.button("Start Processing", type="primary", use_container_width=True):
                    if not os.getenv('OPENAI_API_KEY'):
                        st.error("OpenAI API key not found. Please configure your API key in the sidebar.")
                        return
                    
                    st.session_state['processing_state'] = 'processing'
                    st.session_state['total_queries_to_process'] = max_queries
                    st.rerun()

            elif st.session_state['processing_state'] == 'processing':
                button_placeholder.info("🔄 Processing in progress... Please wait.")

                # This block runs after 'Start Processing' button is clicked
                if 'processing_completed' not in st.session_state:
                    try:
                        # Detailed debug logging
                        debug_info = []
                        debug_info.append("Starting processing...")
                        
                        # Check if import is successful
                        try:
                            from hr_classifier_py import process_hr_queries, get_processing_summary
                            debug_info.append("✅ Import successful")
                        except ImportError as e:
                            debug_info.append(f"❌ Import failed: {str(e)}")
                            raise e
                        
                        # Validate data
                        debug_info.append(f"Data validation: {len(df)} total rows")
                        
                        # Check if the selected column exists and has valid data
                        if query_column not in df.columns:
                            raise ValueError(f"Selected column '{query_column}' not found in data")
                            
                        # Check for empty queries
                        df_subset = df.head(max_queries).copy()
                        empty_queries = df_subset[query_column].isnull().sum()
                        if empty_queries > 0:
                            debug_info.append(f"⚠️ Found {empty_queries} empty queries")
                            # Remove empty queries
                            df_subset = df_subset[df_subset[query_column].notna()]
                            debug_info.append(f"Processing {len(df_subset)} non-empty queries")
                        else:
                            debug_info.append(f"✅ All {len(df_subset)} queries have content")
                        
                        if len(df_subset) == 0:
                            raise ValueError("No valid queries found to process")
                        
                        # Topic categories (same as in your original code)
                        topic_categories = [
                            'Annual Variable Component and CONNECT Plan', 'Recruitment', 'Posting',
                            'Other HR Matters', 'Medical Leave', 'Performance Appraisal',
                            'Professional Development', 'Casual Employment Scheme',
                            'Pro-Family Full Pay Leave', 'Medical and Dental Benefits',
                            'Leaving Service', 'No-Pay Leave', 'Awards',
                            'Letter of Employment, Testimonial',
                            'IT Matters (HRP, HR Online, AM Assist)',
                            'Updating of Personal Particulars and Educational Qualifications',
                            'Pre-NIE/NIE Admission Matters', 'PS and CS Cards', 'Discipline',
                            'Urgent Private Affairs Leave/Vacation Leave/Sabbatical Leave',
                            'Insurance', 'Scholarships and Talent Management', 'Salary Matters',
                            'Confirmation after probation', 'Bond and Liquidated Damages',
                            'Injuries During Course of Work', 'Performance Reward',
                            'Annual Declaration', 'Flexi Work Arrangement', 'Other Leave', 'HRP GRC',
                            'Compensation Policy', 'HRP Payroll', 'HRP Employee Service',
                            'Working Hours / Workload', 'Election Manpower Matters', 'Starting Salary',
                            "Employees requested/CSC provided HR's contact", 'Claims - LDS, Others',
                            'MKE Commitment Payment', 'HRP Others',
                            'Participation in External Activities',
                            'HRG – Organisational Development & Psychology',
                            'Transfer of service/scheme', 'Contractual Matters and Contract Renewal',
                            'HRP Deploy', 'Training-Related (PSIP, Induction Courses)',
                            'Change of Subject Proficiency', 'Contributions to Charity',
                            'Salary Review',
                            'Key Personnel/Senior, Lead, Master And Principal Master Teacher Matters',
                            'Career Development', 'HRP Perform', 'SIUE', 'Emplacement Matters',
                            'HRP Attract', 'Emplacement Salary', 'Other Benefits', 'CPF and Income Tax',
                            'Annual Exercise', 'HRG – Carpark Charges Masterlist',
                            'HRP Data Management', 'Returning to Service', 'Deceased',
                            'Students - Teacher Ratio Matters', 'Benefits Policy',
                            'IT Matters (iCON, CES Email, TRAISI)',
                            'MK Matters: Internship and Work Attachment',
                            'Performance Appraisal (MKEs)', 'Review of Scheme of Service',
                            'Outstanding Contribution Award', 'Posting Policy', 'HRP Develop',
                            'Career Management', 'Senior Key Personnel (KP) Rotation'
                        ]
                        
                        spp_definitions = """
                        Policy: Related to criteria, policy clarifications
                        Process: Related to procedures, requests, timeline, submission of documents
                        System: Related to technical issues
                        """
                        
                        debug_info.append(f"✅ Configuration complete: {len(topic_categories)} categories")
                        
                        # Update status with debug info
                        with status_container.container():
                            st.info("🔍 Processing queries... (Debug mode enabled)")
                            with st.expander("Debug Information"):
                                for info in debug_info:
                                    st.write(info)
                        
                        # Create temporary file with better error handling
                        import tempfile
                        import uuid
                        
                        temp_file = f"temp_processing_data_{uuid.uuid4().hex[:8]}.csv"
                        debug_info.append(f"Creating temp file: {temp_file}")
                        
                        try:
                            df_subset.to_csv(temp_file, index=False)
                            debug_info.append("✅ Temp file created successfully")
                        except Exception as e:
                            debug_info.append(f"❌ Failed to create temp file: {str(e)}")
                            raise e
                        
                        # Update progress
                        with progress_container.container():
                            st.progress(0.3, text="Processing queries with AI models...")
                        
                        # Process the queries with timeout and retry logic
                        max_retries = 3
                        retry_count = 0
                        result_df = None
                        
                        while retry_count < max_retries and result_df is None:
                            try:
                                debug_info.append(f"Processing attempt {retry_count + 1}/{max_retries}")
                                
                                # Update status
                                with status_container.container():
                                    st.info(f"🤖 Processing queries... (Attempt {retry_count + 1})")
                                    with st.expander("Debug Information"):
                                        for info in debug_info:
                                            st.write(info)
                                
                                result_df = process_hr_queries(
                                    temp_file, 
                                    query_column, 
                                    topic_categories, 
                                    spp_definitions
                                )
                                
                                debug_info.append("✅ Processing completed successfully")
                                break
                                
                            except Exception as e:
                                retry_count += 1
                                error_msg = str(e)
                                debug_info.append(f"❌ Attempt {retry_count} failed: {error_msg}")
                                
                                if retry_count < max_retries:
                                    debug_info.append(f"⏳ Retrying in 5 seconds...")
                                    time.sleep(5)
                                else:
                                    debug_info.append("❌ All retry attempts exhausted")
                                    raise e
                        
                        # Clean up temp file
                        try:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                                debug_info.append("✅ Temp file cleaned up")
                        except Exception as e:
                            debug_info.append(f"⚠️ Failed to cleanup temp file: {str(e)}")
                        
                        # Validate results
                        if result_df is None or len(result_df) == 0:
                            raise ValueError("Processing returned no results")
                        
                        debug_info.append(f"✅ Results validated: {len(result_df)} processed queries")
                        
                        # Update progress
                        with progress_container.container():
                            st.progress(0.9, text="Finalizing results...")
                        
                        # Store results
                        st.session_state['processed_data'] = result_df
                        st.session_state['processing_state'] = 'complete'
                        st.session_state['debug_info'] = debug_info
                        
                        # Generate summary
                        try:
                            summary = get_processing_summary(result_df)
                            st.session_state['last_processing_stats'] = summary
                            debug_info.append("✅ Summary generated")
                        except Exception as e:
                            debug_info.append(f"⚠️ Failed to generate summary: {str(e)}")
                            # Continue without summary
                        
                        # Update activity log
                        if 'activity_log' not in st.session_state:
                            st.session_state['activity_log'] = []
                        st.session_state['activity_log'].append(
                            f"{datetime.now().strftime('%H:%M')} - Processed {len(result_df)} queries"
                        )
                        
                        st.session_state['processing_completed'] = True
                        
                        # Final progress update
                        with progress_container.container():
                            st.progress(1.0, text="Processing complete!")
                        
                        st.rerun()
                        
                    except Exception as e:
                        # Detailed error handling
                        error_details = {
                            'error_type': type(e).__name__,
                            'error_message': str(e),
                            'timestamp': datetime.now().isoformat(),
                        }
                        
                        # Add debug information if available
                        if 'debug_info' in locals():
                            error_details['debug_info'] = debug_info
                        
                        st.session_state['processing_state'] = 'error'
                        st.session_state['processing_error'] = error_details
                        st.session_state['processing_completed'] = True
                        st.rerun()

            elif st.session_state['processing_state'] == 'complete':
                button_placeholder.success("✅ Processing Complete! View results below.")
                
                # Show debug info if available
                if 'debug_info' in st.session_state:
                    with st.expander("Processing Debug Log"):
                        for info in st.session_state['debug_info']:
                            st.write(info)
                
                col_go, col_reset = st.columns(2)
                with col_go:
                    if st.button("🤖 Go to RAG Chatbot", type="primary", use_container_width=True):
                        st.session_state['current_page'] = "RAG Chatbot"
                        st.rerun()
                with col_reset:
                    if st.button("🔄 Process Another Batch", use_container_width=True):
                        # Clear processing state
                        keys_to_clear = ['processed_data', 'processing_completed', 'processing_error', 'debug_info']
                        for key in keys_to_clear:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.session_state['processing_state'] = 'ready'
                        st.rerun()
            
            elif st.session_state['processing_state'] == 'error':
                error_details = st.session_state.get('processing_error', {})
                error_msg = error_details.get('error_message', 'Unknown error occurred')
                error_type = error_details.get('error_type', 'Unknown')
                
                button_placeholder.error(f"❌ Processing Error ({error_type}): {error_msg}")
                
                # Show detailed error information
                with st.expander("Error Details"):
                    st.json(error_details)
                
                col_retry, col_reset = st.columns(2)
                with col_retry:
                    if st.button("🔄 Retry Processing", type="primary", use_container_width=True):
                        # Clear error state but keep data
                        keys_to_clear = ['processing_completed', 'processing_error', 'debug_info']
                        for key in keys_to_clear:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.session_state['processing_state'] = 'ready'
                        st.rerun()
                        
                with col_reset:
                    if st.button("🆕 Start Fresh", use_container_width=True):
                        # Clear all processing state
                        keys_to_clear = ['processed_data', 'processing_completed', 'processing_error', 'debug_info']
                        for key in keys_to_clear:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.session_state['processing_state'] = 'ready'
                        st.rerun()

            # Show results if processing is complete
            if ('processed_data' in st.session_state and 
                st.session_state['processing_state'] == 'complete'):
                
                st.markdown("### 📊 Processing Results")
                result_df = st.session_state['processed_data']
                
                # Show summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Processed", len(result_df))
                with col2:
                    # Handle case where 'processing_status' column might not exist
                    if 'processing_status' in result_df.columns:
                        success_count = len(result_df[result_df['processing_status'] == 'SUCCESS'])
                    else:
                        # Assume all are successful if no status column
                        success_count = len(result_df)
                    st.metric("Successful", success_count)
                with col3:
                    success_rate = (success_count / len(result_df)) * 100 if len(result_df) > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")

                # Show the processed data
                st.dataframe(result_df, use_container_width=True)

                # Download button
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name=f"hr_query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Please check that your CSV file is properly formatted.")
            
            # Show file reading error details
            with st.expander("File Error Details"):
                st.write("**Error Type:**", type(e).__name__)
                st.write("**Error Message:**", str(e))
                st.write("**Suggestions:**")
                st.write("- Ensure the file is a valid CSV format")
                st.write("- Check for special characters in column names")
                st.write("- Verify the file is not corrupted")
                st.write("- Try saving the file with UTF-8 encoding")
                
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
