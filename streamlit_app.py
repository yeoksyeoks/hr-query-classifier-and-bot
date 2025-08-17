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
            st.session_state["password_attempted"] = True

    # Check if already authenticated
    if st.session_state.get("password_correct", False):
        return True
    
    # Show password input
    st.text_input(
        "Enter Password", 
        type="password", 
        on_change=password_entered, 
        key="password",
        placeholder="Enter access password"
    )
    
    # Only show error after an attempt has been made and it was incorrect
    if st.session_state.get("password_attempted", False) and not st.session_state.get("password_correct", False):
        st.error("Password incorrect")
    
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
        st.warning("Some system components are not available. Check the sidebar for details.")
    
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

    # Required disclaimer at the bottom
    st.markdown("---")
    with st.expander("Important Notice - Please Read"):
        st.markdown("""
        **IMPORTANT NOTICE:** This web application is developed as a proof-of-concept prototype. The information provided here is **NOT intended for actual usage** and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.
        
        **Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.**
        
        Always consult with qualified professionals for accurate and personalized advice.
        """)

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

            # Initialise processing state
            if 'processing_state' not in st.session_state:
                st.session_state['processing_state'] = 'ready'

            # Create placeholders for live updates
            progress_container = st.empty()
            status_container = st.empty()
            
            # Create placeholder for main action button
            button_placeholder = st.empty()

            # Logic to render correct button and handle state transitions
            if st.session_state['processing_state'] == 'ready':
                if button_placeholder.button("Start Processing", type="secondary", use_container_width=True):
                    if not os.getenv('OPENAI_API_KEY'):
                        st.error("OpenAI API key not found. Please configure your API key in the sidebar.")
                        return # Exit the function if API key is missing
                    
                    st.session_state['processing_state'] = 'processing'
                    st.session_state['total_queries_to_process'] = max_queries
                    st.rerun() # Rerun to show "Processing..." state

            elif st.session_state['processing_state'] == 'processing':
                button_placeholder.info("Processing in progress... Please wait.")

                # This block runs after 'Start Processing' button is clicked
                if 'processing_completed' not in st.session_state:
                    try:
                        # USE THE EXISTING process_queries_with_live_updates FUNCTION
                        result_df = process_queries_with_live_updates(
                            df.head(st.session_state['total_queries_to_process']),
                            query_column,
                            st.session_state['total_queries_to_process'],
                            progress_container,
                            status_container
                        )
                        
                        st.session_state['processed_data'] = result_df
                        st.session_state['processing_state'] = 'complete'
                        
                        # Generate summary (simple version since get_processing_summary might not exist)
                        try:
                            from hr_classifier_py import get_processing_summary
                            summary = get_processing_summary(result_df)
                            st.session_state['last_processing_stats'] = summary
                        except ImportError:
                            # Create a simple summary if the function doesn't exist
                            success_count = len(result_df[result_df['processing_status'] == 'SUCCESS'])
                            total_count = len(result_df)
                            summary = {
                                'total_queries': total_count,
                                'success_count': success_count,
                                'success_rate': (success_count / total_count * 100) if total_count > 0 else 0
                            }
                            st.session_state['last_processing_stats'] = summary
                        
                        if 'activity_log' not in st.session_state:
                            st.session_state['activity_log'] = []
                        st.session_state['activity_log'].append(
                            f"{datetime.now().strftime('%H:%M')} - Processed {len(result_df)} queries"
                        )
                    except Exception as e:
                        st.session_state['processing_state'] = 'error'
                        st.session_state['processing_error'] = str(e)
                    finally:
                        st.session_state['processing_completed'] = True
                        st.rerun() # Rerun to show final results

            elif st.session_state['processing_state'] == 'complete':
                button_placeholder.success("Processing Complete! View results below.")
                
                col_go, col_reset = st.columns(2)
                with col_go:
                    if st.button("Go to RAG Chatbot", type="primary", use_container_width=True):
                        st.session_state['current_page'] = "RAG Chatbot"
                        st.rerun()
                with col_reset:
                    if st.button("Process Another Batch", use_container_width=True):
                        # Clear processing state
                        keys_to_clear = ['processed_data', 'processing_completed', 'processing_error']
                        for key in keys_to_clear:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.session_state['processing_state'] = 'ready'
                        st.rerun()
            
            elif st.session_state['processing_state'] == 'error':
                error_msg = st.session_state.get('processing_error', 'Unknown error occurred')
                button_placeholder.error(f"Processing Error! {error_msg}")
                
                # Show error details
                with st.expander("Error Details"):
                    st.code(error_msg)
                    st.write("**Common solutions:**")
                    st.write("- Check your API key is valid")
                    st.write("- Try processing fewer queries")
                    st.write("- Check your internet connection")
                    st.write("- Wait a few minutes and retry")
                
                if st.button("Retry Processing", use_container_width=True):
                    # Clear error state
                    keys_to_clear = ['processed_data', 'processing_completed', 'processing_error']
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.session_state['processing_state'] = 'ready'
                    st.rerun()

            # Show results if processing is complete
            if 'processed_data' in st.session_state and st.session_state['processing_state'] in ['complete', 'error']:
                st.markdown("### Processing Results")
                result_df = st.session_state['processed_data']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Processed", len(result_df))
                with col2:
                    success_count = len(result_df[result_df['processing_status'] == 'SUCCESS'])
                    st.metric("Successful", success_count)
                with col3:
                    success_rate = (success_count / len(result_df)) * 100 if len(result_df) > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")

                st.dataframe(result_df, use_container_width=True)

                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name=f"hr_query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
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
    
    # Check API key first
    if not os.getenv('OPENAI_API_KEY'):
        st.error("**API Key Required**: Please configure your OpenAI API key in the sidebar before using the chatbot.")
        st.info("**Tip**: Look for the 'Configure API Key' section in the left sidebar.")
        return
    
    # Check if processed data is available from Batch Classifier
    if 'processed_data' not in st.session_state:
        st.warning("**No Processed Data Found**")
        st.info("Please process some HR queries using the **Batch HR Query Classifier** first.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Batch Classifier", use_container_width=True):
                st.session_state['current_page'] = "Batch Classifier"
                st.rerun()
        
        with col2:
            # Option to upload processed file manually
            st.markdown("**Or upload processed data:**")
            uploaded_file = st.file_uploader(
                "Upload processed HR queries CSV",
                type=['csv'],
                help="Upload a CSV file with processed HR queries (must contain 'summary' and 'topic_classification' columns)"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    required_columns = ['summary', 'topic_classification']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        st.error(f"Missing required columns: {missing_columns}")
                    else:
                        st.session_state['processed_data'] = df
                        st.success(f"Loaded {len(df)} processed queries!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        
        return
    
    st.markdown("Chat with our AI assistant about HR topics based on your processed query knowledge base.")
    
    # Initialise RAG bot in session state following original pattern
    if 'rag_bot' not in st.session_state:
        st.session_state['rag_bot'] = None
        st.session_state['rag_bot_ready'] = False
    
    # Setup section - following original structure
    if not st.session_state['rag_bot_ready']:
        st.markdown("### Setup Knowledge Base")
        processed_df = st.session_state['processed_data']
        st.success(f"Using processed data: {len(processed_df)} queries available")
        
        try:
            # Initialise RAG bot
            from hr_rag_bot_py import HRQueryRAGBot
            
            if st.session_state['rag_bot'] is None:
                with st.spinner("Initialising RAG bot..."):
                    rag_bot = HRQueryRAGBot()
                    st.session_state['rag_bot'] = rag_bot
            else:
                rag_bot = st.session_state['rag_bot']
            
            # Load processed data
            temp_file_path = f"temp_processed_data_{int(time.time())}.csv"
            processed_df.to_csv(temp_file_path, index=False)
            
            with st.spinner("Loading processed data..."):
                processed_data = rag_bot.load_processed_data(temp_file_path)
            
            # Clean up temp file
            try:
                os.remove(temp_file_path)
            except:
                pass
            
            # Show available topics
            available_topics = rag_bot.list_available_topics()
            st.success(f"Found {len(available_topics)} different HR topics")
            
            # Topic selection - following original pattern
            st.markdown("### Select Topics for Knowledge Base")
            
            # Show topic distribution
            topic_counts = processed_data['topic_classification'].value_counts()
            
            with st.expander("Topic Distribution"):
                st.bar_chart(topic_counts.head(10))
            
            # Multi-select for topics
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_topics = st.multiselect(
                    "Choose topics to include in knowledge base:",
                    options=available_topics,
                    default=topic_counts.head(5).index.tolist(),
                    help="Select topics you want to chat about. More topics = longer setup time."
                )
            
            with col2:
                max_topics = st.number_input(
                    "Max topics to setup:",
                    min_value=1,
                    max_value=20,
                    value=5,
                    help="Limit topics to avoid long setup times"
                )
            
            if selected_topics:
                if st.button("Setup Knowledge Base", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Setup topics with progress updates - following original
                        selected_topics_limited = selected_topics[:max_topics]
                        
                        for i, topic in enumerate(selected_topics_limited):
                            progress = (i / len(selected_topics_limited))
                            progress_bar.progress(progress)
                            status_text.text(f"Setting up topic {i+1}/{len(selected_topics_limited)}: {topic}")
                            
                            try:
                                rag_bot.create_topic_vector_database(topic)
                            except Exception as e:
                                st.warning(f"Failed to setup topic '{topic}': {str(e)}")
                        
                        progress_bar.progress(1.0)
                        status_text.text("Knowledge base setup complete!")
                        
                        # Update session state
                        st.session_state['rag_bot_ready'] = True
                        
                        st.success("Knowledge base ready! You can now start chatting.")
                        time.sleep(2)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error during setup: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error initialising RAG bot: {str(e)}")
    
    else:
        # Chat interface - simplified but following original structure
        rag_bot = st.session_state['rag_bot']
        ready_topics = rag_bot.list_ready_topics()
        
        if not ready_topics:
            st.error("No topics are ready for chatting. Please restart the setup.")
            if st.button("Restart Setup"):
                st.session_state['rag_bot_ready'] = False
                st.session_state['rag_bot'] = None
                st.rerun()
            return
        
        # Show knowledge base status - following original
        st.success(f"Knowledge base ready with {len(ready_topics)} topics!")
        
        with st.expander("Knowledge Base Status"):
            status = rag_bot.get_setup_status()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Queries", status['total_queries'])
            with col2:
                st.metric("Available Topics", status['total_topics_available'])
            with col3:
                st.metric("Ready Topics", status['topics_with_vector_db'])
        
        # Topic selection for chat - following original
        st.markdown("### Chat Interface")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_topic = st.selectbox(
                "Choose HR topic to discuss:",
                options=ready_topics,
                help="Select the HR topic you want to ask questions about"
            )
        
        with col2:
            show_sources = st.checkbox("Show sources", value=True, help="Display source documents with responses")
        
        # Initialise chat history for the selected topic - following original pattern
        chat_key = f"chat_history_{selected_topic.replace(' ', '_').replace('/', '_')}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []
        
        # Display chat history - following original structure
        st.markdown("### Chat History")
        
        if st.session_state[chat_key]:
            for i, chat in enumerate(st.session_state[chat_key]):
                # User message
                with st.chat_message("user"):
                    st.write(chat['question'])
                
                # Assistant response
                with st.chat_message("assistant"):
                    if 'error' in chat:
                        st.error(chat['error'])
                    else:
                        st.write(chat['answer'])
                        
                        # Show sources if available
                        if show_sources and 'sources' in chat:
                            with st.expander(f"Sources ({chat.get('num_sources', 0)})"):
                                for source in chat['sources']:
                                    st.markdown(f"**Source {source['source_number']}:**")
                                    st.text(source['content'])
                                    if source['metadata']:
                                        st.json(source['metadata'])
        else:
            st.info(f"Start chatting about {selected_topic}! Ask any questions related to this HR topic.")
        
        # Chat input - simplified
        if user_question := st.chat_input(f"Ask about {selected_topic}..."):
            # Add user message to chat history
            with st.chat_message("user"):
                st.write(user_question)
            
            # Get response from RAG bot - following original pattern
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = rag_bot.chat_with_topic(
                        topic=selected_topic,
                        question=user_question,
                        show_sources=show_sources
                    )
                    
                    if 'error' in response:
                        st.error(response['error'])
                    else:
                        st.write(response['answer'])
                        
                        # Show sources if available
                        if show_sources and 'sources' in response:
                            with st.expander(f"Sources ({response.get('num_sources', 0)})"):
                                for source in response['sources']:
                                    st.markdown(f"**Source {source['source_number']}:**")
                                    st.text(source['content'])
                                    if source['metadata']:
                                        st.json(source['metadata'])
                    
                    # Add to chat history - following original
                    st.session_state[chat_key].append(response)
                    
                    # Update activity log
                    if 'activity_log' not in st.session_state:
                        st.session_state['activity_log'] = []
                    st.session_state['activity_log'].append(
                        f"{datetime.now().strftime('%H:%M')} - Asked about {selected_topic}"
                    )
        
        # Additional options
        st.markdown("### Additional Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state[chat_key]:
                # Create DataFrame from chat history for direct download
                chat_data = []
                for chat in st.session_state[chat_key]:
                    chat_data.append({
                        'timestamp': chat.get('timestamp', ''),
                        'topic': chat.get('topic', selected_topic),
                        'question': chat.get('question', ''),
                        'answer': chat.get('answer', ''),
                        'sources_count': chat.get('num_sources', 0)
                    })
                
                chat_df = pd.DataFrame(chat_data)
                csv = chat_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Chat History",
                    data=csv,
                    file_name=f"chat_history_{selected_topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No chat history to download")
        
        with col2:
            if st.button("Restart Setup", use_container_width=True):
                # Clear RAG bot state but keep processed data
                keys_to_clear = ['rag_bot', 'rag_bot_ready']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                # Clear chat histories
                chat_keys = [key for key in st.session_state.keys() if key.startswith('chat_history_')]
                for key in chat_keys:
                    del st.session_state[key]
                st.rerun()
                
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
