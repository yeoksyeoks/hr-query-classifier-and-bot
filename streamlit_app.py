import streamlit as st
import pandas as pd
import time
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="HR Query Analysis System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import functions
try:
    from hr_classifier_py import summarise_hr_query, classify_hr_topic, classify_system_process_policy, process_hr_queries, get_processing_summary
    from hr_rag_bot_py import HRQueryRAGBot
except ImportError:
    st.error("Import Error: Make sure hr_classifier_py.py and hr_rag_bot_py.py are in the same directory")

# Password protection
def check_password():
    """Returns True if user has correct password"""
    def password_entered():
        """Check if entered password is correct"""
        if st.session_state["password"] == "hrqueryclassifierandbot_studentid=2794647y":  # Updated password
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter Password", type="password", on_change=password_entered, key="password")
        st.info("Demo Password: `hrqueryclassifierandbot_studentid=2794647y`")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Enter Password", type="password", on_change=password_entered, key="password")
        st.error("Password incorrect")
        return False
    else:
        return True

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
            api_key_input = st.text_input(
                "Enter OpenAI API Key:",
                type="password",
                help="Your OpenAI API key (sk-...)"
            )

            if st.button("Save API Key"):
                if api_key_input.startswith('sk-'):
                    os.environ['OPENAI_API_KEY'] = api_key_input
                    st.success("API Key saved!")
                    st.rerun()
                else:
                    st.error("Invalid API key format")

def sidebar_navigation():
    with st.sidebar:
        st.title("HR Query System")
        st.markdown("---")

        # Initialise current page
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

        page = st.selectbox(
            "Navigate to:",
            page_options,
            index=current_index,
            key="nav_selectbox"
        )

        # Update session state when selection changes
        if page != st.session_state['current_page']:
            st.session_state['current_page'] = page
            st.rerun()

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
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    return st.session_state['current_page']

def show_home():
    st.title("HR Query Analysis System")
    st.markdown("### Welcome to the Intelligent HR Support Platform")

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
            if st.button("Start Batch Processing", use_container_width=True):
                st.session_state['current_page'] = "Batch Classifier"
                st.rerun()
        with col1b:
            if st.button("Launch Chatbot", use_container_width=True):
                st.session_state['current_page'] = "RAG Chatbot"
                st.rerun()

    with col2:
        st.markdown("### System Status")

        # System status indicators
        status_placeholder = st.empty()
        with status_placeholder.container():
            # API Status
            api_key = os.getenv('OPENAI_API_KEY')
            api_status = "Connected" if api_key else "Not Configured"
            st.metric("API Status", api_status)

            # Processing Status
            if 'last_processing_stats' in st.session_state:
                st.metric("Last Processing", "Complete")
            else:
                st.metric("Last Processing", "None")

            # RAG Bot Status
            if 'rag_bot_ready' in st.session_state and st.session_state['rag_bot_ready']:
                st.metric("RAG Bot", "Ready")
            else:
                st.metric("RAG Bot", "Not Initialised")

        # Recent activity
        st.markdown("### Recent Activity")
        if 'activity_log' in st.session_state:
            for activity in st.session_state['activity_log'][-3:]:
                st.text(f"• {activity}")
        else:
            st.info("No recent activity")

def process_queries_with_live_updates(df_process, query_column, total_queries, progress_container, status_container):
    """Process queries with real-time progress updates"""

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
                        result_df = process_queries_with_live_updates(
                            df.head(st.session_state['total_queries_to_process']),
                            query_column,
                            st.session_state['total_queries_to_process'],
                            progress_container,
                            status_container
                        )
                        st.session_state['processed_data'] = result_df
                        st.session_state['processing_state'] = 'complete'
                        
                        summary = get_processing_summary(result_df)
                        st.session_state['last_processing_stats'] = summary
                        if 'activity_log' not in st.session_state:
                            st.session_state['activity_log'] = []
                        st.session_state['activity_log'].append(
                            f"{datetime.now().strftime('%H:%M')} - Processed {len(result_df)} queries"
                        )
                    except Exception as e:
                        st.session_state['processing_state'] = 'error'
                        st.error(f"Processing error: {str(e)}")
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
                        del st.session_state['processed_data']
                        del st.session_state['processing_completed']
                        st.session_state['processing_state'] = 'ready'
                        st.rerun()
            
            elif st.session_state['processing_state'] == 'error':
                button_placeholder.error("Processing Error! Retry below.")
                if st.button("Retry Processing", use_container_width=True):
                    del st.session_state['processed_data']
                    if 'processing_completed' in st.session_state:
                        del st.session_state['processing_completed']
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
    st.markdown("Chat with our AI assistant about HR topics based on processed query knowledge base.")

    # Initialise RAG bot
    if 'rag_bot' not in st.session_state:
        if st.button("Initialise RAG System", type="primary"):
            try:
                with st.spinner("Setting up RAG system..."):
                    # Check if we have processed data
                    if 'processed_data' in st.session_state:
                        # Use session data
                        temp_file = "session_processed_data.csv"
                        st.session_state['processed_data'].to_csv(temp_file, index=False)

                        bot = HRQueryRAGBot()
                        bot.load_processed_data(temp_file)
                        bot.setup_topics(max_topics=5)  # Limit for demo

                        st.session_state['rag_bot'] = bot
                        st.session_state['rag_bot_ready'] = True
                        st.session_state['rag_bot_status'] = bot.get_setup_status()

                        # Clean up
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

                        st.success("RAG system initialised!")
                        st.rerun()
                    else:
                        st.warning("No processed data found. Please process queries in the Batch Classifier first.")
            except Exception as e:
                st.error(f"Error initialising RAG system: {str(e)}")
        return

    # RAG system is ready
    bot = st.session_state['rag_bot']
    ready_topics = bot.list_ready_topics()

    if not ready_topics:
        st.warning("No topics available. Please initialise the system first.")
        return

    # Topic selection and chat interface
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Topic Selection")
        selected_topic = st.selectbox(
            "Choose HR topic:",
            options=ready_topics,
            help="Select the HR topic you want to discuss"
        )

        # Show topic info
        if selected_topic:
            topic_info = bot.get_topic_info(selected_topic)
            st.info(f"{topic_info['total_queries']} queries available")

            if 'spp_distribution' in topic_info:
                st.markdown("**Categorisation:**")
                for spp, count in topic_info['spp_distribution'].items():
                    st.text(f"• {spp}: {count}")

        # Clear chat button
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.pop(f'chat_history_{selected_topic}', None)
            st.rerun()

    with col2:
        st.markdown(f"### Chat: {selected_topic}")

        # Initialise chat history for this topic
        chat_key = f'chat_history_{selected_topic}'
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, chat in enumerate(st.session_state[chat_key]):
                with st.chat_message("user"):
                    st.write(chat['question'])
                with st.chat_message("assistant"):
                    st.write(chat['answer'])
                    if chat.get('show_sources') and 'sources' in chat:
                        with st.expander(f"Sources ({len(chat['sources'])})"):
                            for j, source in enumerate(chat['sources'], 1):
                                st.text(f"{j}. {source['content'][:200]}...")

        # Chat input
        user_question = st.chat_input("Ask about this HR topic...")

        if user_question:
            # Add user message to chat
            st.session_state[chat_key].append({
                'question': user_question,
                'answer': 'Thinking...',
                'show_sources': False
            })

            # Get response from bot
            with st.spinner("Generating response..."):
                response = bot.chat_with_topic(selected_topic, user_question, show_sources=True)

            if 'error' not in response:
                # Update the last message with actual response
                st.session_state[chat_key][-1].update({
                    'answer': response['answer'],
                    'sources': response.get('sources', []),
                    'show_sources': True
                })
            else:
                st.session_state[chat_key][-1]['answer'] = f"{response['error']}"

            st.rerun()

def show_about_us():
    st.title("About Us")

    # What this system does
    st.markdown(
        """
    ## What We Built

    This is a practical tool for handling HR queries more efficiently. Instead of manually sorting through hundreds of employee questions about leave policies, salary matters, or system issues, this system does it automatically.

    The main idea is simple: take a bunch of HR tickets, let AI categorise them properly, then use those categorised tickets to build a smart chatbot that can answer similar questions.
    """
    )

    # The problem we solved
    st.markdown(
        """
    ## Why We Built This

    HR departments get swamped with repetitive questions. Someone asks about medical leave, another person asks the same thing next week. Staff spend time manually categorising tickets and giving similar answers over and over.

    Our system tackles this by:
    - Automatically sorting HR queries into 66 different categories (like "Medical Leave", "Salary Matters", "HRP System Issues")
    - Creating quick summaries so you don't have to read through long complaint emails
    - Building a chatbot that actually knows your organisation's specific HR situations
    """
    )

    # What you need to use it
    st.markdown(
        """
    ## What You Need

    **Data**: Export your HR tickets to a CSV file. Just needs one column with the actual query text.

    **API Access**: An OpenAI API key (this is where the AI processing happens).

    **That's it**: No complex setup, no training data preparation, no machine learning expertise required.
    """
    )

    # Main features
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        ### Batch Processor
        Upload your CSV file and watch it work:
        - Summarises each query in 2-3 sentences
        - Puts it in the right category 
        - Marks it as System/Process/Policy issue
        - Shows live progress as it works
        - Downloads results when done
        
        *(Code: `hr_classifier_py.py`)*
        """
        )

    with col2:
        st.markdown(
            """
        ### Smart Chatbot
        Chat with your processed data:
        - Pick an HR topic to focus on
        - Ask questions in plain English
        - Get answers based on your actual tickets
        - See which original queries informed each answer
        - Switch between different HR topics
        
        *(Code: `hr_rag_bot_py.py`)*
        """
        )

    # How it works technically
    st.markdown(
        """
    ## How It Works

    **Step 1**: The `process_queries_with_live_updates()` function sends each query to GPT-4o-mini with specific prompts asking it to summarise and categorise.

    **Step 2**: Results get stored and can be used to create a "vector database" - basically a smart search system that finds similar queries.

    **Step 3**: When you chat, the system finds relevant past queries and uses them as context for generating answers.

    **Built With**: Streamlit for the web interface, OpenAI for AI processing, ChromaDB for storing searchable data, and LangChain for connecting everything together.
    """
    )

    # Practical benefits
    st.markdown(
        """
    ## What You Get

    - **Save Time**: Stop manually categorising tickets
    - **Consistent Answers**: Chatbot responses based on actual organisational knowledge
    - **Pattern Recognition**: See what types of queries come up most often
    - **Self-Service**: Employees can get quick answers without waiting
    - **Audit Trail**: Every chatbot answer shows you which original tickets it came from
    """
    )

def show_methodology():
    st.title("Methodology")
    
    # Add Process Flow Chart section
    st.markdown("### System Process Flow")
    
    # Display the mermaid chart image with reduced size using column layout
    try:
        # Create columns to control size while maintaining quality
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("process_flow_mermaid_chart.png", 
                    caption="HR Query Analysis System - Process Flow")  # No width parameter - uses column width
    except FileNotFoundError:
        st.warning("Process flow chart image not found. Please ensure 'process_flow_mermaid_chart.png' is in the same directory as the app.")
    
    st.markdown("---")  # Add separator
    
    st.markdown(
        """
    ### AI Models and Techniques

    The system leverages two primary AI components:
    
    1.  **Batch Query Classifier**:
        - **Model**: Utilises OpenAI's GPT-4o-mini, a highly efficient and capable large language model.
        - **Method**: The model performs few-shot learning by being provided with clear instructions, a list of potential topics, and a definition of the classification types (System, Process, Policy). This allows for dynamic classification without a pre-trained model.
        - **Process**:
            - **Summarisation**: The model first generates a concise summary of the HR query.
            - **Topic Categorisation**: It then matches the query to one of the 66 predefined HR topics.
            - **SPP Categorisation**: Finally, it categorises the query as a System, Process, or Policy issue based on the provided definitions.
    
    2.  **RAG-Powered Chatbot**:
        - **Method**: The chatbot uses a **Retrieval-Augmented Generation (RAG)** framework. This method combines the power of a large language model with a custom knowledge base.
        - **Knowledge Base Creation**:
            - Processed queries from the Batch Classifier are used to create the knowledge base.
            - Each query and its summary/categorisations are embedded into a vector representation.
            - These vectors are stored in a **vector database (ChromaDB)**.
        - **Chat Interaction**:
            - When a user asks a question, the query is also converted into a vector.
            - The system performs a **semantic search** in the vector database to find the most relevant, similar HR queries and their responses.
            - These retrieved "source documents" are provided as context to the language model (GPT-4o-mini).
            - The model then generates a coherent response based on the new user question and the retrieved context. This ensures that the answers are grounded in the specific data from the organisation's HR queries, reducing hallucinations and providing accurate information.
            - Source references are provided to allow users to see the original queries that informed the bot's response.
    
    ### System Workflow
    1.  **User Upload**: A user uploads a CSV file containing raw HR queries.
    2.  **Batch Processing**: The system iterates through each query, calling the AI model for summarisation and classification.
    3.  **Data Persistence**: The processed data (with new columns for summary, topic, and SPP) is stored in the `st.session_state` and can be downloaded.
    4.  **RAG Initialisation**: When the user navigates to the Chatbot, the processed data is used to build the knowledge base and set up the RAG pipeline.
    5.  **Interactive Chat**: The user can select a topic and chat with the bot, which retrieves relevant information from the knowledge base to answer questions.
    
    ### Technical Implementation Details
    
    **Batch Processing Pipeline:**
    - Each query undergoes three sequential AI calls to GPT-4o-mini
    - Live progress tracking with real-time updates
    - Error handling and recovery for failed queries
    - Configurable processing limits to manage API costs
    
    **RAG Architecture:**
    - Vector embeddings using OpenAI's text-embedding-3-small model
    - ChromaDB for efficient similarity search
    - LangChain framework for RAG pipeline orchestration
    - Topic-based knowledge base segmentation for focused responses
    
    **User Experience Features:**
    - Password-protected access for security
    - Session state management for data persistence
    - Real-time processing feedback
    - Downloadable results in CSV format
    - Interactive chat interface with source attribution
    """
    )

# Main app logic
if check_password():
    current_page = sidebar_navigation()

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