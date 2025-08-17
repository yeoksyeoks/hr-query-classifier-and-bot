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
    """Show API key configuration with form to prevent reruns"""
    st.markdown("### 🔑 API Configuration")
    
    # Show current API key status
    current_key = os.getenv('OPENAI_API_KEY')
    if current_key and len(current_key) > 10:
        st.success(f"✅ API Key: ...{current_key[-4:]}")
        api_configured = True
    else:
        st.error("❌ No API Key Found")
        api_configured = False
    
    # API key input form
    with st.form("api_key_form", clear_on_submit=False):
        st.markdown("**Configure OpenAI API Key**")
        api_key_input = st.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            help="Your OpenAI API key (starts with sk-...