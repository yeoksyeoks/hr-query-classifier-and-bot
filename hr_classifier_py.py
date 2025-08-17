import pandas as pd
import time
import os
import streamlit as st
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

load_dotenv()

def get_openai_client():
    """Get OpenAI client with current API key"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)

def get_completion_by_messages(messages, model="gpt-4o-mini", temperature=0, top_p=1.0, max_tokens=1024, n=1):
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1
    )
    return response.choices[0].message.content

def count_tokens(text):
    encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    return len(encoding.encode(text))

def summarise_hr_query(user_query):
    """Creates a short summary of HR query using LLM"""
    system_message = """
    You will be provided with HR queries enclosed in <hr-query> tags.
    
    Your task is to create a concise summary of the query in 2-3 sentences maximum.
    Focus on the main issue or question being asked.
    
    Provide only the summary without any additional formatting or tags.
    """
    
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"<hr-query>{user_query}</hr-query>"}
    ]
    
    summary_response = get_completion_by_messages(messages)
    return summary_response.strip()

def classify_hr_topic(user_query, topic_categories):
    """Classifies HR query into predefined topic categories"""
    numbered_topics = []
    for i, topic in enumerate(topic_categories, 1):
        numbered_topics.append(f"{i}. {topic}")
    
    topics_formatted = "\n".join(numbered_topics)
    
    system_message = f"""
    You will be provided with HR queries enclosed in <hr-query> tags.
    
    Classify the query into the most relevant category from the numbered list below:
    
    {topics_formatted}
    
    Instructions:
    - Choose only ONE category that best matches the query
    - Look for keywords and context that align with the category descriptions
    - If the query mentions specific systems like HRP, match to the appropriate HRP category
    - If no category fits well, choose "Other HR Matters"
    - Be precise with your classification based on the main subject of the query
    
    Respond with only the exact category name as written in the list above (without the number).
    """
    
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"<hr-query>{user_query}</hr-query>"}
    ]
    
    topic_response = get_completion_by_messages(messages, temperature=0.1)
    return topic_response.strip()

def classify_system_process_policy(user_query, spp_definitions):
    """Classifies HR query as System, Process, or Policy"""
    system_message = f"""
    You will be provided with HR queries enclosed in <hr-query> tags.
    
    Classify each query as either "System", "Process", or "Policy" based on the definitions provided below enclosed in triple backticks:
    
    ```
    {spp_definitions}
    ```
    
    Choose the classification that best matches the nature of the query.
    
    Respond with only one word: either "System", "Process", or "Policy".
    """
    
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"<hr-query>{user_query}</hr-query>"}
    ]
    
    spp_response = get_completion_by_messages(messages)
    return spp_response.strip()

def process_queries_with_live_updates(df, query_column, total_queries, progress_container, status_container, topic_categories, spp_definitions):
    """
    Processes queries from a DataFrame with live progress updates using Streamlit.
    """
    total_queries = len(df)
    processed_count = 0
    results = []
    start_time = time.time()
    
    with progress_container:
        progress_bar = st.progress(0, text="Processing queries...")
    with status_container:
        status_text = st.markdown("Starting processing...")

    for index, row in df.iterrows():
        try:
            query = row[query_column]
            
            if pd.isna(query) or str(query).strip() == '':
                status = 'SKIPPED - Empty query'
                topic = None
                spp_class = None
                summary = None
            else:
                topic = classify_hr_topic(query, topic_categories)
                spp_class = classify_system_process_policy(query, spp_definitions)
                summary = summarise_hr_query(query)
                status = "SUCCESS"
            
            results.append({
                'Original Query': query,
                'Topic': topic,
                'SPP Class': spp_class,
                'Summary': summary,
                'processing_status': status
            })
            
        except Exception as e:
            results.append({
                'Original Query': row[query_column],
                'Topic': "Error",
                'SPP Class': "Error",
                'Summary': f"Error during processing: {str(e)}",
                'processing_status': "ERROR"
            })
            
        processed_count += 1
        
        progress_percent = processed_count / total_queries
        progress_bar.progress(progress_percent, text=f"Processing query {processed_count} of {total_queries}...")
        
        status_text.markdown(f"**Last Query:** {row[query_column][:50]}...")
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    progress_bar.progress(1.0, text="Processing Complete!")
    status_text.markdown(f"**Processing complete!** Processed {total_queries} queries in {elapsed_time:.2f} seconds.")
    
    return pd.DataFrame(results)

def get_processing_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Creates summary report of processing results"""
    total_queries = len(df)
    successful = len(df[df['processing_status'] == 'SUCCESS'])
    errors = len(df[df['processing_status'].str.contains('ERROR', na=False)])
    skipped = len(df[df['processing_status'].str.contains('SKIPPED', na=False)])
    
    topic_counts = df['Topic'].value_counts().to_dict()
    spp_counts = df['SPP Class'].value_counts().to_dict()
    
    return {
        'total_queries': total_queries,
        'successful': successful,
        'errors': errors,
        'skipped': skipped,
        'success_rate': (successful / total_queries * 100) if total_queries > 0 else 0,
        'topic_distribution': topic_counts,
        'spp_distribution': spp_counts
    }
