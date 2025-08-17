import pandas as pd
import time
import os
import streamlit as st
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

load_dotenv()

def get_completion_by_messages(messages, model="gpt-4o-mini", temperature=0, top_p=1.0, max_tokens=1024, n=1):
    """Get completion from OpenAI with proper error handling"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    try:
        # Create client with minimal parameters to avoid compatibility issues
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=1
        )
        return response.choices[0].message.content
        
    except Exception as e:
        # Provide more specific error information
        error_msg = str(e)
        if "proxies" in error_msg:
            raise ValueError(f"OpenAI client initialization error (possible version mismatch): {error_msg}")
        else:
            raise ValueError(f"OpenAI API call failed: {error_msg}")

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

def process_hr_queries(csv_file_path: str, query_column_name: str, 
                      topic_categories: List[str], 
                      spp_definitions: str) -> pd.DataFrame:
    """Main function that processes all HR queries from CSV file"""
    print("Loading CSV file...")
    df = pd.read_csv(csv_file_path)
    
    if query_column_name not in df.columns:
        raise ValueError(f"Column '{query_column_name}' not found in CSV. Available columns: {list(df.columns)}")
    
    print(f"Found {len(df)} queries to process")
    
    df['summary'] = ''
    df['topic_classification'] = ''
    df['spp_classification'] = ''
    df['processing_status'] = ''
    
    for index, row in df.iterrows():
        query = row[query_column_name]
        
        if pd.isna(query) or str(query).strip() == '':
            df.at[index, 'processing_status'] = 'SKIPPED - Empty query'
            continue
        
        try:
            print(f"Processing query {index + 1}/{len(df)}")
            
            print("  - Generating summary...")
            summary = summarise_hr_query(query)
            df.at[index, 'summary'] = summary
            time.sleep(0.2)
            
            print("  - Categorising topic...")
            topic = classify_hr_topic(query, topic_categories)
            df.at[index, 'topic_classification'] = topic
            time.sleep(0.2)
            
            print("  - Categorising System/Process/Policy...")
            spp_class = classify_system_process_policy(query, spp_definitions)
            df.at[index, 'spp_classification'] = spp_class
            
            df.at[index, 'processing_status'] = 'SUCCESS'
            time.sleep(0.3)
            
        except Exception as e:
            print(f"Error processing query {index + 1}: {str(e)}")
            df.at[index, 'processing_status'] = f'ERROR: {str(e)}'
            continue
    
    return df

def get_processing_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Creates summary report of processing results"""
    total_queries = len(df)
    successful = len(df[df['processing_status'] == 'SUCCESS'])
    errors = len(df[df['processing_status'].str.contains('ERROR', na=False)])
    skipped = len(df[df['processing_status'].str.contains('SKIPPED', na=False)])
    
    topic_counts = df['topic_classification'].value_counts().to_dict()
    spp_counts = df['spp_classification'].value_counts().to_dict()
    
    return {
        'total_queries': total_queries,
        'successful': successful,
        'errors': errors,
        'skipped': skipped,
        'success_rate': (successful / total_queries * 100) if total_queries > 0 else 0,
        'topic_distribution': topic_counts,
        'spp_distribution': spp_counts
    }