# hr_rag_bot.py

import pandas as pd
import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

class HRQueryRAGBot:
    """RAG chatbot for HR queries utilising processed summaries as knowledge base"""
    
    def __init__(self, model_name: str = 'gpt-4o-mini'):
        """Initialise the RAG bot"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize with the API key explicitly
        self.embeddings_model = OpenAIEmbeddings(
            model='text-embedding-3-small',
            openai_api_key=api_key
        )
        self.llm = ChatOpenAI(
            model=model_name, 
            temperature=0, 
            seed=42,
            openai_api_key=api_key
        )
        
        self.vector_databases = {}
        self.available_topics = []
        self.processed_data = None
        
        print(f"HR Query RAG Bot initialised with {model_name}")
    
    def load_processed_data(self, csv_file_path: str) -> pd.DataFrame:
        """Load processed data from classifier"""
        print(f"Loading processed HR data from: {csv_file_path}")
        
        try:
            df = pd.read_csv(csv_file_path)
            
            required_columns = ['summary', 'topic_classification']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Filter out empty rows
            original_len = len(df)
            df = df.dropna(subset=['summary', 'topic_classification'])
            df = df[df['summary'].str.strip() != '']
            df = df[df['topic_classification'].str.strip() != '']
            
            self.processed_data = df
            self.available_topics = sorted(df['topic_classification'].unique().tolist())
            
            print(f"Loaded {len(df)} processed queries")
            print(f"Available topics: {len(self.available_topics)}")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def create_topic_vector_database(self, topic: str, chunk_size: int = 1100, chunk_overlap: int = 10) -> None:
        """Create vector database for specific topic utilising summaries"""
        print(f"Creating vector database for topic: '{topic}'...")
        
        if self.processed_data is None:
            raise ValueError("No processed data loaded. Call load_processed_data() first.")
        
        topic_data = self.processed_data[self.processed_data['topic_classification'] == topic].copy()
        
        if len(topic_data) == 0:
            print(f"No data found for topic: '{topic}'")
            return
        
        # Create documents from summaries
        documents = []
        for idx, row in topic_data.iterrows():
            metadata = {
                'topic': topic,
                'original_query_index': idx,
                'spp_classification': row.get('spp_classification', 'Unknown'),
                'processing_status': row.get('processing_status', 'Unknown'),
                'source': f"HR_Query_{idx}"
            }
            
            content = f"Summary: {row['summary']}"
            
            # Add original query context if available
            query_col = None
            for col in ['Case Description', 'query', 'question', 'description']:
                if col in row and pd.notna(row[col]):
                    query_col = col
                    break
            
            if query_col:
                original_query = str(row[query_col])
                if len(original_query) > 200:
                    original_query = original_query[:200] + "..."
                content += f"\n\nOriginal Query Context: {original_query}"
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        print(f"Created {len(documents)} documents for topic '{topic}'")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            length_function=count_tokens
        )
        
        splitted_documents = text_splitter.split_documents(documents)
        print(f"Split into {len(splitted_documents)} chunks")
        
        # Create collection name
        collection_name = f"hr_{topic.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}"[:50]
        collection_name = ''.join(c for c in collection_name if c.isalnum() or c == '_')
        
        try:
            vectordb = Chroma.from_documents(
                documents=splitted_documents,
                embedding=self.embeddings_model,
                collection_name=collection_name,
                persist_directory=f"./vector_db_{collection_name}"
            )
            
            self.vector_databases[topic] = vectordb
            print(f"Vector database created for '{topic}'")
            
        except Exception as e:
            print(f"Error creating vector database for '{topic}': {str(e)}")
            raise
    
    def setup_topics(self, selected_topics: Optional[List[str]] = None, max_topics: int = 5) -> None:
        """Create vector databases for selected topics"""
        if self.processed_data is None:
            raise ValueError("No processed data loaded. Call load_processed_data() first.")
        
        if selected_topics is None:
            topic_counts = self.processed_data['topic_classification'].value_counts()
            selected_topics = topic_counts.head(max_topics).index.tolist()
            print(f"Auto-selected top {max_topics} topics with most data:")
            for topic in selected_topics:
                print(f"   {topic} ({topic_counts[topic]} queries)")
        
        print(f"\nSetting up vector databases for {len(selected_topics)} topics...")
        
        successful_setups = 0
        for i, topic in enumerate(selected_topics, 1):
            print(f"\n[{i}/{len(selected_topics)}] Processing topic: {topic}")
            try:
                self.create_topic_vector_database(topic)
                successful_setups += 1
            except Exception as e:
                print(f"Failed to setup topic '{topic}': {str(e)}")
                continue
        
        print(f"\nSetup complete! {successful_setups}/{len(selected_topics)} topic databases ready.")
    
    def create_rag_chain(self, topic: str, custom_prompt: Optional[str] = None) -> RetrievalQA:
        """Create RAG chain for specific topic"""
        if topic not in self.vector_databases:
            raise ValueError(f"No vector database found for topic '{topic}'. Run setup_topics() first.")
        
        if custom_prompt is None:
            custom_prompt = f"""You are an expert HR assistant specialising in "{topic}". 

Use the provided HR knowledge base to answer questions accurately and professionally.

Guidelines:
- Provide clear, helpful answers based on the HR knowledge base context
- If the answer isn't in the knowledge base, say "I don't have specific information about this in my knowledge base"
- Keep answers concise but comprehensive (2-4 sentences typically)
- Reference HR policies, processes, or systems when mentioned in the context
- Maintain a professional, helpful tone suitable for HR inquiries

HR Knowledge Base Context:
{{context}}

Question: {{question}}

HR Assistant Response:"""
        
        QA_CHAIN_PROMPT = PromptTemplate.from_template(custom_prompt)
        
        rag_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.vector_databases[topic].as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        
        return rag_chain
    
    def chat_with_topic(self, topic: str, question: str, show_sources: bool = False) -> Dict[str, Any]:
        """Chat with bot about specific HR topic"""
        if topic not in self.vector_databases:
            available_topics = list(self.vector_databases.keys())
            return {
                'error': f"Topic '{topic}' not available. Available topics: {available_topics}"
            }
        
        try:
            rag_chain = self.create_rag_chain(topic)
            response = rag_chain.invoke(question)
            
            result = {
                'topic': topic,
                'question': question,
                'answer': response['result'],
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            if show_sources and 'source_documents' in response:
                result['sources'] = []
                for i, doc in enumerate(response['source_documents']):
                    source_info = {
                        'source_number': i + 1,
                        'content': doc.page_content[:300] + '...' if len(doc.page_content) > 300 else doc.page_content,
                        'metadata': doc.metadata
                    }
                    result['sources'].append(source_info)
                result['num_sources'] = len(result['sources'])
            
            return result
            
        except Exception as e:
            return {
                'error': f"Error processing question: {str(e)}",
                'topic': topic,
                'question': question
            }
    
    def get_topic_info(self, topic: str) -> Dict[str, Any]:
        """Get info about topic's knowledge base"""
        if self.processed_data is None:
            return {'error': 'No processed data loaded'}
        
        topic_data = self.processed_data[self.processed_data['topic_classification'] == topic]
        
        info = {
            'topic': topic,
            'total_queries': len(topic_data),
            'has_vector_db': topic in self.vector_databases
        }
        
        if len(topic_data) > 0:
            if 'spp_classification' in topic_data.columns:
                spp_dist = topic_data['spp_classification'].value_counts().to_dict()
                info['spp_distribution'] = spp_dist
            
            info['sample_summaries'] = topic_data['summary'].head(3).tolist()
        
        return info
    
    def list_available_topics(self) -> List[str]:
        """Return list of available topics"""
        return self.available_topics.copy()
    
    def list_ready_topics(self) -> List[str]:
        """Return list of topics ready for chatting"""
        return list(self.vector_databases.keys())
    
    def get_setup_status(self) -> Dict[str, Any]:
        """Get current setup status"""
        return {
            'data_loaded': self.processed_data is not None,
            'total_queries': len(self.processed_data) if self.processed_data is not None else 0,
            'total_topics_available': len(self.available_topics),
            'topics_with_vector_db': len(self.vector_databases),
            'ready_topics': list(self.vector_databases.keys()),
            'pending_topics': [t for t in self.available_topics if t not in self.vector_databases]
        }