import streamlit as st
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional
import asyncio
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time
import base64

# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import Document

# LangSmith for observability
from langsmith import Client
import langsmith

import os
from dotenv import load_dotenv

load_dotenv()

default_groq_key = os.getenv("GROQ_API_KEY", "")

# Initializing LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "customer-support-copilot"

def get_base64_of_image(path):
    """Convert image to base64 string"""
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
class TicketClassifier:
    def __init__(self, groq_api_key: str):
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile", 
            temperature=0.1
        )
        
        self.classification_prompt = PromptTemplate(
            input_variables=["ticket_text"],
            template="""
            Analyze the following customer support ticket and classify it:

            Ticket: {ticket_text}

            Provide the classification in the following JSON format:
            {{
                "topic": "one of [How-to, Product, Connector, Lineage, API/SDK, SSO, Glossary, Best practices, Sensitive data]",
                "sentiment": "one of [Frustrated, Curious, Angry, Neutral, Happy, Confused]",
                "priority": "one of [P0, P1, P2]"
            }}

            Classification Guidelines:
            - Topic: Choose the most relevant category based on the ticket content
            - Sentiment: Analyze the emotional tone of the customer
            - Priority: P0 (urgent/critical), P1 (medium), P2 (low priority)
            """
        )
        
        self.parser = JsonOutputParser()
    
    def classify_ticket(self, ticket_text: str) -> Dict:
        """Classify a single ticket"""
        try:
            chain = self.classification_prompt | self.llm | self.parser
            result = chain.invoke({"ticket_text": ticket_text})
            return result
        except Exception as e:
            st.error(f"Classification error: {str(e)}")
            return {"topic": "Unknown", "sentiment": "Neutral", "priority": "P1"}


class AtlanDocLoader:
    """For loading Atlan documentation"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def load_from_url(self, url: str, max_pages: int = 20) -> List[Document]:
        """Loading content from Atlan documentation URL"""
        documents = []
        visited_urls = set()
        urls_to_visit = [url]
        
        for _ in range(max_pages):
            if not urls_to_visit:
                break
                
            current_url = urls_to_visit.pop(0)
            if current_url in visited_urls:
                continue
                
            try:
                response = self.session.get(current_url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract main content
                content = self._extract_content(soup)
                
                if content and len(content.strip()) > 100:  # Only add relevant and important content
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": current_url,
                            "title": self._extract_title(soup)
                        }
                    ))
                
                # Find additional links within the same domain
                additional_links = self._extract_internal_links(soup, current_url)
                urls_to_visit.extend([link for link in additional_links 
                                    if link not in visited_urls and len(visited_urls) < max_pages])
                
                visited_urls.add(current_url)
                time.sleep(0.5)  
                
            except Exception as e:
                st.warning(f"Failed to load {current_url}: {str(e)}")
                continue
        
        return documents
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from the page"""
        # Remove script, style, and navigation elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'main',
            '[role="main"]',
            '.content',
            '.main-content',
            '.documentation',
            '.docs-content',
            'article',
            '.article-content'
        ]
        
        content = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    content += element.get_text(separator='\n', strip=True) + '\n'
                break
        
        # Fallback to body if no specific content area found
        if not content:
            body = soup.find('body')
            if body:
                content = body.get_text(separator='\n', strip=True)
        
        # Clean up the content
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text(strip=True)
        
        return "Atlan Documentation"
    
    def _extract_internal_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extracting internal and hidden links from the same domain"""
        links = []
        base_domain = self._get_domain(base_url)
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Convert relative URLs to absolute
            if href.startswith('/'):
                full_url = f"{base_domain.rstrip('/')}{href}"
            elif href.startswith('http'):
                full_url = href
            else:
                continue
            
            # Only include links from the same domain
            if self._get_domain(full_url) == base_domain:
                links.append(full_url)
        
        return list(set(links))  # Removing duplicates
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"



class KnowledgeBase:
    def __init__(self, groq_api_key: str):
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.3
        )
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        self.vector_store = None
        self.retriever = None
        self.doc_loader = AtlanDocLoader()
        
    def load_knowledge_base(self):
        """Load and process knowledge base from Atlan docs"""
        if self.vector_store is not None:
            return  # Already loaded
            
        with st.spinner("Loading knowledge base from Atlan documentation... This may take a few minutes."):
            try:
                # Atlan documentation URLs
                urls = [
                    "https://docs.atlan.com/",
                    "https://developer.atlan.com/",
                ]
                
                all_documents = []

                for url in urls:
                    st.info(f"Fetching content from {url}...")
                    documents = self.doc_loader.load_from_url(url, max_pages=40)  # Limit pages per URL
                    all_documents.extend(documents)
                    st.success(f"Loaded {len(documents)} pages from {url}")

                if not all_documents:
                    st.error("No documents were loaded. Please check the URLs and your internet connection.")
                    return
                
                st.info(f"Total documents loaded: {len(all_documents)}")
                st.info("Processing and chunking documents...")
                
                # Split documents
                splits = self.text_splitter.split_documents(all_documents)
                st.info(f"Created {len(splits)} chunks for vector store")
                
                # Create vector store
                st.info("Creating vector embeddings... This may take a moment.")
                self.vector_store = FAISS.from_documents(splits, self.embeddings)
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 15}  # Increase to get more context
                )
                
                st.success("Knowledge base loaded successfully from live Atlan documentation!") 

            except Exception as e:
                st.error(f"Error loading knowledge base: {str(e)}")
                st.info("This might be due to network issues or website access restrictions. Please try again later.")
    
    def get_rag_answer(self, question: str) -> Tuple[str, List[str]]:
        """Get RAG-based answer with source citations"""
        if not self.retriever:
            return "Knowledge base not loaded. Please reload the application.", []
        
        try:
            # Retrieve relevant documents
            docs = self.retriever.get_relevant_documents(question)
            
            if not docs:
                return "I couldn't find relevant information in the Atlan documentation to answer your question.", []
            
            # Extract sources
            sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
            
            # Create context from retrieved docs
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # RAG prompt
            rag_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                You are a helpful customer support assistant for Atlan. Use ONLY the following context from Atlan's official documentation to answer the customer's question. 
                
                If the context doesn't contain enough information to provide a complete answer, clearly state what information is missing and what you can tell them based on the available documentation.
                
                Context from Atlan Documentation:
                {context}
                
                Customer Question: {question}
                
                Answer: Provide a clear, helpful response based ONLY on the information in the context above. Include specific details, steps, or examples when available in the documentation. If you cannot find the answer in the provided context, say so clearly.
                """
            )
            
            # Generate answer
            chain = rag_prompt | self.llm
            answer = chain.invoke({"context": context, "question": question})
            
            return answer.content, sources
             
        except Exception as e:
            return f"Error generating answer: {str(e)}", []

class SupportCopilot:
    def __init__(self, groq_api_key: str):
        self.classifier = TicketClassifier(groq_api_key)
        self.knowledge_base = KnowledgeBase(groq_api_key)
        
        # Topics that should use RAG
        self.rag_topics = {"How-to", "Product", "Best practices", "API/SDK", "SSO"}
        
    def process_ticket(self, ticket_text: str) -> Tuple[Dict, str, List[str]]:
        """Process a single ticket and return classification and response"""
        
        # Classify the ticket
        classification = self.classifier.classify_ticket(ticket_text)
        
        # Determine response based on topic
        topic = classification.get("topic", "Unknown")
        
        if topic in self.rag_topics:
            # Use RAG for answer
            answer, sources = self.knowledge_base.get_rag_answer(ticket_text)
            return classification, answer, sources
        else:
            # Route to appropriate team
            routing_message = f"This ticket has been classified as a '{topic}' issue and routed to the appropriate team. You will receive a response within 24 hours."
            return classification, routing_message, []

def main():
    st.set_page_config(
        page_title="Customer Support Copilot",
        page_icon="atlan_icon.png",
        layout="wide"
    )
    
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{get_base64_of_image(r'atlan_icon.png')}" 
                 style="width: 40px; height: 40px; margin-right: 15px;">
            <h1 style="margin: 0; font-size: 3rem;">Customer Support Copilot</h1>
        </div>
        """, 
        unsafe_allow_html=True
    )
    #For configuration
    with st.sidebar:
        st.header("Configuration")
        groq_api_key = st.text_input(
            label="Groq API Key",
            value=default_groq_key,
            type="password",
            help="Enter your Groq API key (will auto-load from .env if available)"
        )
        
        if groq_api_key:
            st.success("✅ API key configured")
        else:
            st.warning("⚠️ Please enter your Groq API key")

        st.markdown("---")
        st.markdown("### Knowledge Base Status")
        if 'knowledge_loaded' in st.session_state and st.session_state.knowledge_loaded:
            st.success("✅ Atlan docs loaded")
        else:
            st.warning("⏳ Knowledge base loading...")

    if not groq_api_key:
        st.error("Please provide a Groq API key to use the application.")
        st.stop()


    
    # Initialize the support copilot
    if 'copilot' not in st.session_state:
        with st.spinner("Initializing AI copilot..."):
            st.session_state.copilot = SupportCopilot(groq_api_key)
            st.session_state.copilot.knowledge_base.load_knowledge_base()
            st.session_state.knowledge_loaded = True
    
    # Main interface
    tab1, tab2 = st.tabs(["📊 Bulk Ticket Analysis", "💬 Interactive Agent"])
    
    with tab1:
        st.header("Bulk Ticket Classification Dashboard")
        
        uploaded_file = st.file_uploader(
            "Upload support tickets file",
            type=['csv', 'json', 'txt'],
            help="Upload a file containing customer support tickets"
        )
        
        if uploaded_file is not None:
            try:
                # Parse uploaded file
                if uploaded_file.type == "application/json":
                    data = json.load(uploaded_file)
                    if isinstance(data, list):
                        tickets = [item.get('ticket', str(item)) for item in data]
                    else:
                        tickets = [str(data)]
                elif uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                    # Try to find the ticket column
                    possible_columns = ['ticket', 'message', 'description', 'text', 'issue']
                    ticket_column = None
                    
                    for col in possible_columns:
                        matching_cols = [c for c in df.columns if col.lower() in c.lower()]
                        if matching_cols:
                            ticket_column = matching_cols[0]
                            break
                    
                    if ticket_column:
                        tickets = df[ticket_column].dropna().tolist()
                    else:
                        tickets = df.iloc[:, 0].dropna().tolist()
                else:
                    # Text file
                    content = uploaded_file.read().decode('utf-8')
                    tickets = [line.strip() for line in content.split('\n') if line.strip()]
                
                st.success(f"Loaded {len(tickets)} tickets")
                
                if st.button("🔍 Analyze All Tickets"):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for i, ticket in enumerate(tickets):
                        if ticket.strip():  # Skip empty tickets
                            classification, _, _ = st.session_state.copilot.process_ticket(ticket)
                            results.append({
                                'Ticket': ticket[:100] + "..." if len(ticket) > 100 else ticket,
                                'Topic': classification.get('topic', 'Unknown'),
                                'Sentiment': classification.get('sentiment', 'Neutral'),
                                'Priority': classification.get('priority', 'P1')
                            })
                        progress_bar.progress((i + 1) / len(tickets))
                    
                    # Display results
                    if results:
                        results_df = pd.DataFrame(results)
                        st.subheader("Classification Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.subheader("Topics")
                            topic_counts = results_df['Topic'].value_counts()
                            st.bar_chart(topic_counts)
                        
                        with col2:
                            st.subheader("Sentiment")
                            sentiment_counts = results_df['Sentiment'].value_counts()
                            st.bar_chart(sentiment_counts)
                        
                        with col3:
                            st.subheader("Priority")
                            priority_counts = results_df['Priority'].value_counts()
                            st.bar_chart(priority_counts)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"ticket_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab2:
        st.header("Interactive AI Agent")
        st.markdown("Submit a support ticket and get instant classification and response")
        
        # Ticket input
        ticket_input = st.text_area(
            "Enter your support ticket:",
            placeholder="Describe your issue or question about Atlan...",
            height=150
        )
        
        if st.button("🚀 Submit Ticket") and ticket_input:
            with st.spinner("Processing your ticket with live Atlan documentation..."):
                classification, response, sources = st.session_state.copilot.process_ticket(ticket_input)
            
            # Display results in two columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🔍 Internal Analysis")
                
                # Classification results
                st.json(classification)
                
                # Visual indicators
                topic = classification.get('topic', 'Unknown')
                sentiment = classification.get('sentiment', 'Neutral')
                priority = classification.get('priority', 'P1')
                channel = classification.get('channel', 'Email')
                
                # Priority color coding
                priority_colors = {'P0': '🔴', 'P1': '🟡', 'P2': '🟢'}
                st.metric("Priority", f"{priority_colors.get(priority, '⚪')} {priority}")
                
                # Sentiment emoji
                sentiment_emojis = {
                    'Happy': '😊', 'Curious': '🤔', 'Neutral': '😐', 
                    'Frustrated': '😤', 'Angry': '😠', 'Confused': '😕'
                }
                st.metric("Sentiment", f"{sentiment_emojis.get(sentiment, '😐')} {sentiment}")

                # Channel icons
                channel_icons = {
                    'Email': '📧', 'Chat': '💬', 'Phone': '📞', 
                    'Social Media': '🌐', 'Forum': '💭', 'In-app': '📱', 'API': '🔌'
                }
                st.metric("Channel", f"{channel_icons.get(channel, '📧')} {channel}")
            
            with col2:
                st.subheader("💬 Response")
                
                st.write(response)
                
                # Show sources if available
                if sources:
                    st.subheader("📚 Sources")
                    for source in sources:
                        st.markdown(f"- [{source}]({source})")
                
                # Show routing info for non-RAG topics
                if topic not in st.session_state.copilot.rag_topics:
                    st.info(f"🔄 Ticket routed to: {topic} team")

if __name__ == "__main__":
    main()
