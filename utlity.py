"""
Utility functions for the Customer Support Copilot
"""
import json
import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Optional
import re
from datetime import datetime

def parse_uploaded_file(uploaded_file) -> List[str]:
    """Parse uploaded file and extract tickets"""
    tickets = []
    
    try:
        if uploaded_file.type == "application/json":
            data = json.load(uploaded_file)
            
            if isinstance(data, list):
                # Handle different JSON structures
                for item in data:
                    if isinstance(item, dict):
                        # Try common field names for tickets
                        ticket_text = (item.get('ticket') or 
                                     item.get('message') or 
                                     item.get('description') or 
                                     item.get('text') or 
                                     str(item))
                        tickets.append(ticket_text)
                    else:
                        tickets.append(str(item))
            else:
                tickets.append(str(data))
                
        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            
            # Try to find the ticket column
            possible_columns = ['ticket', 'message', 'description', 'text', 'issue']
            ticket_column = None
            
            for col in possible_columns:
                if col in df.columns.str.lower():
                    ticket_column = df.columns[df.columns.str.lower() == col][0]
                    break
            
            if ticket_column:
                tickets = df[ticket_column].dropna().tolist()
            else:
                # Use first column as fallback
                tickets = df.iloc[:, 0].dropna().tolist()
                
        else:
            # Text file - split by lines
            content = uploaded_file.read().decode('utf-8')
            tickets = [line.strip() for line in content.split('\n') if line.strip()]
    
    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
        return []
    
    return [str(ticket) for ticket in tickets if ticket and str(ticket).strip()]

def validate_classification(classification: Dict) -> Dict:
    """Validate and sanitize classification results"""
    from config import Config
    
    validated = {}
    
    # Validate topic
    topic = classification.get('topic', 'Unknown')
    if topic not in Config.TOPIC_CATEGORIES:
        # Try to match partial strings
        for valid_topic in Config.TOPIC_CATEGORIES:
            if valid_topic.lower() in topic.lower() or topic.lower() in valid_topic.lower():
                topic = valid_topic
                break
        else:
            topic = 'Product'  # Default fallback
    validated['topic'] = topic
    
    # Validate sentiment
    sentiment = classification.get('sentiment', 'Neutral')
    if sentiment not in Config.SENTIMENT_CATEGORIES:
        sentiment = 'Neutral'  # Default fallback
    validated['sentiment'] = sentiment
    
    # Validate priority
    priority = classification.get('priority', 'P1')
    if priority not in Config.PRIORITY_LEVELS:
        priority = 'P1'  # Default fallback
    validated['priority'] = priority
    
    return validated

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.,!?;:()\-\'"]', '', text)
    
    return text

def format_sources(sources: List[str]) -> str:
    """Format source URLs for display"""
    if not sources:
        return "No sources available"
    
    formatted = []
    for i, source in enumerate(sources, 1):
        if source and source != "Unknown":
            formatted.append(f"{i}. [{source}]({source})")
        else:
            formatted.append(f"{i}. Internal Knowledge Base")
    
    return "\n".join(formatted)

def get_priority_color(priority: str) -> str:
    """Get color for priority display"""
    colors = {
        'P0': '🔴',
        'P1': '🟡', 
        'P2': '🟢'
    }
    return colors.get(priority, '⚪')

def get_sentiment_emoji(sentiment: str) -> str:
    """Get emoji for sentiment display"""
    emojis = {
        'Happy': '😊',
        'Curious': '🤔', 
        'Neutral': '😐',
        'Frustrated': '😤',
        'Angry': '😠',
        'Confused': '😕'
    }
    return emojis.get(sentiment, '😐')

def export_results_to_csv(results: List[Dict]) -> str:
    """Convert results to CSV format"""
    if not results:
        return ""
    
    df = pd.DataFrame(results)
    return df.to_csv(index=False)

def create_sample_tickets() -> List[str]:
    """Create sample tickets for testing"""
    return [
        "How do I set up a new Snowflake connector in Atlan? I need step-by-step instructions.",
        "I'm getting authentication errors when trying to connect to our database. This is urgent!",
        "Can you explain what data lineage means and how it works in Atlan?",
        "Our SSO login stopped working after the recent update. Users can't access the platform.",
        "I need help with the Python SDK. How do I query assets programmatically?",
        "What are the best practices for organizing our data catalog?",
        "I can't find the glossary feature. Where is it located in the interface?",
        "We have sensitive data that needs to be properly classified. How do we set this up?",
        "The connector is running but not extracting metadata. What could be wrong?",
        "I love the new dashboard! It makes finding assets so much easier."
    ]

def log_interaction(ticket: str, classification: Dict, response: str):
    """Log interaction for monitoring (placeholder)"""
    # In production, this would log to a proper monitoring system
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "ticket": ticket[:100] + "..." if len(ticket) > 100 else ticket,
        "classification": classification,
        "response_length": len(response)
    }
    
    # For demo purposes, store in session state
    if 'interaction_logs' not in st.session_state:
        st.session_state.interaction_logs = []
    
    st.session_state.interaction_logs.append(log_entry)