"""
Configuration module for Customer Support Copilot
"""
import os
from typing import List, Dict

class Config:
    GROQ_MODEL = "llama-3.3-70b-versatile"  
    GROQ_TEMPERATURE = 0.1
    
    # Categories
    TOPIC_CATEGORIES = [
        "How-to", "Product", "Connector", "Lineage", 
        "API/SDK", "SSO", "Glossary", "Best practices", 
        "Sensitive data"
    ]
    
    SENTIMENT_CATEGORIES = [
        "Frustrated", "Curious", "Angry", "Neutral", 
        "Happy", "Confused"
    ]
    
    PRIORITY_LEVELS = ["P0", "P1", "P2"]
    
    # Topics that should use RAG (vs routing)
    RAG_TOPICS = {"How-to", "Product", "Best practices", "API/SDK", "SSO"}
    
    # Knowledge Base URLs
    KNOWLEDGE_BASE_URLS = [
        "https://docs.atlan.com/",
        "https://developer.atlan.com/",
    ]
    
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 3
    
    LANGSMITH_PROJECT = "customer-support-copilot"
    
    @classmethod
    def get_classification_prompt(cls) -> str:
        return """
        Analyze the following customer support ticket and classify it:

        Ticket: {ticket_text}

        Provide the classification in the following JSON format:
        {{
            "topic": "one of {topics}",
            "sentiment": "one of {sentiments}",
            "priority": "one of {priorities}"
        }}

        Classification Guidelines:
        - Topic: Choose the most relevant category based on the ticket content
        - Sentiment: Analyze the emotional tone of the customer
        - Priority: P0 (urgent/critical), P1 (medium), P2 (low priority)
        """.format(
            topics=cls.TOPIC_CATEGORIES,
            sentiments=cls.SENTIMENT_CATEGORIES,
            priorities=cls.PRIORITY_LEVELS
        )
    
    @classmethod
    def get_rag_prompt(cls) -> str:
        return """
        Use the following context to answer the customer's question. 
        Provide a helpful, accurate response based on the documentation.
        If you cannot find the answer in the context, say so clearly.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer: Provide a clear, helpful response based on the available information.
        """
    
    @classmethod
    def get_routing_message(cls, topic: str) -> str:
        return f"This ticket has been classified as a '{topic}' issue and routed to the appropriate team. You will receive a response within 24 hours."
