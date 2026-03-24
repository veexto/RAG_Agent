"""
Data Processing Module for Agentic RAG Customer Support System
===========================================================

This module handles all data ingestion, preprocessing, embedding generation,
and vector database operations for the customer support automation system.

Author: Research Team
Version: 1.0.0
Date: March 2024
"""

import json
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Data class for processed documents ready for vector storage."""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    chunk_index: int = 0
    total_chunks: int = 1


class TextPreprocessor:
    """Text preprocessing utilities for customer support data."""
    
    def __init__(self):
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'need',
            'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'her', 'its', 'our', 'their'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\?\!]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        return text.strip()
    
    def remove_stopwords(self, text: str) -> str:
        """Remove common stopwords from text."""
        words = text.split()
        filtered = [w for w in words if w.lower() not in self.stopwords]
        return ' '.join(filtered)
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract keywords from text using simple frequency analysis.
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of keywords
        """
        cleaned = self.clean_text(text)
        words = cleaned.split()
        
        # Filter out stopwords and short words
        words = [w for w in words if len(w) > 2 and w not in self.stopwords]
        
        # Count frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top N keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]
    
    def detect_language(self, text: str) -> str:
        """
        Simple language detection based on common words.
        
        Args:
            text: Input text
            
        Returns:
            Language code (en, vi, es, fr, unknown)
        """
        text_lower = text.lower()
        
        # Vietnamese indicators
        vi_indicators = ['cảm ơn', 'xin chào', 'tôi', 'bạn', 'củả', 'không', 'và', 'là', 'có', 'được']
        # Spanish indicators  
        es_indicators = ['gracias', 'hola', 'por favor', 'el', 'la', 'es', 'son', 'está', 'bien', 'mucho']
        # French indicators
        fr_indicators = ['merci', 'bonjour', 's\'il vous plaît', 'le', 'la', 'est', 'sont', 'très', 'bien', 'beaucoup']
        
        vi_count = sum(1 for word in vi_indicators if word in text_lower)
        es_count = sum(1 for word in es_indicators if word in text_lower)
        fr_count = sum(1 for word in fr_indicators if word in text_lower)
        
        if vi_count >= 2:
            return 'vi'
        elif es_count >= 2:
            return 'es'
        elif fr_count >= 2:
            return 'fr'
        else:
            return 'en'


class DocumentChunker:
    """Chunk documents into smaller pieces for embedding."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize chunker with configuration.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, doc_id: str) -> List[ProcessedDocument]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            doc_id: Document identifier
            
        Returns:
            List of processed document chunks
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_index = 0
        
        # Split by sentences first for better coherence
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(ProcessedDocument(
                        doc_id=f"{doc_id}_chunk_{chunk_index}",
                        content=current_chunk.strip(),
                        metadata={"source_doc": doc_id, "chunk_index": chunk_index},
                        chunk_index=chunk_index,
                        total_chunks=0  # Will update later
                    ))
                    chunk_index += 1
                current_chunk = sentence + " "
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(ProcessedDocument(
                doc_id=f"{doc_id}_chunk_{chunk_index}",
                content=current_chunk.strip(),
                metadata={"source_doc": doc_id, "chunk_index": chunk_index},
                chunk_index=chunk_index,
                total_chunks=0
            ))
        
        # Update total chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def chunk_knowledge_base(self, kb_docs: List[Dict]) -> List[ProcessedDocument]:
        """
        Process knowledge base documents into chunks.
        
        Args:
            kb_docs: List of knowledge base documents
            
        Returns:
            List of processed chunks
        """
        all_chunks = []
        
        for doc in kb_docs:
            content = f"{doc.get('title', '')}\n\n{doc.get('content', '')}"
            chunks = self.chunk_text(content, doc.get('doc_id', 'unknown'))
            
            # Add metadata to each chunk
            for chunk in chunks:
                chunk.metadata.update({
                    'category': doc.get('category', 'general'),
                    'keywords': doc.get('keywords', []),
                    'last_updated': doc.get('last_updated', ''),
                    'doc_type': 'knowledge_base'
                })
            
            all_chunks.extend(chunks)
        
        logger.info(f"Chunked {len(kb_docs)} KB docs into {len(all_chunks)} chunks")
        return all_chunks


class TicketProcessor:
    """Process customer support tickets for analysis and retrieval."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def process_ticket(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single ticket and extract features.
        
        Args:
            ticket: Raw ticket data
            
        Returns:
            Processed ticket with extracted features
        """
        query = ticket.get('query', '')
        
        processed = {
            'ticket_id': ticket.get('ticket_id'),
            'customer_id': ticket.get('customer_id'),
            'original_query': query,
            'cleaned_query': self.preprocessor.clean_text(query),
            'keywords': self.preprocessor.extract_keywords(query, top_n=5),
            'language': self.preprocessor.detect_language(query),
            'category': ticket.get('category'),
            'priority': ticket.get('priority'),
            'channel': ticket.get('channel'),
            'sentiment': self._analyze_sentiment(query),
            'query_length': len(query.split()),
            'has_order_number': bool(re.search(r'order\s*#?\s*\d+', query, re.IGNORECASE)),
            'has_account_issue': any(word in query.lower() for word in ['login', 'password', 'account', 'access']),
            'timestamp': ticket.get('created_at'),
            'processed_at': datetime.now().isoformat()
        }
        
        return processed
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Simple rule-based sentiment analysis.
        
        Args:
            text: Input text
            
        Returns:
            Sentiment analysis results
        """
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'love', 'best', 'amazing', 'wonderful', 
                         'fantastic', 'happy', 'satisfied', 'perfect', 'awesome', 'thank', 'thanks']
        negative_words = ['bad', 'terrible', 'awful', 'worst', 'hate', 'disappointed', 'frustrated',
                         'angry', 'unacceptable', 'horrible', 'poor', 'useless', 'broken', 'fail',
                         'problem', 'issue', 'error', 'bug', 'crash', 'slow', 'wrong']
        urgency_words = ['urgent', 'asap', 'immediately', 'emergency', 'critical', 'now', 'hurry',
                        'quick', 'fast', 'today', 'deadline']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        urgency_count = sum(1 for word in urgency_words if word in text_lower)
        
        # Determine sentiment
        if neg_count > pos_count:
            sentiment = 'negative'
        elif pos_count > neg_count:
            sentiment = 'positive'
        else:
            sentiment = 'neutral'
        
        # Calculate score (-1 to 1)
        total = pos_count + neg_count
        if total > 0:
            score = (pos_count - neg_count) / total
        else:
            score = 0.0
        
        return {
            'label': sentiment,
            'score': round(score, 2),
            'urgency_level': 'high' if urgency_count >= 2 else 'medium' if urgency_count == 1 else 'low',
            'positive_indicators': pos_count,
            'negative_indicators': neg_count,
            'urgency_indicators': urgency_count
        }
    
    def process_tickets_batch(self, tickets: List[Dict]) -> List[Dict]:
        """
        Process multiple tickets in batch.
        
        Args:
            tickets: List of ticket dictionaries
            
        Returns:
            List of processed tickets
        """
        processed = []
        for ticket in tickets:
            try:
                processed.append(self.process_ticket(ticket))
            except Exception as e:
                logger.error(f"Error processing ticket {ticket.get('ticket_id')}: {e}")
        
        logger.info(f"Processed {len(processed)} tickets")
        return processed
    
    def extract_ticket_summary(self, tickets: List[Dict]) -> Dict[str, Any]:
        """
        Generate summary statistics from tickets.
        
        Args:
            tickets: List of processed tickets
            
        Returns:
            Summary statistics
        """
        if not tickets:
            return {}
        
        categories = {}
        priorities = {}
        sentiments = {}
        channels = {}
        
        for ticket in tickets:
            cat = ticket.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
            
            pri = ticket.get('priority', 'unknown')
            priorities[pri] = priorities.get(pri, 0) + 1
            
            sent = ticket.get('sentiment', {}).get('label', 'unknown')
            sentiments[sent] = sentiments.get(sent, 0) + 1
            
            ch = ticket.get('channel', 'unknown')
            channels[ch] = channels.get(ch, 0) + 1
        
        return {
            'total_tickets': len(tickets),
            'category_distribution': categories,
            'priority_distribution': priorities,
            'sentiment_distribution': sentiments,
            'channel_distribution': channels,
            'avg_query_length': sum(t.get('query_length', 0) for t in tickets) / len(tickets),
            'escalation_rate': sum(1 for t in tickets if t.get('escalated')) / len(tickets) * 100
        }


class CustomerProfileProcessor:
    """Process and enrich customer profile data."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def enrich_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich customer profile with computed metrics.
        
        Args:
            profile: Raw customer profile
            
        Returns:
            Enriched profile
        """
        purchase_history = profile.get('purchase_history', [])
        support_history = profile.get('support_history', [])
        
        # Calculate purchase metrics
        total_orders = len(purchase_history)
        completed_orders = sum(1 for p in purchase_history if p.get('status') == 'completed')
        total_spent = sum(p.get('amount', 0) for p in purchase_history)
        avg_order_value = total_spent / total_orders if total_orders > 0 else 0
        
        # Calculate support metrics
        total_tickets = len(support_history)
        resolved_tickets = sum(1 for s in support_history if s.get('status') == 'resolved')
        satisfaction_scores = [s.get('satisfaction') for s in support_history if s.get('satisfaction')]
        avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else None
        
        # Determine customer segment
        if total_spent > 1000 and avg_satisfaction and avg_satisfaction >= 4:
            segment = 'VIP'
        elif total_spent > 500:
            segment = 'High Value'
        elif total_orders > 0:
            segment = 'Regular'
        else:
            segment = 'New'
        
        # Calculate customer health score (0-100)
        health_score = 100
        if avg_satisfaction and avg_satisfaction < 3:
            health_score -= 30
        if total_tickets > 5:
            health_score -= 20
        if profile.get('metrics', {}).get('account_status') == 'suspended':
            health_score -= 50
        health_score = max(0, health_score)
        
        enriched = {
            **profile,
            'computed_metrics': {
                'total_orders': total_orders,
                'completed_orders': completed_orders,
                'total_spent': round(total_spent, 2),
                'avg_order_value': round(avg_order_value, 2),
                'total_support_tickets': total_tickets,
                'resolved_tickets': resolved_tickets,
                'resolution_rate': round(resolved_tickets / total_tickets * 100, 1) if total_tickets > 0 else 100,
                'avg_satisfaction_score': round(avg_satisfaction, 1) if avg_satisfaction else None,
                'customer_segment': segment,
                'health_score': health_score,
                'churn_risk': 'high' if health_score < 50 else 'medium' if health_score < 75 else 'low'
            },
            'processed_at': datetime.now().isoformat()
        }
        
        return enriched
    
    def get_customer_context(self, profile: Dict[str, Any]) -> str:
        """
        Generate text context for a customer for RAG retrieval.
        
        Args:
            profile: Customer profile
            
        Returns:
            Context string
        """
        metrics = profile.get('computed_metrics', {})
        
        context = f"""Customer Profile:
Name: {profile.get('name', 'Unknown')}
Company: {profile.get('company', 'Unknown')}
Industry: {profile.get('industry', 'Unknown')}
Plan: {profile.get('plan', 'Unknown')}
Customer Segment: {metrics.get('customer_segment', 'Unknown')}
Total Spent: ${metrics.get('total_spent', 0)}
Total Orders: {metrics.get('total_orders', 0)}
Support Tickets: {metrics.get('total_support_tickets', 0)}
Average Satisfaction: {metrics.get('avg_satisfaction_score', 'N/A')}
Health Score: {metrics.get('health_score', 0)}/100
Churn Risk: {metrics.get('churn_risk', 'Unknown')}
"""
        return context


class VectorStoreManager:
    """Manager for vector database operations."""
    
    def __init__(self, collection_name: str = "customer_support"):
        """
        Initialize vector store manager.
        
        Args:
            collection_name: Name of the collection
        """
        self.collection_name = collection_name
        self.documents = []  # In-memory storage for demo
        logger.info(f"Initialized VectorStoreManager for collection: {collection_name}")
    
    def add_documents(self, documents: List[ProcessedDocument], embeddings: List[List[float]]):
        """
        Add documents with embeddings to vector store.
        
        Args:
            documents: List of processed documents
            embeddings: List of embedding vectors
        """
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
            self.documents.append(doc)
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def similarity_search(self, query_embedding: List[float], top_k: int = 5,
                         filter_metadata: Optional[Dict] = None) -> List[ProcessedDocument]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar documents
        """
        import numpy as np
        
        query_vec = np.array(query_embedding)
        results = []
        
        for doc in self.documents:
            if doc.embedding is None:
                continue
            
            # Apply metadata filter if provided
            if filter_metadata:
                match = True
                for key, value in filter_metadata.items():
                    if doc.metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            # Calculate cosine similarity
            doc_vec = np.array(doc.embedding)
            similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            
            results.append((doc, similarity))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in results[:top_k]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            'total_documents': len(self.documents),
            'collection_name': self.collection_name,
            'categories': list(set(d.metadata.get('category', 'unknown') for d in self.documents))
        }


class DataPipeline:
    """Main data pipeline orchestrating all processing steps."""
    
    def __init__(self, embedding_model=None):
        """
        Initialize data pipeline.
        
        Args:
            embedding_model: Optional embedding model for generating embeddings
        """
        self.preprocessor = TextPreprocessor()
        self.chunker = DocumentChunker()
        self.ticket_processor = TicketProcessor()
        self.profile_processor = CustomerProfileProcessor()
        self.vector_store = VectorStoreManager()
        self.embedding_model = embedding_model
    
    def load_json_data(self, filepath: str) -> List[Dict]:
        """Load data from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} records from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return []
    
    def process_knowledge_base(self, kb_path: str) -> List[ProcessedDocument]:
        """
        Process knowledge base documents through the pipeline.
        
        Args:
            kb_path: Path to knowledge base JSON file
            
        Returns:
            List of processed document chunks
        """
        kb_docs = self.load_json_data(kb_path)
        chunks = self.chunker.chunk_knowledge_base(kb_docs)
        
        # Generate embeddings if model available
        if self.embedding_model:
            embeddings = self.embedding_model.encode([c.content for c in chunks])
            self.vector_store.add_documents(chunks, embeddings)
        
        return chunks
    
    def process_tickets(self, tickets_path: str) -> List[Dict]:
        """
        Process support tickets through the pipeline.
        
        Args:
            tickets_path: Path to tickets JSON file
            
        Returns:
            List of processed tickets
        """
        tickets = self.load_json_data(tickets_path)
        processed = self.ticket_processor.process_tickets_batch(tickets)
        return processed
    
    def process_customer_profiles(self, profiles_path: str) -> List[Dict]:
        """
        Process customer profiles through the pipeline.
        
        Args:
            profiles_path: Path to profiles JSON file
            
        Returns:
            List of enriched profiles
        """
        profiles = self.load_json_data(profiles_path)
        enriched = [self.profile_processor.enrich_profile(p) for p in profiles]
        return enriched
    
    def run_full_pipeline(self, data_dir: str) -> Dict[str, Any]:
        """
        Run complete data processing pipeline.
        
        Args:
            data_dir: Directory containing data files
            
        Returns:
            Pipeline results summary
        """
        import os
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'knowledge_base': None,
            'tickets': None,
            'profiles': None
        }
        
        # Process knowledge base
        kb_path = os.path.join(data_dir, 'knowledge_base.json')
        if os.path.exists(kb_path):
            kb_chunks = self.process_knowledge_base(kb_path)
            results['knowledge_base'] = {
                'chunks_created': len(kb_chunks),
                'vector_store_stats': self.vector_store.get_stats()
            }
        
        # Process tickets
        tickets_path = os.path.join(data_dir, 'customer_tickets.json')
        if os.path.exists(tickets_path):
            processed_tickets = self.process_tickets(tickets_path)
            results['tickets'] = {
                'processed': len(processed_tickets),
                'summary': self.ticket_processor.extract_ticket_summary(processed_tickets)
            }
        
        # Process profiles
        profiles_path = os.path.join(data_dir, 'customer_profiles.json')
        if os.path.exists(profiles_path):
            enriched_profiles = self.process_customer_profiles(profiles_path)
            results['profiles'] = {
                'enriched': len(enriched_profiles)
            }
        
        logger.info("Full pipeline completed successfully")
        return results


# Demo usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Run pipeline on sample data
    data_directory = "../data"
    results = pipeline.run_full_pipeline(data_directory)
    
    print("\n" + "="*60)
    print("DATA PROCESSING PIPELINE RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))
