"""
Intelligent Customer Support Automation with Agentic RAG
======================================================

Main application demonstrating a multi-agent customer support system
using LangGraph for orchestration and RAG for knowledge retrieval.

Architecture:
- Triage Agent: Intent classification and sentiment analysis
- Retrieval Agent: Knowledge base and CRM data retrieval
- Response Agent: Personalized response generation
- Escalation Agent: Human handoff decisions

Author: Research Team
Version: 1.0.0
Date: March 2024
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import operator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# LangChain and LangGraph imports
try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not installed. Using mock implementations.")
    LANGCHAIN_AVAILABLE = False

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    logger.warning("LangGraph not installed. Using mock implementations.")
    LANGGRAPH_AVAILABLE = False

# Import local modules
from data_processing import (
    DataPipeline, TextPreprocessor, TicketProcessor,
    CustomerProfileProcessor, VectorStoreManager
)
from speech_to_text import SpeechToTextManager, STTProvider


# ============================================================================
# STATE DEFINITIONS
# ============================================================================

class IntentType(Enum):
    """Customer query intent types."""
    COMPLAINT = "complaint"
    INQUIRY = "inquiry"
    FEEDBACK = "feedback"
    TECHNICAL_ISSUE = "technical_issue"
    BILLING = "billing"
    GENERAL = "general"


class PriorityLevel(Enum):
    """Ticket priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class SentimentType(Enum):
    """Customer sentiment types."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class CustomerContext:
    """Customer context information."""
    customer_id: str
    name: str = "Unknown"
    plan: str = "Basic"
    lifetime_value: float = 0.0
    total_orders: int = 0
    total_tickets: int = 0
    avg_satisfaction: Optional[float] = None
    segment: str = "New"
    churn_risk: str = "low"
    support_history: List[Dict] = field(default_factory=list)


@dataclass
class AgentState:
    """State maintained across agent workflow."""
    # Input
    query: str = ""
    customer_id: str = ""
    channel: str = "chat"
    timestamp: str = ""
    
    # Context
    customer_context: Optional[CustomerContext] = None
    
    # Analysis results
    intent: str = ""
    intent_confidence: float = 0.0
    sentiment: str = ""
    sentiment_score: float = 0.0
    priority: str = ""
    
    # Retrieved information
    retrieved_docs: List[Dict] = field(default_factory=list)
    crm_data: Dict = field(default_factory=dict)
    
    # Response
    response: str = ""
    response_confidence: float = 0.0
    suggested_actions: List[str] = field(default_factory=list)
    
    # Routing
    should_escalate: bool = False
    escalation_reason: str = ""
    assigned_agent: str = ""
    
    # Tracking
    processing_steps: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# TypedDict for LangGraph state
class GraphState(TypedDict):
    """LangGraph state definition."""
    query: str
    customer_id: str
    channel: str
    customer_context: Optional[Dict]
    intent_analysis: Dict
    sentiment_analysis: Dict
    priority: str
    retrieved_docs: List[Dict]
    crm_data: Dict
    response: str
    response_confidence: float
    should_escalate: bool
    escalation_reason: str
    processing_history: List[str]


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

class TriageAgent:
    """
    Triage Agent: Classifies intent and analyzes sentiment.
    Uses lightweight LLM for fast classification.
    """
    
    def __init__(self, llm=None):
        """Initialize Triage Agent."""
        self.llm = llm
        self.preprocessor = TextPreprocessor()
        
        # Intent patterns for rule-based fallback
        self.intent_patterns = {
            IntentType.COMPLAINT: [
                'disappointed', 'unacceptable', 'terrible', 'worst', 'angry',
                'frustrated', 'complaint', 'unhappy', 'dissatisfied', 'problem'
            ],
            IntentType.TECHNICAL_ISSUE: [
                'error', 'bug', 'crash', 'not working', 'broken', 'failed',
                'login', 'password', 'access', 'connection', 'slow'
            ],
            IntentType.BILLING: [
                'charge', 'payment', 'refund', 'billing', 'invoice', 'price',
                'cost', 'subscription', 'overcharged', 'money'
            ],
            IntentType.FEEDBACK: [
                'feedback', 'suggestion', 'improve', 'feature', 'like', 'love',
                'great', 'excellent', 'amazing', 'wonderful'
            ],
            IntentType.INQUIRY: [
                'how to', 'what is', 'can you', 'question', 'information',
                'help', 'support', 'guide', 'tutorial', 'docs'
            ]
        }
    
    def analyze(self, query: str, customer_context: Optional[CustomerContext] = None) -> Dict[str, Any]:
        """
        Analyze query for intent and sentiment.
        
        Args:
            query: Customer query text
            customer_context: Optional customer context
            
        Returns:
            Analysis results
        """
        logger.info(f"TriageAgent analyzing query: {query[:50]}...")
        
        # Intent classification
        intent_result = self._classify_intent(query)
        
        # Sentiment analysis
        sentiment_result = self._analyze_sentiment(query)
        
        # Priority determination
        priority = self._determine_priority(
            intent_result['intent'],
            sentiment_result['sentiment'],
            customer_context
        )
        
        result = {
            'intent': intent_result['intent'],
            'intent_confidence': intent_result['confidence'],
            'intent_scores': intent_result['scores'],
            'sentiment': sentiment_result['sentiment'],
            'sentiment_score': sentiment_result['score'],
            'urgency_level': sentiment_result['urgency'],
            'priority': priority,
            'keywords': self.preprocessor.extract_keywords(query, top_n=5)
        }
        
        logger.info(f"Triage complete: intent={result['intent']}, sentiment={result['sentiment']}")
        return result
    
    def _classify_intent(self, query: str) -> Dict[str, Any]:
        """Classify query intent using rule-based approach."""
        query_lower = query.lower()
        scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = sum(2 if pattern in query_lower else 0 for pattern in patterns)
            # Check for partial matches
            score += sum(1 for pattern in patterns if any(word in query_lower for word in pattern.split()))
            scores[intent.value] = score
        
        # Get best match
        if max(scores.values()) > 0:
            best_intent = max(scores, key=scores.get)
            confidence = min(max(scores.values()) / 5, 1.0)
        else:
            best_intent = IntentType.GENERAL.value
            confidence = 0.5
            scores[IntentType.GENERAL.value] = 0
        
        return {
            'intent': best_intent,
            'confidence': round(confidence, 2),
            'scores': scores
        }
    
    def _analyze_sentiment(self, query: str) -> Dict[str, Any]:
        """Analyze sentiment of query."""
        query_lower = query.lower()
        
        # Sentiment keywords
        positive = ['good', 'great', 'excellent', 'love', 'best', 'amazing', 'thank', 'thanks', 'happy']
        negative = ['bad', 'terrible', 'awful', 'worst', 'hate', 'disappointed', 'angry', 'frustrated']
        very_negative = ['unacceptable', 'ridiculous', 'outrageous', 'disgusting', 'furious']
        urgency = ['urgent', 'asap', 'immediately', 'emergency', 'critical', 'now']
        
        pos_count = sum(1 for word in positive if word in query_lower)
        neg_count = sum(1 for word in negative if word in query_lower)
        very_neg_count = sum(1 for word in very_negative if word in query_lower)
        urgency_count = sum(1 for word in urgency if word in query_lower)
        
        # Determine sentiment
        if very_neg_count > 0 or neg_count >= 3:
            sentiment = SentimentType.VERY_NEGATIVE.value
            score = -0.8
        elif neg_count > pos_count:
            sentiment = SentimentType.NEGATIVE.value
            score = -0.4
        elif pos_count > neg_count:
            sentiment = SentimentType.POSITIVE.value
            score = 0.5
        else:
            sentiment = SentimentType.NEUTRAL.value
            score = 0.0
        
        return {
            'sentiment': sentiment,
            'score': round(score, 2),
            'urgency': 'high' if urgency_count >= 2 else 'medium' if urgency_count == 1 else 'low',
            'positive_indicators': pos_count,
            'negative_indicators': neg_count
        }
    
    def _determine_priority(self, intent: str, sentiment: str, 
                           customer_context: Optional[CustomerContext]) -> str:
        """Determine ticket priority."""
        # Base priority by intent
        priority_map = {
            IntentType.COMPLAINT.value: PriorityLevel.HIGH.value,
            IntentType.TECHNICAL_ISSUE.value: PriorityLevel.HIGH.value,
            IntentType.BILLING.value: PriorityLevel.MEDIUM.value,
            IntentType.INQUIRY.value: PriorityLevel.LOW.value,
            IntentType.FEEDBACK.value: PriorityLevel.LOW.value,
            IntentType.GENERAL.value: PriorityLevel.LOW.value
        }
        
        priority = priority_map.get(intent, PriorityLevel.LOW.value)
        
        # Adjust by sentiment
        if sentiment == SentimentType.VERY_NEGATIVE.value:
            priority = PriorityLevel.URGENT.value
        elif sentiment == SentimentType.NEGATIVE.value and priority != PriorityLevel.URGENT.value:
            priority = PriorityLevel.HIGH.value
        
        # Adjust by customer segment
        if customer_context:
            if customer_context.segment == 'VIP':
                if priority == PriorityLevel.LOW.value:
                    priority = PriorityLevel.MEDIUM.value
                elif priority == PriorityLevel.MEDIUM.value:
                    priority = PriorityLevel.HIGH.value
            
            if customer_context.churn_risk == 'high':
                priority = PriorityLevel.HIGH.value
        
        return priority


class RetrievalAgent:
    """
    Retrieval Agent: Retrieves relevant information from knowledge base and CRM.
    Uses RAG with multi-query expansion and reranking.
    """
    
    def __init__(self, vector_store: VectorStoreManager = None, embeddings=None):
        """Initialize Retrieval Agent."""
        self.vector_store = vector_store or VectorStoreManager()
        self.embeddings = embeddings
        self.preprocessor = TextPreprocessor()
    
    def retrieve(self, query: str, customer_context: Optional[CustomerContext] = None,
                 intent: str = "", top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve relevant documents and CRM data.
        
        Args:
            query: Customer query
            customer_context: Customer context
            intent: Classified intent
            top_k: Number of documents to retrieve
            
        Returns:
            Retrieved information
        """
        logger.info(f"RetrievalAgent retrieving for query: {query[:50]}...")
        
        results = {
            'knowledge_docs': [],
            'crm_data': {},
            'retrieval_confidence': 0.0
        }
        
        # Retrieve from knowledge base
        kb_docs = self._retrieve_knowledge_base(query, intent, top_k)
        results['knowledge_docs'] = kb_docs
        
        # Retrieve CRM data
        if customer_context:
            crm_data = self._retrieve_crm_data(customer_context)
            results['crm_data'] = crm_data
        
        # Calculate retrieval confidence
        if kb_docs:
            results['retrieval_confidence'] = sum(d.get('relevance_score', 0) for d in kb_docs) / len(kb_docs)
        
        logger.info(f"Retrieved {len(kb_docs)} KB docs, CRM data: {bool(results['crm_data'])}")
        return results
    
    def _retrieve_knowledge_base(self, query: str, intent: str, top_k: int) -> List[Dict]:
        """Retrieve from knowledge base with multi-query expansion."""
        # Generate query variations
        queries = self._generate_query_variations(query, intent)
        
        all_results = []
        
        # Search with each query variation
        for q in queries:
            # Simple keyword matching (in production, use embeddings)
            results = self._keyword_search(q, top_k=3)
            all_results.extend(results)
        
        # Deduplicate and rerank
        seen_ids = set()
        unique_results = []
        for r in all_results:
            if r['doc_id'] not in seen_ids:
                seen_ids.add(r['doc_id'])
                unique_results.append(r)
        
        # Sort by relevance score
        unique_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return unique_results[:top_k]
    
    def _generate_query_variations(self, query: str, intent: str) -> List[str]:
        """Generate query variations for better retrieval."""
        variations = [query]
        
        # Add intent-specific terms
        intent_terms = {
            'billing': ['payment', 'charge', 'billing', 'invoice', 'price', 'cost'],
            'technical_issue': ['error', 'bug', 'fix', 'troubleshoot', 'solution'],
            'complaint': ['refund', 'return', 'policy', 'compensation'],
            'inquiry': ['how to', 'guide', 'tutorial', 'documentation']
        }
        
        if intent in intent_terms:
            for term in intent_terms[intent][:2]:
                variations.append(f"{query} {term}")
        
        # Add keyword-only query
        keywords = self.preprocessor.extract_keywords(query, top_n=3)
        if keywords:
            variations.append(' '.join(keywords))
        
        return list(set(variations))
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Simple keyword-based search (replace with vector search in production)."""
        # This is a mock implementation
        # In production, use: self.vector_store.similarity_search(embedding, top_k)
        
        query_keywords = set(query.lower().split())
        results = []
        
        # Mock knowledge base documents
        kb_docs = [
            {
                'doc_id': 'KB-001',
                'title': 'Shipping Policy',
                'content': 'Standard shipping takes 5-7 business days. Express shipping available.',
                'category': 'policies'
            },
            {
                'doc_id': 'KB-002',
                'title': 'Return and Refund Policy',
                'content': '30-day return window. Full refund available for unused items.',
                'category': 'policies'
            },
            {
                'doc_id': 'KB-003',
                'title': 'Subscription Plans',
                'content': 'Basic $9.99, Pro $29.99, Enterprise $99.99 per month.',
                'category': 'billing'
            },
            {
                'doc_id': 'KB-004',
                'title': 'Account Security',
                'content': 'Enable 2FA for enhanced security. Password reset available.',
                'category': 'technical'
            },
            {
                'doc_id': 'KB-006',
                'title': 'Troubleshooting',
                'content': 'Error 502: Wait and retry. Clear cache. Check status page.',
                'category': 'technical'
            }
        ]
        
        for doc in kb_docs:
            content = f"{doc['title']} {doc['content']}".lower()
            content_words = set(content.split())
            
            # Calculate overlap
            overlap = len(query_keywords & content_words)
            total = len(query_keywords)
            
            if total > 0 and overlap > 0:
                score = overlap / total
                if score > 0.1:  # Minimum threshold
                    results.append({
                        'doc_id': doc['doc_id'],
                        'title': doc['title'],
                        'content': doc['content'],
                        'category': doc['category'],
                        'relevance_score': round(score, 2)
                    })
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:top_k]
    
    def _retrieve_crm_data(self, customer_context: CustomerContext) -> Dict:
        """Retrieve relevant CRM data for customer."""
        return {
            'customer_id': customer_context.customer_id,
            'name': customer_context.name,
            'plan': customer_context.plan,
            'segment': customer_context.segment,
            'lifetime_value': customer_context.lifetime_value,
            'total_orders': customer_context.total_orders,
            'total_tickets': customer_context.total_tickets,
            'avg_satisfaction': customer_context.avg_satisfaction,
            'churn_risk': customer_context.churn_risk,
            'recent_tickets': customer_context.support_history[-3:] if customer_context.support_history else []
        }


class ResponseAgent:
    """
    Response Agent: Generates personalized responses based on retrieved information.
    Considers customer context, sentiment, and intent.
    """
    
    def __init__(self, llm=None):
        """Initialize Response Agent."""
        self.llm = llm
        
        # Response templates for different scenarios
        self.templates = {
            'greeting': "Hello {name}, thank you for contacting us. ",
            'acknowledgment': "I understand your concern about {topic}. ",
            'empathy_negative': "I'm sorry to hear about the issue you're experiencing. ",
            'empathy_very_negative': "I sincerely apologize for the inconvenience caused. ",
            'solution': "Here's what I can do to help: {solution}",
            'escalation': "I'll connect you with a specialist who can assist you further. ",
            'closing': "Is there anything else I can help you with today?"
        }
    
    def generate_response(self, query: str, analysis: Dict, retrieval: Dict,
                         customer_context: Optional[CustomerContext]) -> Dict[str, Any]:
        """
        Generate personalized response.
        
        Args:
            query: Customer query
            analysis: Triage analysis results
            retrieval: Retrieval results
            customer_context: Customer context
            
        Returns:
            Generated response and metadata
        """
        logger.info("ResponseAgent generating response...")
        
        sentiment = analysis.get('sentiment', 'neutral')
        intent = analysis.get('intent', 'general')
        priority = analysis.get('priority', 'low')
        
        # Build response
        response_parts = []
        
        # Greeting
        name = customer_context.name if customer_context else "there"
        response_parts.append(self.templates['greeting'].format(name=name))
        
        # Empathy for negative sentiment
        if sentiment == SentimentType.NEGATIVE.value:
            response_parts.append(self.templates['empathy_negative'])
        elif sentiment == SentimentType.VERY_NEGATIVE.value:
            response_parts.append(self.templates['empathy_very_negative'])
        
        # Acknowledgment
        keywords = analysis.get('keywords', [])
        topic = keywords[0] if keywords else "this"
        response_parts.append(self.templates['acknowledgment'].format(topic=topic))
        
        # Add retrieved information
        kb_docs = retrieval.get('knowledge_docs', [])
        if kb_docs:
            best_doc = kb_docs[0]
            response_parts.append(f"Based on our {best_doc['title']}: {best_doc['content'][:200]}... ")
        
        # Intent-specific response
        intent_response = self._get_intent_response(intent, retrieval, customer_context)
        response_parts.append(intent_response)
        
        # Closing
        response_parts.append(self.templates['closing'])
        
        response = ' '.join(response_parts)
        
        # Calculate confidence
        confidence = self._calculate_confidence(analysis, retrieval)
        
        # Suggest actions
        suggested_actions = self._suggest_actions(intent, priority, customer_context)
        
        result = {
            'response': response,
            'confidence': round(confidence, 2),
            'suggested_actions': suggested_actions,
            'response_type': 'automated' if confidence > 0.7 else 'needs_review'
        }
        
        logger.info(f"Response generated with confidence: {result['confidence']}")
        return result
    
    def _get_intent_response(self, intent: str, retrieval: Dict, 
                            customer_context: Optional[CustomerContext]) -> str:
        """Get intent-specific response content."""
        crm_data = retrieval.get('crm_data', {})
        
        if intent == IntentType.BILLING.value:
            plan = crm_data.get('plan', 'Basic')
            return f"Your current plan is {plan}. For billing changes, you can update in your account settings."
        
        elif intent == IntentType.TECHNICAL_ISSUE.value:
            return "I've found some troubleshooting steps that may help resolve this issue."
        
        elif intent == IntentType.COMPLAINT.value:
            segment = crm_data.get('segment', 'Regular')
            if segment in ['VIP', 'High Value']:
                return "As a valued customer, I'll ensure this is resolved promptly."
            return "I understand your frustration and want to make this right."
        
        elif intent == IntentType.INQUIRY.value:
            return "I've gathered the relevant information for you."
        
        else:
            return "I'm here to help with any questions you may have."
    
    def _calculate_confidence(self, analysis: Dict, retrieval: Dict) -> float:
        """Calculate response confidence score."""
        intent_conf = analysis.get('intent_confidence', 0.5)
        retrieval_conf = retrieval.get('retrieval_confidence', 0.5)
        
        # Weight factors
        confidence = (intent_conf * 0.4) + (retrieval_conf * 0.6)
        
        return min(confidence, 1.0)
    
    def _suggest_actions(self, intent: str, priority: str, 
                        customer_context: Optional[CustomerContext]) -> List[str]:
        """Suggest follow-up actions."""
        actions = []
        
        if priority in ['high', 'urgent']:
            actions.append('escalate_to_human')
        
        if intent == IntentType.BILLING.value:
            actions.append('check_billing_history')
        
        if intent == IntentType.TECHNICAL_ISSUE.value:
            actions.append('create_support_ticket')
        
        if customer_context and customer_context.churn_risk == 'high':
            actions.append('apply_retention_offer')
        
        if not actions:
            actions.append('mark_resolved')
        
        return actions


class EscalationAgent:
    """
    Escalation Agent: Decides when to hand off to human agents.
    Considers confidence scores, sentiment, and business rules.
    """
    
    def __init__(self):
        """Initialize Escalation Agent."""
        # Escalation rules
        self.escalation_keywords = [
            'lawsuit', 'lawyer', 'legal', 'attorney', 'sue',
            'fraud', 'scam', 'unethical', 'illegal',
            'media', 'journalist', 'reporter', 'news',
            'cancel service', 'close account', 'switch to competitor'
        ]
    
    def should_escalate(self, query: str, analysis: Dict, response: Dict,
                       customer_context: Optional[CustomerContext]) -> Dict[str, Any]:
        """
        Determine if conversation should be escalated to human.
        
        Args:
            query: Customer query
            analysis: Triage analysis
            response: Generated response
            customer_context: Customer context
            
        Returns:
            Escalation decision
        """
        logger.info("EscalationAgent evaluating escalation...")
        
        reasons = []
        should_escalate = False
        
        # Check confidence score
        response_conf = response.get('confidence', 0)
        if response_conf < 0.5:
            should_escalate = True
            reasons.append(f"Low response confidence: {response_conf:.2f}")
        
        # Check sentiment
        sentiment = analysis.get('sentiment', 'neutral')
        if sentiment == SentimentType.VERY_NEGATIVE.value:
            should_escalate = True
            reasons.append("Very negative customer sentiment")
        
        # Check priority
        priority = analysis.get('priority', 'low')
        if priority == PriorityLevel.URGENT.value:
            should_escalate = True
            reasons.append("Urgent priority level")
        
        # Check for escalation keywords
        query_lower = query.lower()
        for keyword in self.escalation_keywords:
            if keyword in query_lower:
                should_escalate = True
                reasons.append(f"Escalation keyword detected: '{keyword}'")
                break
        
        # Check customer segment
        if customer_context:
            if customer_context.segment == 'VIP' and sentiment == SentimentType.NEGATIVE.value:
                should_escalate = True
                reasons.append("VIP customer with negative sentiment")
            
            if customer_context.churn_risk == 'high' and priority in ['high', 'urgent']:
                should_escalate = True
                reasons.append("High churn risk customer")
        
        # Check intent
        intent = analysis.get('intent', '')
        if intent in [IntentType.COMPLAINT.value] and sentiment == SentimentType.NEGATIVE.value:
            if customer_context and customer_context.segment in ['VIP', 'High Value']:
                should_escalate = True
                reasons.append("High-value customer complaint")
        
        result = {
            'should_escalate': should_escalate,
            'reasons': reasons,
            'priority': priority if should_escalate else 'normal',
            'assigned_team': self._get_assigned_team(intent, priority),
            'estimated_wait_time': self._estimate_wait_time(priority) if should_escalate else 0
        }
        
        logger.info(f"Escalation decision: {should_escalate}, reasons: {reasons}")
        return result
    
    def _get_assigned_team(self, intent: str, priority: str) -> str:
        """Determine which team to assign escalated ticket to."""
        team_map = {
            IntentType.BILLING.value: 'billing_support',
            IntentType.TECHNICAL_ISSUE.value: 'technical_support',
            IntentType.COMPLAINT.value: 'customer_retention',
            IntentType.INQUIRY.value: 'general_support',
            IntentType.FEEDBACK.value: 'product_team'
        }
        
        team = team_map.get(intent, 'general_support')
        
        if priority == PriorityLevel.URGENT.value:
            team = 'senior_' + team
        
        return team
    
    def _estimate_wait_time(self, priority: str) -> int:
        """Estimate wait time for human agent in minutes."""
        wait_times = {
            PriorityLevel.URGENT.value: 2,
            PriorityLevel.HIGH.value: 5,
            PriorityLevel.MEDIUM.value: 15,
            PriorityLevel.LOW.value: 30
        }
        return wait_times.get(priority, 15)


# ============================================================================
# LANGGRAPH WORKFLOW
# ============================================================================

def create_workflow(triage_agent: TriageAgent, 
                   retrieval_agent: RetrievalAgent,
                   response_agent: ResponseAgent,
                   escalation_agent: EscalationAgent) -> StateGraph:
    """
    Create LangGraph workflow for customer support.
    
    Args:
        triage_agent: Triage agent instance
        retrieval_agent: Retrieval agent instance
        response_agent: Response agent instance
        escalation_agent: Escalation agent instance
        
    Returns:
        Compiled StateGraph
    """
    
    def triage_node(state: GraphState) -> GraphState:
        """Triage node: classify intent and sentiment."""
        analysis = triage_agent.analyze(
            state['query'],
            CustomerContext(**state['customer_context']) if state['customer_context'] else None
        )
        
        state['intent_analysis'] = {
            'intent': analysis['intent'],
            'confidence': analysis['intent_confidence'],
            'scores': analysis['intent_scores']
        }
        state['sentiment_analysis'] = {
            'sentiment': analysis['sentiment'],
            'score': analysis['sentiment_score'],
            'urgency': analysis['urgency_level']
        }
        state['priority'] = analysis['priority']
        state['processing_history'].append('triage_complete')
        
        return state
    
    def retrieval_node(state: GraphState) -> GraphState:
        """Retrieval node: get relevant information."""
        retrieval = retrieval_agent.retrieve(
            state['query'],
            CustomerContext(**state['customer_context']) if state['customer_context'] else None,
            state['intent_analysis']['intent']
        )
        
        state['retrieved_docs'] = retrieval['knowledge_docs']
        state['crm_data'] = retrieval['crm_data']
        state['processing_history'].append('retrieval_complete')
        
        return state
    
    def response_node(state: GraphState) -> GraphState:
        """Response node: generate response."""
        response = response_agent.generate_response(
            state['query'],
            {**state['intent_analysis'], **state['sentiment_analysis'], 'priority': state['priority']},
            {'knowledge_docs': state['retrieved_docs'], 'crm_data': state['crm_data']},
            CustomerContext(**state['customer_context']) if state['customer_context'] else None
        )
        
        state['response'] = response['response']
        state['response_confidence'] = response['confidence']
        state['processing_history'].append('response_generated')
        
        return state
    
    def escalation_node(state: GraphState) -> GraphState:
        """Escalation node: decide on human handoff."""
        escalation = escalation_agent.should_escalate(
            state['query'],
            {**state['intent_analysis'], **state['sentiment_analysis'], 'priority': state['priority']},
            {'confidence': state['response_confidence']},
            CustomerContext(**state['customer_context']) if state['customer_context'] else None
        )
        
        state['should_escalate'] = escalation['should_escalate']
        state['escalation_reason'] = '; '.join(escalation['reasons'])
        state['processing_history'].append('escalation_checked')
        
        return state
    
    # Build graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("triage", triage_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("response", response_node)
    workflow.add_node("escalation", escalation_node)
    
    # Add edges
    workflow.set_entry_point("triage")
    workflow.add_edge("triage", "retrieval")
    workflow.add_edge("retrieval", "response")
    workflow.add_edge("response", "escalation")
    workflow.add_edge("escalation", END)
    
    return workflow.compile()


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class CustomerSupportSystem:
    """
    Main Customer Support System integrating all components.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize customer support system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize agents
        self.triage_agent = TriageAgent()
        self.retrieval_agent = RetrievalAgent()
        self.response_agent = ResponseAgent()
        self.escalation_agent = EscalationAgent()
        
        # Initialize workflow
        self.workflow = create_workflow(
            self.triage_agent,
            self.retrieval_agent,
            self.response_agent,
            self.escalation_agent
        )
        
        # Initialize STT
        self.stt_manager = None
        if self.config.get('enable_voice', False):
            try:
                self.stt_manager = SpeechToTextManager(
                    provider=STTProvider.GOOGLE_SPEECH
                )
            except Exception as e:
                logger.warning(f"STT not available: {e}")
        
        # Data pipeline
        self.data_pipeline = DataPipeline()
        
        logger.info("Customer Support System initialized")
    
    def process_query(self, query: str, customer_id: str = "",
                     channel: str = "chat") -> Dict[str, Any]:
        """
        Process a customer query through the full workflow.
        
        Args:
            query: Customer query text
            customer_id: Customer identifier
            channel: Communication channel
            
        Returns:
            Complete processing result
        """
        logger.info(f"Processing query from customer {customer_id} via {channel}")
        
        # Load customer context
        customer_context = self._load_customer_context(customer_id)
        
        # Initialize state
        initial_state = GraphState(
            query=query,
            customer_id=customer_id,
            channel=channel,
            customer_context=vars(customer_context) if customer_context else None,
            intent_analysis={},
            sentiment_analysis={},
            priority='',
            retrieved_docs=[],
            crm_data={},
            response='',
            response_confidence=0.0,
            should_escalate=False,
            escalation_reason='',
            processing_history=[]
        )
        
        # Run workflow
        try:
            result = self.workflow.invoke(initial_state)
            
            # Format output
            output = {
                'query': result['query'],
                'customer_id': result['customer_id'],
                'channel': result['channel'],
                'analysis': {
                    'intent': result['intent_analysis'],
                    'sentiment': result['sentiment_analysis'],
                    'priority': result['priority']
                },
                'retrieved_information': {
                    'knowledge_docs': result['retrieved_docs'],
                    'crm_data': result['crm_data']
                },
                'response': {
                    'text': result['response'],
                    'confidence': result['response_confidence']
                },
                'escalation': {
                    'should_escalate': result['should_escalate'],
                    'reason': result['escalation_reason']
                },
                'processing_history': result['processing_history'],
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Query processing complete")
            return output
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'error': str(e),
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
    
    def process_voice(self, audio_data: bytes, customer_id: str = "") -> Dict[str, Any]:
        """
        Process voice input from customer.
        
        Args:
            audio_data: Audio bytes
            customer_id: Customer identifier
            
        Returns:
            Processing result
        """
        if not self.stt_manager:
            return {'error': 'Voice processing not enabled'}
        
        # Transcribe audio
        transcription = self.stt_manager.transcribe(audio_data)
        
        # Process transcribed text
        result = self.process_query(
            transcription.text,
            customer_id,
            channel='voice'
        )
        
        # Add transcription info
        result['voice_input'] = {
            'transcription': transcription.to_dict(),
            'detected_language': transcription.language
        }
        
        return result
    
    def _load_customer_context(self, customer_id: str) -> Optional[CustomerContext]:
        """Load customer context from data."""
        if not customer_id:
            return None
        
        # In production, load from CRM/database
        # For demo, return mock context
        return CustomerContext(
            customer_id=customer_id,
            name="John Doe",
            plan="Pro",
            lifetime_value=1250.00,
            total_orders=15,
            total_tickets=3,
            avg_satisfaction=4.2,
            segment="High Value",
            churn_risk="low"
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            'agents': {
                'triage': 'active',
                'retrieval': 'active',
                'response': 'active',
                'escalation': 'active'
            },
            'features': {
                'voice_enabled': self.stt_manager is not None,
                'workflow_enabled': self.workflow is not None
            },
            'timestamp': datetime.now().isoformat()
        }


# ============================================================================
# DEMO AND TESTING
# ============================================================================

def run_demo():
    """Run demonstration of the customer support system."""
    print("\n" + "="*70)
    print("INTELLIGENT CUSTOMER SUPPORT AUTOMATION WITH AGENTIC RAG")
    print("="*70)
    
    # Initialize system
    system = CustomerSupportSystem(config={'enable_voice': False})
    
    # Test queries
    test_queries = [
        {
            'query': 'I was charged twice for my subscription this month. This is unacceptable!',
            'customer_id': 'CUST-1234',
            'channel': 'chat'
        },
        {
            'query': 'How do I reset my password? I can\'t login to my account.',
            'customer_id': 'CUST-5678',
            'channel': 'email'
        },
        {
            'query': 'What are the shipping options for international orders?',
            'customer_id': 'CUST-9012',
            'channel': 'chat'
        },
        {
            'query': 'Your service is terrible! I want to cancel my account immediately.',
            'customer_id': 'CUST-3456',
            'channel': 'chat'
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'-'*70}")
        print(f"TEST CASE {i}")
        print(f"{'-'*70}")
        print(f"Customer: {test['customer_id']}")
        print(f"Channel: {test['channel']}")
        print(f"Query: {test['query']}")
        print()
        
        # Process query
        result = system.process_query(
            test['query'],
            test['customer_id'],
            test['channel']
        )
        
        # Display results
        print("RESULTS:")
        print(f"  Intent: {result['analysis']['intent'].get('intent', 'N/A')} "
              f"(confidence: {result['analysis']['intent'].get('confidence', 0):.2f})")
        print(f"  Sentiment: {result['analysis']['sentiment'].get('sentiment', 'N/A')} "
              f"(score: {result['analysis']['sentiment'].get('score', 0):.2f})")
        print(f"  Priority: {result['analysis']['priority']}")
        print(f"  Retrieved Docs: {len(result['retrieved_information']['knowledge_docs'])}")
        print(f"  Response Confidence: {result['response']['confidence']:.2f}")
        print(f"  Should Escalate: {result['escalation']['should_escalate']}")
        if result['escalation']['should_escalate']:
            print(f"  Escalation Reason: {result['escalation']['reason']}")
        print()
        print(f"  Generated Response:")
        print(f"  {result['response']['text'][:200]}...")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    
    # System stats
    stats = system.get_system_stats()
    print("\nSystem Statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    run_demo()
