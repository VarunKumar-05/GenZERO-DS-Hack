# LangGraph Reasoning with Gemini for Evidence Extraction
# Copyright 2026 - Kharagpur Data Science Hackathon

"""
Reasoning component using LangGraph and Gemini API for:
1. Extracting supporting sentences from narratives
2. Explaining classification decisions
3. Linking evidence to backstory elements
"""

import os
from typing import List, Dict, Tuple, Optional, TypedDict, Annotated
from dataclasses import dataclass

from config import GeminiConfig

# LangGraph and Gemini imports (with fallback)
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


@dataclass
class Evidence:
    """Single piece of evidence from the narrative"""
    sentence_idx: int
    sentence_text: str
    relevance_score: float
    explanation: str


@dataclass
class ReasoningResult:
    """Complete reasoning result"""
    classification: str  # "consistent" or "inconsistent"
    confidence: float
    evidence: List[Evidence]
    summary: str


class ReasoningState(TypedDict):
    """State for LangGraph reasoning flow"""
    backstory: str
    narrative_sentences: List[str]
    character: str
    candidate_evidence: List[Tuple[int, str, float]]
    verified_evidence: List[Evidence]
    classification: Optional[str]
    confidence: Optional[float]
    summary: Optional[str]


class GeminiReasoner:
    """
    Gemini-based reasoner for narrative consistency analysis.
    Works standalone or as part of LangGraph flow.
    """
    
    def __init__(self, config: GeminiConfig):
        self.config = config
        self.model = None
        
        if GEMINI_AVAILABLE:
            # Use API key from config, fallback to environment variable
            api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(config.model_name)
                print(f"Gemini initialized with model: {config.model_name}")
    
    def is_available(self) -> bool:
        return self.model is not None
    
    def verify_evidence(
        self,
        backstory: str,
        candidate_sentence: str,
        character: str
    ) -> Tuple[bool, str, float]:
        """
        Verify if a sentence supports or contradicts the backstory.
        
        Returns:
            (is_relevant, explanation, confidence)
        """
        if not self.is_available():
            return True, "Gemini unavailable - using heuristic", 0.5
        
        prompt = f"""Analyze if this narrative sentence is relevant to verifying the character's backstory.

Character: {character}

Backstory claim: {backstory}

Narrative sentence: {candidate_sentence}

Respond in exactly this format:
RELEVANT: [yes/no]
EXPLANATION: [one sentence explaining why]
CONFIDENCE: [0.0-1.0]"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=self.config.max_output_tokens,
                    temperature=self.config.temperature,
                )
            )
            
            text = response.text.strip()
            
            # Parse response
            is_relevant = "yes" in text.lower().split("relevant:")[1].split("\n")[0].lower()
            
            explanation = ""
            if "EXPLANATION:" in text:
                explanation = text.split("EXPLANATION:")[1].split("\n")[0].strip()
            
            confidence = 0.5
            if "CONFIDENCE:" in text:
                try:
                    confidence = float(text.split("CONFIDENCE:")[1].split("\n")[0].strip())
                except:
                    pass
            
            return is_relevant, explanation, confidence
            
        except Exception as e:
            return True, f"Error: {str(e)[:50]}", 0.5
    
    def classify_consistency(
        self,
        backstory: str,
        evidence_sentences: List[str],
        character: str
    ) -> Tuple[str, float, str]:
        """
        Classify if backstory is consistent with narrative evidence.
        
        Returns:
            (classification, confidence, summary)
        """
        if not self.is_available():
            return "consistent", 0.5, "Gemini unavailable"
        
        evidence_text = "\n".join([f"- {s}" for s in evidence_sentences[:10]])
        
        prompt = f"""Determine if this hypothetical backstory is consistent with the narrative evidence.

Character: {character}

Hypothetical Backstory:
{backstory}

Relevant Narrative Evidence:
{evidence_text}

Instructions:
1. Analyze if the backstory is CONSISTENT or CONTRADICTS the narrative
2. Consider timing, character traits, relationships, and events
3. A backstory is INCONSISTENT if it directly contradicts narrative facts

Respond in exactly this format:
CLASSIFICATION: [consistent/inconsistent]
CONFIDENCE: [0.0-1.0]
SUMMARY: [2-3 sentences explaining your reasoning]"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=self.config.max_output_tokens,
                    temperature=self.config.temperature,
                )
            )
            
            text = response.text.strip()
            
            # Parse classification
            classification = "consistent"
            if "CLASSIFICATION:" in text:
                class_line = text.split("CLASSIFICATION:")[1].split("\n")[0].lower()
                if "inconsistent" in class_line or "contradict" in class_line:
                    classification = "inconsistent"
            
            # Parse confidence
            confidence = 0.5
            if "CONFIDENCE:" in text:
                try:
                    confidence = float(text.split("CONFIDENCE:")[1].split("\n")[0].strip())
                except:
                    pass
            
            # Parse summary
            summary = ""
            if "SUMMARY:" in text:
                summary = text.split("SUMMARY:")[1].strip()
            
            return classification, confidence, summary
            
        except Exception as e:
            return "consistent", 0.5, f"Error: {str(e)[:100]}"


class ReasoningGraph:
    """
    LangGraph-based reasoning pipeline for evidence extraction and classification.
    """
    
    def __init__(self, config: GeminiConfig):
        self.config = config
        self.reasoner = GeminiReasoner(config)
        self.graph = self._build_graph() if LANGGRAPH_AVAILABLE else None
    
    def _build_graph(self) -> Optional[StateGraph]:
        """Build the LangGraph reasoning flow"""
        if not LANGGRAPH_AVAILABLE:
            return None
        
        workflow = StateGraph(ReasoningState)
        
        # Add nodes
        workflow.add_node("filter_evidence", self._filter_evidence)
        workflow.add_node("verify_evidence", self._verify_evidence)
        workflow.add_node("classify", self._classify)
        workflow.add_node("summarize", self._summarize)
        
        # Add edges
        workflow.set_entry_point("filter_evidence")
        workflow.add_edge("filter_evidence", "verify_evidence")
        workflow.add_edge("verify_evidence", "classify")
        workflow.add_edge("classify", "summarize")
        workflow.add_edge("summarize", END)
        
        return workflow.compile()
    
    def _filter_evidence(self, state: ReasoningState) -> ReasoningState:
        """Filter candidate sentences by initial relevance"""
        # Already filtered by attention - just pass through top candidates
        state["candidate_evidence"] = state["candidate_evidence"][:self.config.top_k_evidence * 2]
        return state
    
    def _verify_evidence(self, state: ReasoningState) -> ReasoningState:
        """Verify each candidate with Gemini"""
        verified = []
        
        for idx, sentence, attn_score in state["candidate_evidence"]:
            is_relevant, explanation, confidence = self.reasoner.verify_evidence(
                state["backstory"],
                sentence,
                state["character"]
            )
            
            if is_relevant:
                verified.append(Evidence(
                    sentence_idx=idx,
                    sentence_text=sentence,
                    relevance_score=attn_score * confidence,
                    explanation=explanation
                ))
        
        # Keep top-k verified
        verified.sort(key=lambda e: e.relevance_score, reverse=True)
        state["verified_evidence"] = verified[:self.config.top_k_evidence]
        return state
    
    def _classify(self, state: ReasoningState) -> ReasoningState:
        """Classify consistency based on verified evidence"""
        evidence_texts = [e.sentence_text for e in state["verified_evidence"]]
        
        classification, confidence, summary = self.reasoner.classify_consistency(
            state["backstory"],
            evidence_texts,
            state["character"]
        )
        
        state["classification"] = classification
        state["confidence"] = confidence
        state["summary"] = summary
        return state
    
    def _summarize(self, state: ReasoningState) -> ReasoningState:
        """Final summarization (already done in classify)"""
        return state
    
    def run(
        self,
        backstory: str,
        narrative_sentences: List[str],
        character: str,
        candidate_evidence: List[Tuple[int, str, float]]
    ) -> ReasoningResult:
        """
        Run the full reasoning pipeline.
        
        Args:
            backstory: The hypothetical backstory
            narrative_sentences: All sentences from the narrative
            character: Character name
            candidate_evidence: List of (idx, sentence, score) from attention
            
        Returns:
            ReasoningResult with classification, evidence, and summary
        """
        initial_state = ReasoningState(
            backstory=backstory,
            narrative_sentences=narrative_sentences,
            character=character,
            candidate_evidence=candidate_evidence,
            verified_evidence=[],
            classification=None,
            confidence=None,
            summary=None
        )
        
        if self.graph is not None:
            final_state = self.graph.invoke(initial_state)
        else:
            # Fallback without LangGraph
            final_state = self._filter_evidence(initial_state)
            final_state = self._verify_evidence(final_state)
            final_state = self._classify(final_state)
        
        return ReasoningResult(
            classification=final_state["classification"] or "consistent",
            confidence=final_state["confidence"] or 0.5,
            evidence=final_state["verified_evidence"],
            summary=final_state["summary"] or ""
        )


def create_reasoning_graph(config: GeminiConfig) -> ReasoningGraph:
    """Factory function to create reasoning graph"""
    return ReasoningGraph(config)
