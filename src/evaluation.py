"""
Evaluation Module for Multilingual RAG System

This module implements various evaluation metrics for RAG systems including:
- Groundedness (is the answer supported by retrieved context?)
- Relevance (does the system fetch appropriate documents?)
- Semantic similarity
- ROUGE scores
- BLEU scores
"""

import logging
import re
import math
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
import numpy as np

# Evaluation metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("rouge_score not available. ROUGE evaluation will be limited.")

try:
    from sacrebleu import BLEU
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    logging.warning("sacrebleu not available. BLEU evaluation will be limited.")

try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    logging.warning("bert_score not available. BERTScore evaluation will be limited.")

# Sentence transformers for semantic similarity
from sentence_transformers import SentenceTransformer
import torch

# Local imports
from config import config

@dataclass
class EvaluationResult:
    """Data class for evaluation results"""
    metric: str
    score: float
    details: Dict = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}

@dataclass
class RAGEvaluationResult:
    """Comprehensive RAG evaluation result"""
    query: str
    response: str
    retrieved_chunks: List[Dict]
    expected_answer: Optional[str] = None
    groundedness_score: float = 0.0
    relevance_score: float = 0.0
    semantic_similarity: float = 0.0
    rouge_scores: Dict = None
    bleu_score: float = 0.0
    bert_score: float = 0.0
    overall_score: float = 0.0
    evaluation_details: Dict = None
    
    def __post_init__(self):
        if self.rouge_scores is None:
            self.rouge_scores = {}
        if self.evaluation_details is None:
            self.evaluation_details = {}

class RAGEvaluator:
    """
    Comprehensive evaluator for RAG systems
    """
    
    def __init__(self, embedding_model: str = None):
        """
        Initialize the evaluator
        
        Args:
            embedding_model: Sentence transformer model for semantic similarity
        """
        self.logger = logging.getLogger(__name__)
        self.embedding_model_name = embedding_model or config.EMBEDDING_MODEL
        
        # Initialize embedding model for semantic similarity
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.logger.info(f"Embedding model loaded: {self.embedding_model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
        
        # Initialize ROUGE scorer
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        else:
            self.rouge_scorer = None
        
        # Initialize BLEU scorer
        if BLEU_AVAILABLE:
            self.bleu_scorer = BLEU()
        else:
            self.bleu_scorer = None
    
    def evaluate_groundedness(
        self, 
        response: str, 
        retrieved_chunks: List[Dict]
    ) -> EvaluationResult:
        """
        Evaluate how well the response is grounded in the retrieved context
        
        Args:
            response: Generated response
            retrieved_chunks: List of retrieved document chunks
            
        Returns:
            EvaluationResult with groundedness score
        """
        if not retrieved_chunks:
            return EvaluationResult(
                metric="groundedness",
                score=0.0,
                details={"reason": "No chunks retrieved"}
            )
        
        # Combine all retrieved text
        context_text = " ".join([chunk.get('text', '') for chunk in retrieved_chunks])
        
        if not context_text.strip():
            return EvaluationResult(
                metric="groundedness",
                score=0.0,
                details={"reason": "Empty context"}
            )
        
        # Method 1: Simple token overlap
        token_overlap_score = self._calculate_token_overlap(response, context_text)
        
        # Method 2: Semantic similarity between response and context
        semantic_score = 0.0
        if self.embedding_model:
            try:
                response_embedding = self.embedding_model.encode([response])
                context_embedding = self.embedding_model.encode([context_text])
                
                # Calculate cosine similarity
                similarity = np.dot(response_embedding[0], context_embedding[0]) / (
                    np.linalg.norm(response_embedding[0]) * np.linalg.norm(context_embedding[0])
                )
                semantic_score = max(0.0, similarity)  # Ensure non-negative
                
            except Exception as e:
                self.logger.warning(f"Semantic similarity calculation failed: {e}")
        
        # Method 3: Check for factual claims grounding
        factual_score = self._evaluate_factual_grounding(response, context_text)
        
        # Combine scores (weighted average)
        weights = [0.3, 0.5, 0.2]  # token_overlap, semantic, factual
        scores = [token_overlap_score, semantic_score, factual_score]
        
        final_score = sum(w * s for w, s in zip(weights, scores))
        
        return EvaluationResult(
            metric="groundedness",
            score=final_score,
            details={
                "token_overlap": token_overlap_score,
                "semantic_similarity": semantic_score,
                "factual_grounding": factual_score,
                "context_length": len(context_text),
                "response_length": len(response)
            }
        )
    
    def evaluate_relevance(
        self, 
        query: str, 
        retrieved_chunks: List[Dict]
    ) -> EvaluationResult:
        """
        Evaluate how relevant the retrieved chunks are to the query
        
        Args:
            query: User query
            retrieved_chunks: List of retrieved document chunks
            
        Returns:
            EvaluationResult with relevance score
        """
        if not retrieved_chunks:
            return EvaluationResult(
                metric="relevance",
                score=0.0,
                details={"reason": "No chunks retrieved"}
            )
        
        relevance_scores = []
        chunk_details = []
        
        for i, chunk in enumerate(retrieved_chunks):
            chunk_text = chunk.get('text', '')
            chunk_similarity = chunk.get('similarity', 0.0)
            
            # Method 1: Use the similarity score from retrieval (if available)
            retrieval_score = chunk_similarity
            
            # Method 2: Calculate semantic similarity with embedding model
            semantic_score = 0.0
            if self.embedding_model and chunk_text:
                try:
                    query_embedding = self.embedding_model.encode([query])
                    chunk_embedding = self.embedding_model.encode([chunk_text])
                    
                    similarity = np.dot(query_embedding[0], chunk_embedding[0]) / (
                        np.linalg.norm(query_embedding[0]) * np.linalg.norm(chunk_embedding[0])
                    )
                    semantic_score = max(0.0, similarity)
                    
                except Exception as e:
                    self.logger.warning(f"Semantic similarity calculation failed for chunk {i}: {e}")
            
            # Method 3: Token overlap score
            token_score = self._calculate_token_overlap(query, chunk_text)
            
            # Combine scores for this chunk
            weights = [0.5, 0.3, 0.2]  # retrieval, semantic, token
            scores = [retrieval_score, semantic_score, token_score]
            
            chunk_relevance = sum(w * s for w, s in zip(weights, scores))
            relevance_scores.append(chunk_relevance)
            
            chunk_details.append({
                "chunk_index": i,
                "retrieval_score": retrieval_score,
                "semantic_score": semantic_score,
                "token_score": token_score,
                "combined_score": chunk_relevance,
                "text_length": len(chunk_text)
            })
        
        # Calculate overall relevance (average of top chunks with position weighting)
        if relevance_scores:
            # Weight scores by position (earlier chunks are more important)
            position_weights = [1.0 / (i + 1) for i in range(len(relevance_scores))]
            weighted_scores = [score * weight for score, weight in zip(relevance_scores, position_weights)]
            
            average_relevance = sum(weighted_scores) / sum(position_weights)
        else:
            average_relevance = 0.0
        
        return EvaluationResult(
            metric="relevance",
            score=average_relevance,
            details={
                "chunk_details": chunk_details,
                "average_score": np.mean(relevance_scores) if relevance_scores else 0.0,
                "max_score": max(relevance_scores) if relevance_scores else 0.0,
                "min_score": min(relevance_scores) if relevance_scores else 0.0,
                "num_chunks": len(retrieved_chunks)
            }
        )
    
    def evaluate_semantic_similarity(
        self, 
        response: str, 
        expected_answer: str
    ) -> EvaluationResult:
        """
        Evaluate semantic similarity between response and expected answer
        
        Args:
            response: Generated response
            expected_answer: Expected/reference answer
            
        Returns:
            EvaluationResult with semantic similarity score
        """
        if not self.embedding_model:
            return EvaluationResult(
                metric="semantic_similarity",
                score=0.0,
                details={"reason": "Embedding model not available"}
            )
        
        try:
            # Generate embeddings
            response_embedding = self.embedding_model.encode([response])
            expected_embedding = self.embedding_model.encode([expected_answer])
            
            # Calculate cosine similarity
            similarity = np.dot(response_embedding[0], expected_embedding[0]) / (
                np.linalg.norm(response_embedding[0]) * np.linalg.norm(expected_embedding[0])
            )
            
            # Ensure score is between 0 and 1
            similarity_score = max(0.0, min(1.0, similarity))
            
            return EvaluationResult(
                metric="semantic_similarity",
                score=similarity_score,
                details={
                    "cosine_similarity": similarity,
                    "response_length": len(response),
                    "expected_length": len(expected_answer)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Semantic similarity calculation failed: {e}")
            return EvaluationResult(
                metric="semantic_similarity",
                score=0.0,
                details={"error": str(e)}
            )
    
    def evaluate_rouge(
        self, 
        response: str, 
        expected_answer: str
    ) -> EvaluationResult:
        """
        Evaluate ROUGE scores
        
        Args:
            response: Generated response
            expected_answer: Expected/reference answer
            
        Returns:
            EvaluationResult with ROUGE scores
        """
        if not self.rouge_scorer:
            return EvaluationResult(
                metric="rouge",
                score=0.0,
                details={"reason": "ROUGE scorer not available"}
            )
        
        try:
            scores = self.rouge_scorer.score(expected_answer, response)
            
            # Extract F1 scores
            rouge_scores = {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
            
            # Calculate average ROUGE score
            average_rouge = np.mean(list(rouge_scores.values()))
            
            return EvaluationResult(
                metric="rouge",
                score=average_rouge,
                details=rouge_scores
            )
            
        except Exception as e:
            self.logger.error(f"ROUGE calculation failed: {e}")
            return EvaluationResult(
                metric="rouge",
                score=0.0,
                details={"error": str(e)}
            )
    
    def evaluate_bleu(
        self, 
        response: str, 
        expected_answer: str
    ) -> EvaluationResult:
        """
        Evaluate BLEU score
        
        Args:
            response: Generated response
            expected_answer: Expected/reference answer
            
        Returns:
            EvaluationResult with BLEU score
        """
        if not self.bleu_scorer:
            return EvaluationResult(
                metric="bleu",
                score=0.0,
                details={"reason": "BLEU scorer not available"}
            )
        
        try:
            # BLEU expects lists of references
            references = [expected_answer.split()]
            hypothesis = response.split()
            
            bleu_score = self.bleu_scorer.sentence_score(hypothesis, references)
            
            return EvaluationResult(
                metric="bleu",
                score=bleu_score.score / 100.0,  # Convert to 0-1 range
                details={
                    "bleu_score": bleu_score.score,
                    "brevity_penalty": bleu_score.bp,
                    "precision_scores": bleu_score.precisions
                }
            )
            
        except Exception as e:
            self.logger.error(f"BLEU calculation failed: {e}")
            return EvaluationResult(
                metric="bleu",
                score=0.0,
                details={"error": str(e)}
            )
    
    def evaluate_bert_score(
        self, 
        response: str, 
        expected_answer: str
    ) -> EvaluationResult:
        """
        Evaluate BERTScore
        
        Args:
            response: Generated response
            expected_answer: Expected/reference answer
            
        Returns:
            EvaluationResult with BERTScore
        """
        if not BERT_SCORE_AVAILABLE:
            return EvaluationResult(
                metric="bert_score",
                score=0.0,
                details={"reason": "BERTScore not available"}
            )
        
        try:
            # Calculate BERTScore
            P, R, F1 = bert_score([response], [expected_answer], lang="en")
            
            # Use F1 score as the main metric
            f1_score = F1[0].item()
            
            return EvaluationResult(
                metric="bert_score",
                score=f1_score,
                details={
                    "precision": P[0].item(),
                    "recall": R[0].item(),
                    "f1": f1_score
                }
            )
            
        except Exception as e:
            self.logger.error(f"BERTScore calculation failed: {e}")
            return EvaluationResult(
                metric="bert_score",
                score=0.0,
                details={"error": str(e)}
            )
    
    def _calculate_token_overlap(self, text1: str, text2: str) -> float:
        """Calculate token overlap between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Simple tokenization (split by whitespace and punctuation)
        tokens1 = set(re.findall(r'\w+', text1.lower()))
        tokens2 = set(re.findall(r'\w+', text2.lower()))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        # Jaccard similarity
        return len(intersection) / len(union) if union else 0.0
    
    def _evaluate_factual_grounding(self, response: str, context: str) -> float:
        """
        Evaluate how well factual claims in response are supported by context
        This is a simplified implementation - could be enhanced with NER and fact checking
        """
        # Extract potential factual claims (numbers, proper nouns, etc.)
        response_facts = self._extract_facts(response)
        context_facts = self._extract_facts(context)
        
        if not response_facts:
            return 1.0  # No facts to verify
        
        # Check how many response facts are supported by context
        supported_facts = 0
        for fact in response_facts:
            if any(fact.lower() in context_fact.lower() or context_fact.lower() in fact.lower() 
                   for context_fact in context_facts):
                supported_facts += 1
        
        return supported_facts / len(response_facts)
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract potential facts from text (simplified)"""
        facts = []
        
        # Extract numbers
        numbers = re.findall(r'\d+', text)
        facts.extend(numbers)
        
        # Extract capitalized words (potential proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        facts.extend(proper_nouns)
        
        # Extract Bengali numbers and names (basic patterns)
        bengali_numbers = re.findall(r'[০-৯]+', text)
        facts.extend(bengali_numbers)
        
        return facts
    
    def evaluate_rag_response(
        self, 
        query: str,
        response: str,
        retrieved_chunks: List[Dict],
        expected_answer: Optional[str] = None,
        metrics: List[str] = None
    ) -> RAGEvaluationResult:
        """
        Comprehensive evaluation of a RAG response
        
        Args:
            query: User query
            response: Generated response
            retrieved_chunks: Retrieved document chunks
            expected_answer: Expected answer (optional)
            metrics: List of metrics to evaluate (if None, evaluate all available)
            
        Returns:
            RAGEvaluationResult with comprehensive evaluation
        """
        if metrics is None:
            metrics = ["groundedness", "relevance", "semantic_similarity", "rouge", "bleu", "bert_score"]
        
        result = RAGEvaluationResult(
            query=query,
            response=response,
            retrieved_chunks=retrieved_chunks,
            expected_answer=expected_answer
        )
        
        evaluation_details = {}
        
        # Evaluate groundedness
        if "groundedness" in metrics:
            groundedness = self.evaluate_groundedness(response, retrieved_chunks)
            result.groundedness_score = groundedness.score
            evaluation_details["groundedness"] = groundedness.details
        
        # Evaluate relevance
        if "relevance" in metrics:
            relevance = self.evaluate_relevance(query, retrieved_chunks)
            result.relevance_score = relevance.score
            evaluation_details["relevance"] = relevance.details
        
        # Evaluate metrics that require expected answer
        if expected_answer:
            if "semantic_similarity" in metrics:
                semantic_sim = self.evaluate_semantic_similarity(response, expected_answer)
                result.semantic_similarity = semantic_sim.score
                evaluation_details["semantic_similarity"] = semantic_sim.details
            
            if "rouge" in metrics:
                rouge = self.evaluate_rouge(response, expected_answer)
                result.rouge_scores = rouge.details
                evaluation_details["rouge"] = rouge.details
            
            if "bleu" in metrics:
                bleu = self.evaluate_bleu(response, expected_answer)
                result.bleu_score = bleu.score
                evaluation_details["bleu"] = bleu.details
            
            if "bert_score" in metrics:
                bert = self.evaluate_bert_score(response, expected_answer)
                result.bert_score = bert.score
                evaluation_details["bert_score"] = bert.details
        
        # Calculate overall score (weighted average of available metrics)
        scores = []
        weights = []
        
        if result.groundedness_score > 0:
            scores.append(result.groundedness_score)
            weights.append(0.3)  # Groundedness is important
        
        if result.relevance_score > 0:
            scores.append(result.relevance_score)
            weights.append(0.3)  # Relevance is important
        
        if result.semantic_similarity > 0:
            scores.append(result.semantic_similarity)
            weights.append(0.2)
        
        if result.rouge_scores and 'rouge1' in result.rouge_scores:
            scores.append(result.rouge_scores['rouge1'])
            weights.append(0.1)
        
        if result.bleu_score > 0:
            scores.append(result.bleu_score)
            weights.append(0.05)
        
        if result.bert_score > 0:
            scores.append(result.bert_score)
            weights.append(0.05)
        
        # Calculate weighted average
        if scores and weights:
            result.overall_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        result.evaluation_details = evaluation_details
        
        return result
    
    def evaluate_batch(
        self, 
        test_cases: List[Dict],
        metrics: List[str] = None
    ) -> Dict:
        """
        Evaluate a batch of test cases
        
        Args:
            test_cases: List of test cases with query, response, chunks, expected_answer
            metrics: List of metrics to evaluate
            
        Returns:
            Dictionary with batch evaluation results
        """
        individual_results = []
        
        for i, test_case in enumerate(test_cases):
            self.logger.info(f"Evaluating test case {i+1}/{len(test_cases)}")
            
            result = self.evaluate_rag_response(
                query=test_case.get('query', ''),
                response=test_case.get('response', ''),
                retrieved_chunks=test_case.get('chunks', []),
                expected_answer=test_case.get('expected_answer'),
                metrics=metrics
            )
            
            individual_results.append(result)
        
        # Calculate aggregate statistics
        if individual_results:
            aggregate_stats = {
                "total_test_cases": len(individual_results),
                "average_groundedness": np.mean([r.groundedness_score for r in individual_results]),
                "average_relevance": np.mean([r.relevance_score for r in individual_results]),
                "average_semantic_similarity": np.mean([r.semantic_similarity for r in individual_results if r.semantic_similarity > 0]),
                "average_overall_score": np.mean([r.overall_score for r in individual_results]),
                "individual_results": individual_results
            }
            
            # Add ROUGE statistics if available
            rouge1_scores = [r.rouge_scores.get('rouge1', 0) for r in individual_results if r.rouge_scores]
            if rouge1_scores:
                aggregate_stats["average_rouge1"] = np.mean(rouge1_scores)
            
            return aggregate_stats
        else:
            return {"total_test_cases": 0, "individual_results": []}

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create evaluator
    evaluator = RAGEvaluator()
    
    # Example evaluation
    test_response = "শম্ভুনাথকে অনুপম সুপুরুষ বলে মনে করে।"
    expected_answer = "শম্ভুনাথ"
    
    test_chunks = [
        {
            "text": "অনুপম তার জীবনে অনেক মানুষের সাথে পরিচিত হয়েছে। তার মধ্যে শম্ভুনাথ একজন বিশেষ ব্যক্তি। শম্ভুনাথকে অনুপম সুপুরুষ বলে মনে করে।",
            "similarity": 0.85
        }
    ]
    
    test_query = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
    
    # Evaluate
    result = evaluator.evaluate_rag_response(
        query=test_query,
        response=test_response,
        retrieved_chunks=test_chunks,
        expected_answer=expected_answer
    )
    
    print(f"Evaluation Results:")
    print(f"Groundedness: {result.groundedness_score:.3f}")
    print(f"Relevance: {result.relevance_score:.3f}")
    print(f"Semantic Similarity: {result.semantic_similarity:.3f}")
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"ROUGE Scores: {result.rouge_scores}")
    print(f"BLEU Score: {result.bleu_score:.3f}")
    print(f"BERT Score: {result.bert_score:.3f}") 