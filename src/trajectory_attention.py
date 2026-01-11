# Trajectory Attention Pooling for Classification
# Copyright 2026 - Kharagpur Data Science Hackathon

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from config import ModelConfig


class TrajectoryAttentionPooling(nn.Module):
    """
    Trajectory Attention Pooling for aggregating temporal representations.
    
    Computes attention-weighted pooling over a sequence of representations,
    producing a fixed-size output suitable for classification.
    
    Key Features:
    - Learnable query for attention computation
    - Multi-head attention for diverse feature extraction
    - Positional weighting for trajectory-aware pooling
    """
    
    def __init__(self, config: ModelConfig, num_classes: int = 2):
        super().__init__()
        self.config = config
        D = config.n_embd
        nh = config.n_head
        
        # Learnable query for attention
        self.query = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)
        self.out_proj = nn.Linear(D, D)
        
        self.n_head = nh
        self.head_dim = D // nh
        
        # Trajectory position encoding
        self.position_weight = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(D // 2, num_classes)
        )
        
        # "Fuel Injection": Dynamic Query Generation
        self.dynamic_query = True
        self.query_generator = nn.Sequential(
            nn.Linear(D, D),
            nn.LayerNorm(D),
            nn.Tanh()
        )
        
        # Attention temperature
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_pooled: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, T, D] sequence of representations
            attention_mask: [B, T] optional mask (1 for valid, 0 for padding)
            return_attention: whether to return attention weights
            return_pooled: if True, return pooled representation instead of logits
            
        Returns:
            output: [B, D] pooled if return_pooled else [B, num_classes] logits
            attention_weights: [B, nh, T] if return_attention=True
        """
        B, T, D = x.size()
        nh = self.n_head
        head_dim = self.head_dim
        
        # "Fuel Injection": Generate dynamic query from context
        if self.dynamic_query:
            # Context is global average pooling of input sequence
            context = x.mean(dim=1)  # [B, D]
            generated_query = self.query_generator(context).unsqueeze(1)  # [B, 1, D]
            
            # Combine static learnable bias with dynamic content
            query = (self.query + generated_query).expand(B, 1, D)
        else:
            # Expand learnable query
            query = self.query.expand(B, 1, D)  # [B, 1, D]
        
        # Project Q, K, V
        Q = self.q_proj(query).view(B, 1, nh, head_dim).transpose(1, 2)  # [B, nh, 1, head_dim]
        K = self.k_proj(x).view(B, T, nh, head_dim).transpose(1, 2)  # [B, nh, T, head_dim]
        V = self.v_proj(x).view(B, T, nh, head_dim).transpose(1, 2)  # [B, nh, T, head_dim]
        
        # Compute attention scores
        scale = (head_dim ** -0.5) * self.temperature
        attn_scores = (Q @ K.transpose(-2, -1)) * scale  # [B, nh, 1, T]
        
        # Apply trajectory position weighting
        positions = torch.linspace(0, 1, T, device=x.device).view(1, T, 1)
        pos_weights = self.position_weight(positions).squeeze(-1)  # [1, T]
        attn_scores = attn_scores + pos_weights.unsqueeze(0).unsqueeze(0).log()
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, nh, 1, T]
        
        # Apply attention to values
        pooled = (attn_weights @ V).squeeze(2)  # [B, nh, head_dim]
        pooled = pooled.transpose(1, 2).reshape(B, D)  # [B, D]
        pooled = self.out_proj(pooled.unsqueeze(1)).squeeze(1)  # [B, D]
        
        # Return pooled representation or classification logits
        if return_pooled:
            output = pooled
        else:
            output = self.classifier(pooled)  # [B, num_classes]
        
        if return_attention:
            return output, attn_weights.squeeze(2)  # [B, nh, T]
        return output, None


class HierarchicalTrajectoryPooling(nn.Module):
    """
    Hierarchical pooling that combines trajectory attention at multiple scales.
    Designed to work with MultiScaleBDH outputs.
    """
    
    def __init__(self, config: ModelConfig, num_classes: int = 2):
        super().__init__()
        self.config = config
        D = config.n_embd
        
        # Separate pooling for each scale
        self.sentence_pooling = TrajectoryAttentionPooling(config, num_classes=D)
        self.paragraph_pooling = TrajectoryAttentionPooling(config, num_classes=D)
        self.chapter_pooling = TrajectoryAttentionPooling(config, num_classes=D)
        
        # Cross-scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(D * 3, D * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(D * 2, D),
            nn.LayerNorm(D)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(D // 2, num_classes)
        )
        
    def forward(
        self,
        sentence_repr: torch.Tensor,
        paragraph_repr: torch.Tensor,
        chapter_repr: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            sentence_repr: [B, T_s, D] sentence-level representations
            paragraph_repr: [B, T_p, D] paragraph-level representations
            chapter_repr: [B, T_c, D] chapter-level representations
            
        Returns:
            logits: [B, num_classes] classification logits
            attention_dict: attention weights for each scale
        """
        # Pool each scale
        sent_pooled, sent_attn = self.sentence_pooling(sentence_repr, attention_mask, return_attention=True)
        para_pooled, para_attn = self.paragraph_pooling(paragraph_repr, None, return_attention=True)
        chap_pooled, chap_attn = self.chapter_pooling(chapter_repr, None, return_attention=True)
        
        # Fuse across scales
        combined = torch.cat([sent_pooled, para_pooled, chap_pooled], dim=-1)
        fused = self.scale_fusion(combined)
        
        # Final classification
        logits = self.classifier(fused)
        
        attention_dict = {
            'sentence': sent_attn,
            'paragraph': para_attn,
            'chapter': chap_attn
        }
        
        return logits, attention_dict


class EvidenceExtractor(nn.Module):
    """
    Extracts supporting sentences based on attention weights.
    Works with TrajectoryAttentionPooling to identify key evidence.
    """
    
    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k
        
    def forward(
        self,
        attention_weights: torch.Tensor,
        token_to_sentence_map: torch.Tensor,
        sentences: List[str]
    ) -> List[Tuple[int, str, float]]:
        """
        Extract top-k sentences based on attention weights.
        
        Args:
            attention_weights: [B, nh, T] attention weights
            token_to_sentence_map: [T] mapping from token idx to sentence idx
            sentences: list of original sentences
            
        Returns:
            List of (sentence_idx, sentence_text, attention_score) tuples
        """
        # Average attention across heads
        avg_attn = attention_weights.mean(dim=1)  # [B, T]
        
        # Aggregate attention by sentence
        max_sent_idx = token_to_sentence_map.max().item() + 1
        sentence_scores = torch.zeros(avg_attn.size(0), max_sent_idx, device=avg_attn.device)
        
        for i in range(max_sent_idx):
            mask = (token_to_sentence_map == i)
            if mask.any():
                sentence_scores[:, i] = avg_attn[:, mask].sum(dim=-1)
        
        # Normalize
        sentence_scores = sentence_scores / sentence_scores.sum(dim=-1, keepdim=True)
        
        # Get top-k sentences
        results = []
        for b in range(sentence_scores.size(0)):
            scores = sentence_scores[b]
            top_indices = torch.topk(scores, min(self.top_k, len(sentences))).indices
            
            batch_results = []
            for idx in top_indices:
                idx = idx.item()
                if idx < len(sentences):
                    batch_results.append((idx, sentences[idx], scores[idx].item()))
            results.append(batch_results)
        
        return results
