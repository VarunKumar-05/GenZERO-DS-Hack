# BDH Narrative Consistency Classifier
## Comprehensive Project Report

> **Track B**: BDH-Driven Continuous Narrative Reasoning  
> **Hackathon**: Kharagpur Data Science Hackathon 2026  
> **Date**: January 2026

---

## Executive Summary

This project implements a **BDH (Brain-inspired Dynamic Hierarchical) narrative consistency classifier** that determines whether hypothetical character backstories are consistent with long-form narratives (novels). The system incorporates multi-scale temporal dynamics, Hebbian learning checkpoints, and LLM-powered reasoning to extract supporting evidence.

---

## 1. Overall Approach

### 1.1 Problem Statement

Given:
- A **backstory** claim about a character
- A **long-form narrative** (complete novel, 50K-500K tokens)

Determine:
- **Classification**: Consistent or Inconsistent
- **Evidence**: Supporting/contradicting sentences from the narrative
- **Explanation**: Reasoning for the classification

### 1.2 Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BDH NARRATIVE CLASSIFIER                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   SENTENCE   │    │  PARAGRAPH   │    │   CHAPTER    │          │
│  │   ENCODER    │    │   ENCODER    │    │   ENCODER    │          │
│  │  (Fine-grain)│    │ (Med-grain)  │    │(Coarse-grain)│          │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
│         │                   │                   │                   │
│         └─────────┬─────────┴─────────┬─────────┘                   │
│                   ▼                   ▼                             │
│            ┌─────────────────────────────────┐                      │
│            │     CROSS-SCALE FUSION          │                      │
│            │   (Learnable Weighted Sum)      │                      │
│            └───────────────┬─────────────────┘                      │
│                            ▼                                        │
│            ┌─────────────────────────────────┐                      │
│            │  TRAJECTORY ATTENTION POOLING   │                      │
│            │   (Position-aware Aggregation)  │                      │
│            └───────────────┬─────────────────┘                      │
│                            ▼                                        │
│  ┌──────────────────┐     │     ┌──────────────────────┐           │
│  │ HEBBIAN CHECKPOINT│◄────┼────►│  LANGGRAPH + GEMINI  │           │
│  │    MANAGER        │     │     │    REASONING         │           │
│  └──────────────────┘     ▼     └──────────────────────┘           │
│                    ┌─────────────┐                                  │
│                    │ CLASSIFIER  │                                  │
│                    └──────┬──────┘                                  │
│                           ▼                                         │
│              [Classification + Evidence + Rationale]                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Multi-scale temporal encoding | Novels have structure at sentence, paragraph, and chapter levels |
| Hebbian checkpoints | Track layer evolution without storing full states |
| Trajectory attention | Position-aware pooling preserves narrative order |
| LLM-assisted reasoning | Hard to verify backstory consistency with pure neural models |
| TOON format | Human-readable checkpoints for cross-session memory |

---

## 2. Handling Long Context

### 2.1 The Challenge

Novels range from **50,000 to 500,000+ tokens**—far exceeding typical transformer context windows (2K-8K tokens).

### 2.2 Our Multi-Scale Hierarchical Approach

```
Novel Text (500K tokens)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                   HIERARCHICAL CHUNKING                        │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  CHAPTER LEVEL (1024 tokens/chunk)                            │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐           │
│  │ C1  │ C2  │ C3  │ C4  │ C5  │ ... │ Cn  │     │           │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘           │
│                                                                │
│  PARAGRAPH LEVEL (256 tokens/chunk)                           │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐           │
│  │P1│P2│P3│P4│P5│P6│P7│P8│P9│..│..│..│..│..│Pn│  │           │
│  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘           │
│                                                                │
│  SENTENCE LEVEL (64 tokens/chunk)                             │
│  ┌┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┬┐   │
│  │S1│S2│S3│...│Sn│                                       │   │
│  └┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┴┘   │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### 2.3 Technical Implementation

```python
# TextProcessor extracts hierarchy
sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
paragraphs = re.split(r'\n\s*\n', text)
chapters = re.split(r'Chapter\s+\d+', text)  # Pattern-based

# Each scale has dedicated BDH encoder
sentence_encoder = TemporalScaleBDH(config, scale_name="sentence")
paragraph_encoder = TemporalScaleBDH(config, scale_name="paragraph")
chapter_encoder = TemporalScaleBDH(config, scale_name="chapter")

# Fusion with learnable weights
scale_weights = nn.Parameter(torch.ones(3) / 3)  # [0.33, 0.33, 0.33]
fused = w[0]*sent + w[1]*para + w[2]*chap
```

### 2.4 Attention-Based Relevance Filtering

Instead of processing entire novels, we use **backstory-guided filtering**:

1. **Heuristic alignment**: Find sentences with word overlap to backstory
2. **Attention weighting**: Use trajectory attention to rank passages
3. **Top-k selection**: Process only most relevant 50-100 sentences

---

## 3. Distinguishing Causal Signals from Noise

### 3.1 The Signal-Noise Problem

| Signal | Noise |
|--------|-------|
| "He was born in Paris" | General descriptions of Paris |
| "She lost her father at age 10" | Other references to fathers |
| "He worked as a sailor" | Mentions of other sailors |

### 3.2 Our Multi-Stage Filtering Approach

```
Stage 1: LEXICAL FILTERING
─────────────────────────
Backstory: "learned navigation from a ship captain"

Filter by keywords: {captain, navigation, ship, learn}
         ↓
     500 → 50 candidates

Stage 2: SEMANTIC ALIGNMENT
───────────────────────────
BackstoryAligner computes:
- Word overlap score
- Entity matching (character names)
- Temporal consistency

         ↓
     50 → 15 candidates

Stage 3: NEURAL ATTENTION
─────────────────────────
TrajectoryAttentionPooling:
- Cross-attention between backstory and narrative
- Position-aware weighting (beginning/end matter more)
- Multi-head diverse perspectives

         ↓
     15 → 8 verified

Stage 4: LLM VERIFICATION
─────────────────────────
GeminiReasoner.verify_evidence():
"Is this sentence relevant to verifying the backstory?"

         ↓
     8 → 5 final evidence
```

### 3.3 Hebbian Learning for Pattern Recognition

The Hebbian learning rule helps the model learn **co-activation patterns**:

```python
# Hebbian update: "neurons that fire together wire together"
delta_weight = learning_rate * (pre @ post.T) - decay * weight

# This strengthens connections for:
# - Backstory mentions → Character references
# - Temporal markers → Event descriptions
# - Location names → Activity descriptions
```

### 3.4 Trajectory Attention Mechanics

```python
# Position-weighted attention
positions = torch.linspace(0, 1, T)  # 0=start, 1=end
pos_weights = sigmoid(MLP(positions))  # Learn importance

# Combine content and position
attn_scores = (Q @ K.T) * scale + log(pos_weights)
```

---

## 4. Key Limitations and Failure Cases

### 4.1 Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Context window** | Can't process entire novel at once | Hierarchical chunking + filtering |
| **Implicit information** | Misses unstated implications | LLM reasoning helps |
| **Temporal reasoning** | Struggles with complex timelines | Multi-scale encoding partially helps |
| **Character disambiguation** | Confuses characters with same name | Entity tracking (future work) |
| **Sarcasm/irony** | Takes statements literally | LLM can sometimes detect |

### 4.2 Failure Case Examples

```
❌ FAILURE: Subtle contradiction
   Backstory: "He never learned to swim"
   Narrative: "He dove into the lake without hesitation"
   Problem: Model may not connect swimming → diving

❌ FAILURE: Temporal inconsistency  
   Backstory: "Born in 1820, he witnessed the 1789 revolution"
   Problem: Requires arithmetic reasoning

❌ FAILURE: Character confusion
   Backstory about "Captain Grant"
   Narrative mentions "the captain" (different person)
   Problem: Coreference resolution needed

✅ SUCCESS: Direct contradiction
   Backstory: "He was an only child"
   Narrative: "His brother helped him escape"
   Success: Clear lexical contradiction detected
```

### 4.3 Confidence Calibration

We provide confidence scores, but they should be interpreted as:

| Confidence | Interpretation |
|------------|----------------|
| 90-100% | Strong evidence found |
| 70-90% | Moderate evidence, likely correct |
| 50-70% | Weak evidence, uncertain |
| <50% | Insufficient evidence |

---

## 5. Special Features

### 5.1 TOON (Token-Oriented Object Notation)

**Purpose**: Human-readable format for persistent memory across sessions.

```toon
# BDH Checkpoint - Session 20260111
# Generated: 2026-01-11T20:16:04

[metadata]
session_id = "training_v3"
created_at = "2026-01-11T20:16:04"
total_checkpoints = 50
current_step = 5000
layers_tracked = 54

[[checkpoints]]
index = 49
step = 5000
timestamp = 1768142764.93
tensor_file = "cp_training_v3_5000.pt"
loss = 0.342
accuracy = 0.875

[[deltas]]
from_step = 4900
to_step = 5000
# Layer changes (L2 norm of delta)
sentence-layer0 = 0.023456
paragraph-layer2 = 0.018234
chapter-layer5 = 0.041892
```

**Benefits**:
- Human-readable (vs binary pickle)
- Git-friendly (text diffs)
- Easy to parse in any language
- Supports both metadata and references to binary files

### 5.2 Memory Retention System

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY RETENTION SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   SESSION 1                    SESSION 2                        │
│   ─────────                    ─────────                        │
│   ┌─────────┐                  ┌─────────┐                      │
│   │ Train   │                  │ Resume  │                      │
│   │ Model   │                  │ Training│                      │
│   └────┬────┘                  └────┬────┘                      │
│        │                            │                            │
│        ▼                            ▼                            │
│   ┌─────────┐      Read       ┌─────────┐                      │
│   │ Save    │ ─────────────►  │ Load    │                      │
│   │ .toon   │                 │ .toon   │                      │
│   └────┬────┘                 └────┬────┘                      │
│        │                           │                            │
│        ▼                           ▼                            │
│   checkpoint_v1.toon  ───────►  Resume from step 5000          │
│   tensors/cp_v1_5000.pt         Load layer states              │
│                                                                  │
│   DELTA TRACKING:                                               │
│   ─────────────────                                             │
│   Only store CHANGES between checkpoints                        │
│   Full state = Base + Σ(deltas)                                 │
│   Compression: zlib on delta tensors                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Hebbian Checkpoint Features

| Feature | Description |
|---------|-------------|
| **Delta compression** | Store only weight changes, not full states |
| **Trajectory tracking** | Monitor how each layer evolves over training |
| **TOON export** | Human-readable checkpoint summaries |
| **State reconstruction** | Rebuild any past state from base + deltas |
| **Layer analysis** | Identify which layers change most |

```python
# Save checkpoint
manager.save_checkpoint(layer_states, step=1000, metadata={'loss': 0.5})

# Get layer trajectory
trajectory = manager.get_layer_trajectory("sentence_encoder.layer0")
# Returns: [(step, {'weight': norm, 'bias': norm}), ...]

# Export to TOON
save_checkpoint_as_toon(manager, "checkpoints", "session_01")

# Resume later
manager = load_checkpoint_from_toon("checkpoint_session_01.toon", config)
```

### 5.4 LangGraph Reasoning Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH REASONING FLOW                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   INPUT                                                          │
│   ─────                                                          │
│   • Backstory text                                               │
│   • Candidate evidence (from attention)                          │
│   • Character name                                               │
│                                                                   │
│           ▼                                                       │
│   ┌───────────────┐                                              │
│   │ filter_evidence│  Remove obvious non-matches                 │
│   └───────┬───────┘                                              │
│           ▼                                                       │
│   ┌───────────────┐                                              │
│   │verify_evidence │  Gemini: "Is this relevant?"                │
│   │               │  Returns: (yes/no, explanation, confidence)  │
│   └───────┬───────┘                                              │
│           ▼                                                       │
│   ┌───────────────┐                                              │
│   │   classify    │  Gemini: "Consistent or contradicts?"        │
│   │               │  Considers all verified evidence             │
│   └───────┬───────┘                                              │
│           ▼                                                       │
│   ┌───────────────┐                                              │
│   │  summarize    │  Generate human-readable explanation         │
│   └───────┬───────┘                                              │
│           ▼                                                       │
│   OUTPUT                                                          │
│   ──────                                                          │
│   • classification: "consistent" / "inconsistent"                │
│   • confidence: 0.0 - 1.0                                        │
│   • evidence: List[Evidence]                                     │
│   • summary: Explanation string                                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

### 5.5 Dynamic Architecture Upgrades

We implemented three key "car upgrades" to enhance model dynamics:

#### 1. "Fuel Injection" (Dynamic Queries)
- **Problem**: Static attention queries miss context-specific evidence.
- **Solution**: Dynamically generate query vectors from the input sequence summary.
- **Effect**: Attention mechanism adapts its "search strategy" based on the narrative content.

#### 2. "Better Suspension" (Homeostatic Plasticity)
- **Problem**: Long sequences causing vanishing/exploding activity ("bumpy ride").
- **Solution**: `HomeostaticNorm` layer scales activations to maintain target sparsity (10%).
- **Effect**: Stabilizes learning in deep SNN-like architectures.

#### 3. "Automatic Transmission" (Dynamic Gating)
- **Problem**: Fixed weights [0.33, 0.33, 0.33] for temporal scales are suboptimal.
- **Solution**: `GatingNetwork` (MLP) dynamically predicts fusion weights.
- **Effect**: Model automatically "shifts gears" to focus on sentence, paragraph, or chapter level as needed.

---

## 6. Technical Architecture Diagram

### 6.1 Complete System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     BDH NARRATIVE CONSISTENCY CLASSIFIER                      ║
║                         Complete System Architecture                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                           INPUT LAYER                                    │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                          │ ║
║  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │ ║
║  │  │  train.csv   │    │  test.csv    │    │     Books/*.txt          │  │ ║
║  │  │  (80 rows)   │    │  (60 rows)   │    │  (Novel text files)      │  │ ║
║  │  └──────┬───────┘    └──────┬───────┘    └────────────┬─────────────┘  │ ║
║  │         │                   │                         │                 │ ║
║  │         └───────────────────┴─────────────────────────┘                 │ ║
║  │                             │                                           │ ║
║  └─────────────────────────────┼───────────────────────────────────────────┘ ║
║                                ▼                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                        DATA PIPELINE LAYER                               │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                          │ ║
║  │  ┌────────────────────┐        ┌────────────────────────────────────┐  │ ║
║  │  │ NarrativeDataPipeline│       │         TextProcessor              │  │ ║
║  │  │                    │        │                                    │  │ ║
║  │  │ • load_train_csv() │        │ • extract_sentences()             │  │ ║
║  │  │ • load_test_csv()  │        │ • extract_paragraphs()            │  │ ║
║  │  │ • prepare_examples()│        │ • extract_chapters()              │  │ ║
║  │  │ • load_book()      │        │                                    │  │ ║
║  │  └─────────┬──────────┘        └────────────────┬───────────────────┘  │ ║
║  │            │                                    │                       │ ║
║  │            └────────────────┬───────────────────┘                       │ ║
║  │                             │                                           │ ║
║  │  ┌──────────────────────────┴───────────────────────────────────────┐  │ ║
║  │  │                    NarrativeExample                               │  │ ║
║  │  │  • id, book_name, character, backstory, label                    │  │ ║
║  │  │  • processed_narrative (sentences, paragraphs, chapters)         │  │ ║
║  │  └──────────────────────────────────────────────────────────────────┘  │ ║
║  │                                                                          │ ║
║  └─────────────────────────────┬───────────────────────────────────────────┘ ║
║                                ▼                                              ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                         ENCODING LAYER                                   │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                          │ ║
║  │  ┌────────────────────────── MultiScaleBDH ─────────────────────────┐  │ ║
║  │  │                                                                   │  │ ║
║  │  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │  │ ║
║  │  │   │TemporalScale │  │TemporalScale │  │TemporalScale │          │  │ ║
║  │  │   │    BDH       │  │    BDH       │  │    BDH       │          │  │ ║
║  │  │   │  (sentence)  │  │ (paragraph)  │  │  (chapter)   │          │  │ ║
║  │  │   │              │  │              │  │              │          │  │ ║
║  │  │   │ window=64    │  │ window=256   │  │ window=1024  │          │  │ ║
║  │  │   │ 6 BDH layers │  │ 6 BDH layers │  │ 6 BDH layers │          │  │ ║
║  │  │   │ RoPE attn    │  │ RoPE attn    │  │ RoPE attn    │          │  │ ║
║  │  │   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │  │ ║
║  │  │          │                 │                 │                   │  │ ║
║  │  │          │    ┌────────────┴────────────┐    │                   │  │ ║
║  │  │          └────┤   Scale Fusion Layer    ├────┘                   │  │ ║
║  │  │               │  (learnable weights)    │                        │  │ ║
║  │  │               │  [0.33, 0.33, 0.33]     │                        │  │ ║
║  │  │               └────────────┬────────────┘                        │  │ ║
║  │  │                            │                                     │  │ ║
║  │  └────────────────────────────┼─────────────────────────────────────┘  │ ║
║  │                               │                                         │ ║
║  │                  ┌────────────┴────────────┐                            │ ║
║  │                  │   fused_output [B,T,D]  │                            │ ║
║  │                  └────────────┬────────────┘                            │ ║
║  │                               │                                         │ ║
║  └───────────────────────────────┼─────────────────────────────────────────┘ ║
║                                  ▼                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                        POOLING & ATTENTION LAYER                         │ ║
║  ├─────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                          │ ║
║  │  ┌───────────────────────────────────────────────────────────────────┐  │ ║
║  │  │              TrajectoryAttentionPooling                            │  │ ║
║  │  │                                                                    │  │ ║
║  │  │   • Learnable query vector                                        │  │ ║
║  │  │   • Multi-head attention (4 heads)                                │  │ ║
║  │  │   • Position-aware weighting                                      │  │ ║
║  │  │   • Output: [B, D] pooled representation                          │  │ ║
║  │  │                                                                    │  │ ║
║  │  └───────────────────────────────────────────────────────────────────┘  │ ║
║  │                               │                                          │ ║
║  │     ┌─────────────────────────┼─────────────────────────┐               │ ║
║  │     │                         │                         │               │ ║
║  │     ▼                         ▼                         ▼               │ ║
║  │  ┌──────────┐          ┌────────────┐          ┌─────────────────┐     │ ║
║  │  │ Backstory│          │ Narrative  │          │ Cross-Attention │     │ ║
║  │  │ Pooled   │──────────│  Pooled    │──────────│  B → N          │     │ ║
║  │  └──────────┘          └────────────┘          └─────────────────┘     │ ║
║  │                                                                          │ ║
║  └─────────────────────────────┬───────────────────────────────────────────┘ ║
║                                │                                              ║
║          ┌─────────────────────┼─────────────────────┐                       ║
║          ▼                     ▼                     ▼                       ║
║  ┌──────────────┐     ┌──────────────┐     ┌───────────────────────────────┐║
║  │   TRAINING   │     │ CHECKPOINT   │     │         REASONING             │║
║  │    PATH      │     │   SYSTEM     │     │           LAYER               │║
║  ├──────────────┤     ├──────────────┤     ├───────────────────────────────┤║
║  │              │     │              │     │                               │║
║  │ Fusion Layer │     │ Hebbian      │     │  ┌─────────────────────────┐ │║
║  │      ↓       │     │ Checkpoint   │     │  │      LangGraph Flow     │ │║
║  │ Linear(D*2,D)│     │ Manager      │     │  │                         │ │║
║  │      ↓       │     │              │     │  │  filter → verify →      │ │║
║  │ Linear(D, 2) │     │ • save()     │     │  │  classify → summarize   │ │║
║  │      ↓       │     │ • retrieve() │     │  │                         │ │║
║  │   logits     │     │ • trajectory │     │  └───────────┬─────────────┘ │║
║  │              │     │              │     │              │               │║
║  │ CrossEntropy │     │ ┌──────────┐ │     │  ┌───────────▼─────────────┐ │║
║  │    Loss      │     │ │   TOON   │ │     │  │    GeminiReasoner       │ │║
║  │              │     │ │  Export  │ │     │  │                         │ │║
║  │              │     │ └──────────┘ │     │  │  • verify_evidence()    │ │║
║  └──────────────┘     └──────────────┘     │  │  • classify_consistency │ │║
║                                            │  │  • Model: gemini-2.0    │ │║
║                                            │  └─────────────────────────┘ │║
║                                            │                               │║
║                                            └───────────────┬───────────────┘║
║                                                            │                 ║
║  ┌─────────────────────────────────────────────────────────┼────────────────┐║
║  │                          OUTPUT LAYER                   │                │║
║  ├─────────────────────────────────────────────────────────┼────────────────┤║
║  │                                                         ▼                │║
║  │  ┌───────────────────────────────────────────────────────────────────┐  │║
║  │  │                      ReasoningResult                               │  │║
║  │  │                                                                    │  │║
║  │  │  classification: "consistent" | "inconsistent"                    │  │║
║  │  │  confidence: 0.0 - 1.0                                            │  │║
║  │  │  evidence: List[Evidence]                                         │  │║
║  │  │  summary: "The backstory is consistent because..."                │  │║
║  │  │                                                                    │  │║
║  │  └───────────────────────────────────────────────────────────────────┘  │║
║  │                                    │                                     │║
║  │                                    ▼                                     │║
║  │                         ┌──────────────────┐                            │║
║  │                         │   results.csv    │                            │║
║  │                         │                  │                            │║
║  │                         │ Story ID, Pred,  │                            │║
║  │                         │ Rationale        │                            │║
║  │                         └──────────────────┘                            │║
║  │                                                                          │║
║  └──────────────────────────────────────────────────────────────────────────┘║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 6.2 Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPONENT INTERACTIONS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                           ┌─────────────┐                                   │
│                           │   Config    │                                   │
│                           │   (config.py)│                                   │
│                           └──────┬──────┘                                   │
│                                  │                                          │
│           ┌──────────────────────┼──────────────────────┐                   │
│           │                      │                      │                   │
│           ▼                      ▼                      ▼                   │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐           │
│  │  data_pipeline  │   │  multi_scale_bdh │   │ reasoning_graph │           │
│  │                 │   │                 │   │                 │           │
│  │ PathwayConfig   │   │ ModelConfig     │   │ GeminiConfig    │           │
│  │ Books loading   │   │ TemporalConfig  │   │ API key         │           │
│  │ TOON converter  │   │                 │   │ temperature     │           │
│  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘           │
│           │                     │                     │                     │
│           │                     ▼                     │                     │
│           │            ┌─────────────────┐            │                     │
│           │            │trajectory_attn  │            │                     │
│           │            │                 │            │                     │
│           │            │ Pooling layer   │            │                     │
│           │            │ Evidence extract│            │                     │
│           │            └────────┬────────┘            │                     │
│           │                     │                     │                     │
│           │         ┌───────────┴───────────┐         │                     │
│           │         │                       │         │                     │
│           ▼         ▼                       ▼         ▼                     │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │                     train_classifier                        │           │
│  │                                                             │           │
│  │  NarrativeClassifier:                                       │           │
│  │  • encoder = MultiScaleBDH                                  │           │
│  │  • backstory_encoder = MultiScaleBDH                        │           │
│  │  • pooling = TrajectoryAttentionPooling                     │           │
│  │  • cross_attention = MultiheadAttention                     │           │
│  │  • fusion = Linear layers                                   │           │
│  │                                                             │           │
│  │  Training loop:                                             │           │
│  │  • Forward pass → loss → backward → optimizer               │           │
│  │  • Checkpoint at intervals                                  │           │
│  └─────────────────────────────────────────────────────────────┘           │
│           │                                                                 │
│           │                    ┌─────────────────────────┐                 │
│           └───────────────────►│   hebbian_checkpoint    │                 │
│                                │                         │                 │
│                                │ • save_checkpoint()     │                 │
│                                │ • delta compression     │                 │
│                                │ • TOON export           │                 │
│                                │ • trajectory analysis   │                 │
│                                └─────────────────────────┘                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Performance Metrics

| Metric | Value |
|--------|-------|
| Model parameters | 908M (full) / 298K (lightweight) |
| Training examples | 80 |
| Test examples | 60 |
| Supported books | 2 (extensible) |
| Inference speed | ~3-5 sec/example with Gemini |
| TOON export time | <1 sec |

---

## 8. Future Improvements

1. **Coreference resolution** - Better handle "he/she/they" references
2. **Temporal reasoning** - Add explicit date/time parsing
3. **Full novel encoding** - Use sliding window for complete coverage
4. **Fine-tuned embeddings** - Train on narrative-specific corpus
5. **Ensemble methods** - Combine multiple model predictions
6. **Active learning** - Prioritize uncertain examples for labeling

---

## Appendix: File Reference

| File | Lines | Purpose |
|------|-------|---------|
| config.py | 88 | Configuration classes |
| multi_scale_bdh.py | 258 | Multi-scale BDH encoder |
| hebbian_checkpoint.py | 612 | Checkpoint + TOON |
| trajectory_attention.py | 253 | Pooling + evidence |
| text_processor.py | 228 | Text chunking |
| data_pipeline.py | 477 | Data + TOON convert |
| reasoning_graph.py | 345 | LangGraph + Gemini |
| train_classifier.py | 439 | Training pipeline |
| inference.py | 280 | Inference pipeline |
| utils.py | 120 | Utilities |
