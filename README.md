# SEP-Memory: Memory-Augmented Explainable Stock Predictions

**SEP-Memory** is a research and engineering project that integrates the **Summarize–Explain–Predict (SEP)** forecasting framework with a **multi-layer memory system** to improve both the **accuracy** and **interpretability** of stock predictions.  

The system enhances predictive modeling by combining LLM-driven explanations with structured memory layers, reinforcement learning, and efficient retrieval.

---

## ✨ Features

### 🧠 Multi-Layer Memory Architecture
- **Short-Term Memory**: Daily summaries of news, tweets, and stock movements  
- **Mid-Term Memory**: Refined explanations and self-reflections for correction  
- **Long-Term Memory**: Consolidated high-reward patterns and signals  
- **Reflection Memory**: Error cases and failed predictions for targeted retraining  

### 🔍 Memory-Augmented Explainability
- Retrieve relevant past insights with **FAISS-powered embeddings**  
- Inject memory context into prompts for more **consistent and reliable explanations**  
- Enable **self-reflective correction** loops in explanations  

### ♻️ End-to-End Reinforcement Loop
1. **Summarize** → Ingest daily market data & generate structured summaries  
2. **Explain** → LLM produces reasoning → reflection step refines explanations  
3. **Predict** → PPO/GRPO policy generates trade signals  
4. **Reinforce** → Rewards from real price movements written back into memory layers  

### ⏫ Automated Memory Promotion ("Jump")
- Important knowledge is automatically promoted:  
  _short → mid → long-term_ memory  
- Low-value or stale knowledge is pruned  

---

## 🛠️ Tech Stack
- **LLM Backbone**: Transformers + PEFT (LoRA, 4-bit QLoRA)  
- **Reinforcement Learning**: PPO / GRPO with reward models  
- **Memory System**: Custom `MemoryDB` + `BrainDB` with multi-layer storage  
- **Retrieval**: OpenAI embeddings + FAISS for sub-second lookup  
- **Training Data**: Daily financial news + social media streams  

---

## 📂 Workflow

```mermaid
flowchart TD
    A[Market Data] --> B[Summarize]
    B --> C[Explain v1]
    C --> D[Self-Reflection with Memory]
    D --> E[Explain v2]
    E --> F[Predict with PPO Agent]
    F -->|Rewards| G[Update Long-Term & Reflection Memory]
    G -->|Promote/Prune| B