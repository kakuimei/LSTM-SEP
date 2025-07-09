# SEP-Memory: Enhance Explainable Stock Predictions with Multi-Layer Memory

This project combines the **Summarize–Explain–Predict (SEP)** framework with a custom multi-layer memory system (MemoryDB & BrainDB) to deliver both accurate and human-readable stock predictions. Leveraging daily news & social media data, SEP-Memory adds:

- **Short-term**, **Mid-term**, **Long-term** & **Reflection** memories  
- **Memory-augmented self-reflection prompts**  
- **PPO-driven reinforcement** with memory feedback loops  
- **Automated memory “jump”** for pattern consolidation  

---

## 🚀 Key Features

1. **Multi-Layer Memory**  
   - _Short-term_: Daily summaries & market moves  
   - _Mid-term_: Self-reflection corrections & refined explanations  
   - _Long-term_: High-reward patterns & robust signals  
   - _Reflection_: Error cases for targeted retraining  

2. **Memory-Augmented Explainability**  
   - Retrieve relevant past summaries & reflections  
   - Inject context into LLM prompts for sharper, more reliable explanations  

3. **End-to-End Reinforcement Loop**  
   - **Predict** with PPO policy network  
   - Compute real-world rewards (price change or portfolio return)  
   - Write back successes/failures into memory layers  

4. **Automated Memory “Jump”**  
   - Hot items automatically promoted from short → mid/long based on importance  
   - Low-value items demoted or pruned  

5. **OpenAI Embeddings + FAISS** for sub-second retrieval

---

- **Summarize**: ingest & compress daily text → add to _short-term_  
- **Explain**: LLM v1 → self-reflect with memory context → v2 → add to _mid-term_  
- **Predict**: PPO agent reads combined state (summary + explanations + retrieved memory) → outputs trade signals → writes outcomes to _long-term_ or _reflection_

---

## 📂 Installation & Setup

```bash
git clone https://github.com/your-org/sep-memory.git
cd sep-memory
pip install -r requirements.txt
