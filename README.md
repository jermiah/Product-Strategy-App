#  GrandView – Strategic Product Copilot for Retail (Prosol)

**Note:** The raw data and initial EDA notebook are not included in this repository due to confidentiality restrictions.

##  Overview

This repository contains the codebase for **GrandView**, a fully functional, lightweight AI app built during an in-company project with **Prosol (Grand Frais)**. The goal was to create a **Strategic Product Copilot** that transforms raw transactional data into tailored business strategies using:

- Product Segmentation (via unsupervised clustering)
- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)

This tool is designed for retail teams to gain strategic clarity and act on insights — all within a Streamlit app that runs locally.

---

##  Modules in This Repository

### 1. **Data Preparation & Clustering**
> *(Note: Raw data and full EDA files are not shared in this public repository)*

- Performed advanced feature engineering from offline retail transactional data
- Applied unsupervised clustering to build segmentation categories (e.g., timing, loyalty, payment behavior)
- Annotated cluster outputs with interpretable product labels to aid LLM reasoning

### 2. **Strategy Builder (Product-Specific)**
- Generates product-level strategies using segmentation labels, sales metrics, and prompting techniques
- Powered by **FAISS vector search** and **Gemini/Mistral LLMs**

### 3. **Strategy Explorer (General Query Support)**
- Handles open-ended questions like _"Which products need better retention?"_
- Implements RAG to retrieve similar product profiles and injects them into LLM prompts

### 4. **Feedback Vault**
- Collects business expert feedback
- Embeds a feedback-memory loop to continuously improve LLM responses
- Fine-tuning ready

### 5. **Product Export Tool**
- Enables download of filtered product lists by segmentation label and product hierarchy (category, family, sub-family)
- Useful for campaign planning or strategic analysis

### 6. **Maintenance Center**
- Clear cache, reset files, and manage app stability in one place

### 7. **Dual LLM Support**
- Gemini API and Mistral 7B via Hugging Face Transformers

---

##  What Makes GrandView Unique?

- Fully functional without cloud infra – built entirely on a local laptop
- Designed to be adopted by business users without technical training
- Modular design with future integration potential (e.g., Store Review System, Forecasting Modules)

---

##  Next Steps & Roadmap

- Integrate with Prosol’s Store Review (POS Insight) platform
- Connect teammate modules: demand forecasting, price elasticity, and customer segmentation
- Expand into a fully agentic decision-support system

---

##  Acknowledgements

Special thanks to the Prosol Data team and professors at **emlyon business school** for guidance and feedback throughout the project.
