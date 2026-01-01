# 🎮 FPS Bottleneck Predictor (ML-Powered)

### 🚀 [Launch Live App](https://fps-bottleneck-predictor-ttjwzgmrzefpbmo4bj8nzc.streamlit.app/)

## 📌 Overview
An end-to-end Machine Learning application that predicts gaming FPS based on raw hardware specifications (CPU/GPU) and identifies system bottlenecks. unlike generic calculators, this tool analyzes technical architecture (Cores, CUDA Cores, VRAM, Bandwidth) to provide engineering-grade estimates.


## 📂 Project Structure & Workflow
This repository is organized into the three phases of the Data Science lifecycle:

### 1️⃣ Phase 1: Research & Analysis (`/notebooks`)
* **[01_EDA_and_Research.ipynb](notebooks/01_EDA_and_Research.ipynb):** * **Objective:** Validated the "Hardware vs. FPS" hypothesis.
    * **Key Findings:** Discovered "Menu Screen Bias" (outliers > 4000 FPS) and implemented a Winsorization strategy to handle server-grade CPU outliers.
    * **Tech:** Pandas, Matplotlib, Seaborn.

### 2️⃣ Phase 2: Pipeline Engineering (`/notebooks`)
* **[02_Model_Training.ipynb](notebooks/02_Model_Training.ipynb):** * **Model:** Random Forest Regressor (Scikit-Learn).
    * **Optimization:** Achieved ~26 FPS RMSE. Compressing model artifacts by 60% (Joblib) for cloud deployment.
    * **ETL:** Automated regex-based cleaning for resolution mapping (e.g., '1080p' -> 'FHD').

### 3️⃣ Phase 3: Production Deployment (`Root`)
* **`app.py`**: The full-stack Streamlit application.
* **Features:**
    * Real-time FPS Inference.
    * Dynamic Bottleneck Detection Algorithm (CPU vs GPU load analysis).
    * Interactive Hardware Lookup Engine.


## 🔧 Technical Stack
* **Data Science:** Random Forest Regressor (Scikit-Learn), Pandas for complex data mapping.
* **Engineering:** Streamlit for frontend, Python for backend logic.
* **Data Strategy:** * Implemented a **Regex-based Resolution Mapper** to handle inconsistent dataset labeling.
    * Built a **Hardware Lookup Engine** to map user-friendly names (e.g., "RTX 2060") to raw technical specs (FP32 Performance, Bandwidth, Core Counts).
    * Engineered a custom **Translation Layer** to sanitize legacy dataset naming conventions.

## 💡 Business Impact
For hardware manufacturers (NVIDIA/AMD) or retailers:
* **Reduces Return Rates:** Helps consumers buy the correct matching components, reducing returns due to "bottlenecking" or poor performance.
* **Upsell Opportunity:** The tool automatically detects bottlenecks and suggests if a CPU upgrade is needed to unlock full GPU potential.


## 🔧 Installation
```bash
pip install -r requirements.txt
streamlit run app.py
