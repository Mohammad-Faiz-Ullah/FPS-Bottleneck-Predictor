# 🎮 FPS Bottleneck Predictor (ML-Powered)

### 🚀 Live Demo: [Click Here to Predict FPS](https://fps-bottleneck-predictor-ttjwzgmrzefpbmo4bj8nzc.streamlit.app/)

## 📌 Overview
This project is a Machine Learning tool designed to estimate gaming performance and identify hardware bottlenecks (CPU vs GPU) for PC builders. It analyzes distinct hardware specifications rather than relying on generic model names.

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
