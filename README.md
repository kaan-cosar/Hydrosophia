# Hydrosophia

Hydrosophia focuses on **GLOFS (Glacial Lake Outburst Floods)** and ways to reduce their damage to people and the environment. This project combines **Machine Learning**, **hydrogel**, and **fluid mechanics** to forecast outbursts and mitigate their effects.

---

## Project Overview

Our approach:

1. **Machine Learning Forecasting:**  
   We use a **Random Forest** classifier with **Self-Training Semi-Supervised Learning** to predict possible outbursts before they occur.

2. **Hydrogel Usage:**  
   Hydrogel is applied to increase water viscosity and slow down flood flow.

3. **Fluid Mechanics Solutions:**  
   Shaped stones and terrain features are used to control and slow water movement.

---

## Data Collection & Feature Engineering

We use multiple data sources:

- **Meteorological Data:** Precipitation, daily mean temperature, daily mean wind.  
- **Lake Surface Area:**  
  - Satellite imagery processed using **image processing techniques** to extract daily lake area (km²)  
  - Volume estimation using an average depth of 5m  
  - Daily area differences to detect rapid changes  

**Data processing steps in code:**

- Standardize dates to daily granularity.  
- Remove outliers from lake area.  
- Fill missing numeric values with column means.

---

## Semi-Supervised Labeling

- Flood date window: **Oct 3 ± 1 day, 2023** labeled as positive (1).  
- Randomly sampled negatives (0) from remaining data.  
- Unlabeled data is kept as -1 for semi-supervised learning.

---

## Model & Evaluation

- **Model:** Random Forest (500 trees, max_depth=5, class_weight=balanced)  
- **Wrapper:** SelfTrainingClassifier with threshold=0.9  
- **Features:** Numeric meteorological and lake area features  
- **Evaluation:** Precision, Recall, Confusion Matrix, ROC-AUC, Precision-Recall curves

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the model
python main.py
