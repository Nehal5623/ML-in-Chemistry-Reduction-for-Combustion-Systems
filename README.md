# 🔬 ML-in-Chemistry: Reduction for Combustion Systems

Developed ML models for real-time prediction of species concentrations in **Well-Stirred** and **Constant Volume Reactors**.  
Used **PCA** for dimensionality reduction and trained **ANN**, **XGBoost**, and **Gradient Boosting**, achieving **R² > 0.99** and up to **10× speedup**.

---

## 📁 1. Data Collection

### 1.1 Well-Stirred Reactor

Simulating combustion in a well-stirred reactor using **Cantera** and the `gri30.yaml` mechanism.  
Tracking species evolution and enthalpy over time.  
CSV file (`WSR.csv`) generated includes:

- Time (s)
- Temperature (K)
- Initial and Final Enthalpy
- Mole fractions for 30+ chemical species

---

## 🧪 2. Dimensionality Reduction

To manage high-dimensional outputs (many species), we apply **Principal Component Analysis (PCA)**:

- Standardized mole fraction outputs using `StandardScaler`
- Reduced dimensions from 30+ species to configurable `n_components` (default: 10)
- Retained ~99% variance using top components

---

## 🤖 3. Machine Learning Models

Trained three models on PCA-reduced species data:

### 🔧 3.1 HistGradientBoostingRegressor (with MultiOutput)
- Supports custom tuning of:
  - Learning rate
  - Max depth
  - Min samples per leaf
- Achieved `R² > 0.99` and low MSE on test data
- Serialized with `joblib` → `wsr_model.pkl`

### 🔬 3.2 ANN (TensorFlow / Keras)
- Fully connected network with:
  - Input → 128 → 64 → Output layers
  - ReLU activation, dropout layers
  - Adam optimizer with early stopping
- Saved as `wsr_ann_model.h5`

### 🌲 3.3 XGBoost
- Trained individual regressors for each principal component
- Parameters controlled via interactive widgets
- Saved as `wsr_xgb_model.pkl`

---

## 📊 4. User Interface (via `ipywidgets`)

Interactive widgets allow:
- Hyperparameter tuning for each model
- Real-time predictions by entering `Time (s)` and `Temperature (K)`
- Display of **Top 5 species** by predicted mole fraction

---

## 📈 5. Results

Each model displays:
- Training, validation, and test R² and MSE
- Example prediction output:

```text
Top 5 Species by Mole Fraction:
CO2: 4.310e-01
H2O: 3.120e-01
N2: 2.800e-01
O2: 1.500e-02
CH4: 1.000e-03
