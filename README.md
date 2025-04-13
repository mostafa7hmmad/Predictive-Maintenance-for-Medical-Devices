# 🚀 Multi-Class Classification Using ML and ANN 📈  

![header](healthcare.wep) *<!-- Replace with your actual header image -->*  

This project focuses on classifying tabular data into three distinct classes using a combination of traditional machine learning models and an Artificial Neural Network (ANN). The dataset consists of samples with 11 features, and the pipeline leverages clustering, feature engineering, and model optimization to achieve robust predictions.  

The entire workflow includes:  
- **Data preprocessing** (cleaning, standardization)  
- **Clustering** with K-Means to explore data structure  
- **Model training** with ML classifiers and ANN  
- **Evaluation** using comprehensive metrics  

---

## 📊 Data Processing Pipeline  

### **1️⃣ Data Collection**  
- The dataset comprises tabular data with **11 features** and a target variable with **3 classes**.  

### **2️⃣ Data Exploration**  
- **Statistical summary**: Analyzing feature distributions, correlations, and outliers.  
- **Class distribution**: Checking balance across the three classes.  

### **3️⃣ Data Preprocessing**  
- **Cleaning**: Handling missing values (if any).  
- **Standardization**: Scaling features to zero mean and unit variance.  
- **Encoding**: Converting the target variable into one-hot format for ANN training.  

### **4️⃣ Clustering**  
- **K-Means**: Grouping samples into 3 clusters to explore natural data patterns.  
- **Analysis**: Comparing cluster assignments with actual labels for insights.  

---

## 🧠 Model Architectures  

### **Traditional ML Models**  
A variety of classifiers are trained and tuned:  
- **Naive Bayes**: `GaussianNB` with `var_smoothing=1e-8`.  
- **Decision Tree**: Tuned with `max_depth=5`, `min_samples_split=5`, `min_samples_leaf=3`.  
- **Random Forest**: Ensemble with `n_estimators=50`, `max_depth=5`, `max_features='log2'`.  
- **SVM**: Using `rbf` kernel, `C=10.0`, `gamma='auto'`.  

### **Artificial Neural Network (ANN)**  
- **Input Layer**: 11 features  
- **Hidden Layers**:  
  - `Dense(64, activation='relu')` → `Dropout(0.3)`  
  - `Dense(32, activation='relu')` → `Dropout(0.2)`  
- **Output Layer**: `Dense(3, activation='softmax')`  
- **Optimizers Tested**:  
  - `RMSProp` (lr=0.001)  
  - `SGD with Momentum` (lr=0.01, momentum=0.9)  
  - `Adam` (lr=0.001)  
- **Training**: 50 epochs, batch size 32, `categorical_crossentropy` loss.  

---

## 📈 Evaluation  
- **Metrics**:  
  - **Accuracy**: Overall correctness of predictions.  
  - **Recall**: Macro-averaged for multi-class sensitivity.  
  - **Precision**: Macro averaged for prediction reliability.  
  - **F1-Score**: Harmonic mean of precision and recall.  
  - **Specificity**: Macro-averaged true negative rate.  
- **Comparison**: Performance assessed across all models (ML and ANN) on the test set to identify the best approach.  

