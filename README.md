# CS5228 Singapore HDB Resale Price Prediction (Team 2)

This project predicts Singapore HDB resale prices using machine learning models. The solution involves extensive data preprocessing, feature engineering, and ensemble modeling using CatBoost, LightGBM, and XGBoost. The dataset used in this report is available in our GitHub repository at [link]

## Project Overview

The project consists of three main phases:
1. **Data Preprocessing and Feature Engineering** - Three Jupyter notebooks handle different aspects of data preparation
2. **Model Training and Validation** - Python scripts for training and cross-validation
3. **Prediction Generation** - Script to generate final predictions for test data

## Environment Configuration

### Prerequisites
- Python 3.8 or higher
- Git (for version control)

### Required Dependencies

Install the required packages using pip:

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost category-encoders joblib geopy tqdm matplotlib seaborn
```

### GPU Support (Optional but Recommended)
The training scripts are configured to use GPU acceleration by default. To enable GPU support:

- **For LightGBM**: Install with GPU support
  ```bash
  pip uninstall lightgbm
  pip install lightgbm --install-option=--gpu
  ```

- **For XGBoost**: Ensure CUDA is installed and use the GPU-enabled version
  ```bash
  pip install xgboost-gpu
  ```

- **For CatBoost**: GPU support is built-in, but requires CUDA toolkit

If you don't have GPU access, modify the training scripts (`train_full.py` and `validate_kfold.py`) to set `use_gpu=False`.

## Project Structure

```
├── README.md                    # Project documentation
├── CS5228_Final_Report.pdf      # Final project report
├── data/                       # Processed training and test data
│   ├── train_encoded.csv       # Preprocessed training data
│   └── test_encoded.csv        # Preprocessed test data
├── au-data/                    # Auxiliary economic data
│   ├── COE.csv                 # Certificate of Entitlement data
│   ├── CPI.csv                 # Consumer Price Index data
│   ├── GDP.csv                 # GDP growth rate data
│   ├── SORA.csv                # Singapore Overnight Rate Average data
│   └── STI.csv                 # Straits Times Index data
│
├── EDA_Data_processing_Encoding.ipynb    # Data cleaning and basic feature engineering
├── EDA_Tem_Eco.ipynb                     # Temporal and economic feature engineering
├── EDA_Tem_Geo.ipynb                     # Geospatial feature engineering
│
├── models.py                   # Model definitions and base classes
├── validate_kfold.py           # K-fold cross-validation script
├── train_full.py               # Full dataset training script
└── predict.py                  # Prediction generation script
```



## How to Run the Project

### Step 1: Data Preprocessing
Run the three Jupyter notebooks in sequence to prepare your data:

1. **Basic Data Processing**:
   ```bash
   jupyter notebook EDA_Data_processing_Encoding.ipynb
   ```

2. **Temporal and Economic Features**:
   ```bash
   jupyter notebook EDA_Tem_Eco.ipynb
   ```

3. **Geospatial Features**:
   ```bash
   jupyter notebook EDA_Tem_Geo.ipynb
   ```

> **Note**: Make sure you have the required auxiliary data in the `au-data/` directory and HDB block details in `auxiliary-data/` directory before running the notebooks.

### Step 2: Model Validation (Optional but Recommended)
Perform K-fold cross-validation to evaluate model performance and determine optimal training iterations:

```bash
python validate_kfold.py
```

This will:
- Perform 5-fold cross-validation
- Log feature importances and model performance
- Determine optimal iteration counts for each model
- Save validation results to `logs/validation_log.txt`

### Step 3: Full Model Training
Train the final ensemble models on the complete dataset:

```bash
python train_full.py
```

This will:
- Train CatBoost, LightGBM, and XGBoost models
- Save trained models to the `models/` directory
- Use iteration counts from validation (or hardcoded values if validation wasn't run)

### Step 4: Generate Predictions
Create predictions for the test dataset:

```bash
python predict.py
```

This will:
- Load the trained models
- Generate ensemble predictions (average of all three models)
- Save results to `results/predictions_v{VERSION}.csv`
