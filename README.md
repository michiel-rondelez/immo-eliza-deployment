# 🏠 Immo Eliza ML Deployment Pipeline

A comprehensive machine learning pipeline for Belgian real estate price prediction with FastAPI backend and Streamlit UI. Features JSON-based model serialization, parameter tuning, and overfitting detection.

## 📋 Features

### Core Functionality
- **Multiple ML Models**: Linear Regression, SVR, Decision Tree, Random Forest, XGBoost
- **JSON Serialization**: All models and preprocessing pipelines stored as pure JSON (no pickle/joblib)
- **Feature Engineering**: Automatic region mapping, outdoor space calculations, luxury scoring
- **Preprocessing Pipeline**: Outlier capping, scaling, encoding with full JSON persistence
- **Overfitting Detection**: Real-time monitoring of train/test performance gaps
- **Parameter Tuning**: Interactive UI for adjusting all model hyperparameters

### API (FastAPI)
- Train models with custom parameters
- Make predictions via REST API
- Check model performance and overfitting status
- Upload training data
- Get feature information

### UI (Streamlit)
- **Data & Training Page**: Upload CSV, configure preprocessing, tune model parameters, train
- **Prediction Page**: Interactive form for property features, multi-model predictions
- **Model Analysis Page**: Visualizations, metrics comparison, overfitting detection
- **Parameters Page**: View and export model configurations

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x run_api.sh run_streamlit.sh
```

### Running the Application

#### Option 1: Streamlit Only (Recommended for beginners)

```bash
# Start Streamlit UI
./run_streamlit.sh
# or
streamlit run streamlit_app.py
```

Open your browser at `http://localhost:8501`

#### Option 2: FastAPI Only

```bash
# Start FastAPI server
./run_api.sh
# or
cd src && python -m uvicorn main:app --reload
```

API docs available at `http://localhost:8000/docs`

#### Option 3: Both (Full Stack)

```bash
# Terminal 1: Start FastAPI
./run_api.sh

# Terminal 2: Start Streamlit
./run_streamlit.sh
```

## 📊 Data Format

Your CSV file should contain the following columns:

### Required Features

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `number_of_rooms` | int | Number of rooms | 3 |
| `living_area` | float | Living area in m² | 100.5 |
| `number_of_facades` | int | Number of facades | 2 |
| `postal_code` | int | Belgian postal code | 1000 |
| `subtype_of_property` | str | Property type | "apartment", "house", "villa" |
| `state_of_building` | str | Building condition | "good", "as_new", "to_renovate" |
| `price` | float | **Target variable** | 250000 |

### Optional Features

| Column | Type | Description |
|--------|------|-------------|
| `garden_surface` | float | Garden area in m² |
| `terrace_surface` | float | Terrace area in m² |
| `equipped_kitchen` | int | Has equipped kitchen (0/1) |
| `furnished` | int | Is furnished (0/1) |
| `open_fire` | int | Has fireplace (0/1) |
| `terrace` | int | Has terrace (0/1) |
| `garden` | int | Has garden (0/1) |
| `swimming_pool` | int | Has pool (0/1) |

See `data/sample_data_structure.csv` for an example.

## 🎯 Usage Guide

### 1. Training Models via Streamlit

1. Navigate to **📊 Data & Training** page
2. Upload your cleaned CSV file
3. Configure preprocessing options:
   - Test size (default: 20%)
   - Cross-validation folds (default: 5)
   - Outlier capping (recommended: enabled)
4. Adjust model parameters in the tabs:
   - **Linear Regression**: No parameters
   - **SVR**: Kernel, C, epsilon, gamma
   - **Decision Tree**: max_depth, min_samples_split, min_samples_leaf
   - **Random Forest**: n_estimators, max_depth, min_samples, max_features
   - **XGBoost**: n_estimators, max_depth, learning_rate, regularization
5. Click **🚀 Train All Models**
6. View training results table

### 2. Making Predictions

1. Navigate to **🎯 Prediction** page
2. Select a trained model
3. Fill in property features:
   - Basic: living area, rooms, facades, postal code
   - Details: property type, building state, outdoor spaces
   - Amenities: kitchen, furnished, fireplace, etc.
4. Click **🔮 Predict Price**
5. View prediction from selected model + all models

### 3. Analyzing Overfitting

1. Navigate to **📈 Model Analysis** page
2. Review performance metrics table
3. Adjust R² gap threshold (default: 0.1)
4. Examine visualizations:
   - Train vs Test RMSE
   - Train vs Test R²
   - Overfitting Gap Analysis
   - Cross-validation RMSE
5. Check overfitting status table
6. Read recommendations for overfitting models

### 4. Using the API

#### Train Models

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "data/properties.csv",
    "test_size": 0.2,
    "cv_folds": 5,
    "use_capping": true,
    "capping_percentiles": [1, 99],
    "model_params": {
      "XGBoost": {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05
      }
    }
  }'
```

#### Make Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "XGBoost",
    "features": {
      "living_area": 120,
      "number_of_rooms": 3,
      "number_of_facades": 2,
      "postal_code": 1000,
      "subtype_of_property": "apartment",
      "state_of_building": "good",
      "garden_surface": 0,
      "terrace_surface": 15,
      "equipped_kitchen": 1,
      "furnished": 0,
      "open_fire": 0,
      "terrace": 1,
      "garden": 0,
      "swimming_pool": 0
    }
  }'
```

#### Check Overfitting

```bash
curl "http://localhost:8000/overfitting?threshold=0.1"
```

## 🏗️ Project Structure

```
immo-eliza-deployment/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Feature preprocessing with JSON serialization
│   ├── models.py              # Model training and evaluation
│   └── main.py                # FastAPI application
├── data/
│   └── sample_data_structure.csv  # Example data format
├── models/                    # Saved models (created after training)
│   ├── preprocessor.json
│   └── trainer.json
├── logs/                      # Training logs (optional)
├── streamlit_app.py           # Streamlit UI
├── requirements.txt           # Python dependencies
├── run_api.sh                 # API startup script
├── run_streamlit.sh           # Streamlit startup script
└── README.md                  # This file
```

## 🔧 Model Parameters Guide

### Decision Tree Regressor

- **max_depth** (2-20): Maximum tree depth. Lower = less overfitting
- **min_samples_split** (2-50): Minimum samples to split a node
- **min_samples_leaf** (1-30): Minimum samples per leaf node

**Overfitting tips**: Decrease max_depth, increase min_samples

### Random Forest Regressor

- **n_estimators** (50-500): Number of trees
- **max_depth** (2-30): Maximum tree depth per tree
- **min_samples_split** (2-30): Minimum samples to split
- **min_samples_leaf** (1-20): Minimum samples per leaf
- **max_features** (sqrt/log2): Features to consider per split

**Overfitting tips**: Decrease max_depth, use "sqrt" for max_features

### XGBoost Regressor

- **n_estimators** (50-500): Number of boosting rounds
- **max_depth** (2-10): Maximum tree depth
- **learning_rate** (0.01-0.3): Step size shrinkage
- **reg_alpha** (0-1): L1 regularization
- **reg_lambda** (0-2): L2 regularization
- **subsample** (0.5-1.0): Fraction of samples per tree
- **colsample_bytree** (0.5-1.0): Fraction of features per tree

**Overfitting tips**: Decrease max_depth, decrease learning_rate, increase reg_lambda

## 📈 Understanding Metrics

- **RMSE** (Root Mean Squared Error): Lower is better. Average prediction error.
- **MAE** (Mean Absolute Error): Lower is better. Average absolute error.
- **R²** (R-squared): 0 to 1, higher is better. Proportion of variance explained.
- **CV RMSE**: Cross-validation RMSE. More reliable than single train/test split.
- **Overfitting Gap**: Test RMSE - Train RMSE. Should be close to 0.
- **R² Gap**: Train R² - Test R². Should be < 0.1 for good generalization.

## 🎓 Overfitting Detection

The pipeline automatically detects overfitting by comparing train and test performance:

### Signs of Overfitting:
- Train R² much higher than Test R² (gap > 0.1)
- Train RMSE much lower than Test RMSE
- High CV standard deviation
- Perfect or near-perfect train metrics

### Solutions:
1. **Reduce model complexity**: Lower max_depth, fewer trees
2. **Add regularization**: Increase reg_alpha, reg_lambda (XGBoost)
3. **Collect more data**: Larger training set
4. **Feature selection**: Remove irrelevant features
5. **Ensemble methods**: Use Random Forest instead of Decision Tree
6. **Cross-validation**: Ensure consistent performance across folds

## 🔒 JSON Serialization

All models and preprocessing pipelines are stored as pure JSON for:
- **Portability**: Easy to share and version control
- **Transparency**: Human-readable model parameters
- **Security**: No pickle deserialization vulnerabilities
- **Cross-platform**: Works across Python versions

Note: Tree-based models (Decision Tree, Random Forest) store tree structure metadata. For production inference with these models, consider:
1. Re-training when loading from JSON
2. Using simpler models (Linear Regression, SVR) for API deployment
3. Implementing custom tree reconstruction logic

## 🐛 Troubleshooting

### "No training data available"
- Upload CSV file on the Data & Training page first
- Ensure CSV has all required columns

### "Models not trained yet"
- Train models on the Data & Training page before prediction
- Wait for training to complete

### High overfitting gap
- Reduce model complexity (see Model Parameters Guide)
- Check if you have enough training data (recommended: 1000+ samples)
- Enable outlier capping in preprocessing

### Poor R² scores
- Check data quality (missing values, outliers)
- Ensure target variable (price) is present and numeric
- Try different model parameter combinations
- Consider feature engineering

## 📝 License

MIT License - Feel free to use and modify for your projects.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional models (LightGBM, CatBoost)
- Hyperparameter optimization (GridSearch, Bayesian)
- Feature importance visualization
- Model explanation (SHAP values)
- Docker deployment
- Database integration

---

Built with ❤️ using FastAPI, Streamlit, scikit-learn, and XGBoost