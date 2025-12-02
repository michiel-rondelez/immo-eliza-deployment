"""
Streamlit UI for ML model training and prediction.
Allows parameter tuning, training, prediction, and overfitting detection.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from preprocessing import FeaturePreprocessor
from models import ModelTrainer, XGBOOST_AVAILABLE

# Page config
st.set_page_config(
    page_title="Immo Eliza ML Pipeline",
    page_icon="🏠",
    layout="wide"
)

# Initialize session state
if "trained" not in st.session_state:
    st.session_state.trained = False
if "preprocessor" not in st.session_state:
    st.session_state.preprocessor = None
if "trainer" not in st.session_state:
    st.session_state.trainer = None
if "results" not in st.session_state:
    st.session_state.results = None
if "data" not in st.session_state:
    st.session_state.data = None


def main():
    """Main Streamlit application."""
    st.title("🏠 Immo Eliza ML Pipeline")
    st.markdown("**Train models, tune parameters, and predict property prices**")

    # Show XGBoost availability warning
    if not XGBOOST_AVAILABLE:
        st.error("⚠️ XGBoost is not installed! Install it with: `pip install xgboost`")
        st.info("The application will work with other models (Linear Regression, SVR, Decision Tree, Random Forest)")

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["📊 Data & Training", "🎯 Prediction", "📈 Model Analysis", "⚙️ Model Parameters"]
    )

    if page == "📊 Data & Training":
        data_and_training_page()
    elif page == "🎯 Prediction":
        prediction_page()
    elif page == "📈 Model Analysis":
        analysis_page()
    elif page == "⚙️ Model Parameters":
        parameters_page()


def data_and_training_page():
    """Page for data upload and model training."""
    st.header("📊 Data Upload & Model Training")

    # File upload
    st.subheader("1. Upload Training Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df

        st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns")

        # Show data preview
        with st.expander("Preview Data"):
            st.dataframe(df.head(10))
            st.write(f"**Shape:** {df.shape}")
            st.write(f"**Columns:** {', '.join(df.columns)}")

    # Training configuration
    st.subheader("2. Training Configuration")

    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)

    with col2:
        use_capping = st.checkbox("Use Outlier Capping", value=True)
        if use_capping:
            lower_percentile = st.number_input("Lower Percentile", 0.0, 10.0, 1.0)
            upper_percentile = st.number_input("Upper Percentile", 90.0, 100.0, 99.0)
        else:
            lower_percentile, upper_percentile = 1, 99

    # Model parameters
    st.subheader("3. Model Parameters")

    model_params = {}

    # Get default params
    default_params = ModelTrainer.DEFAULT_PARAMS

    # Create tabs for each model
    tab_names = ["Linear Regression", "SVR", "Decision Tree", "Random Forest"]
    if XGBOOST_AVAILABLE:
        tab_names.append("XGBoost")

    tabs = st.tabs(tab_names)

    # Linear Regression
    with tabs[0]:
        st.markdown("**Linear Regression** - No hyperparameters to tune")
        model_params["Linear Regression"] = {}

    # SVR
    with tabs[1]:
        st.markdown("**Support Vector Regression**")
        col1, col2 = st.columns(2)
        with col1:
            kernel = st.selectbox(
                "Kernel",
                ["rbf", "linear", "poly"],
                index=0,
                help="Kernel function: 'rbf' (radial basis) for non-linear relationships, 'linear' for linear relationships, 'poly' for polynomial relationships"
            )
            C = st.number_input(
                "C (Regularization)",
                0.01, 10.0,
                default_params["SVR"]["C"],
                0.1,
                help="Regularization strength (inverse). Lower values = stronger regularization, higher values = less regularization. Controls the trade-off between model complexity and training error."
            )
        with col2:
            epsilon = st.number_input(
                "Epsilon",
                0.01, 1.0,
                0.1,
                0.01,
                help="Epsilon-tube parameter. Defines a margin of tolerance where no penalty is given to errors. Larger values = more tolerance for errors."
            )
            gamma = st.selectbox(
                "Gamma",
                ["scale", "auto"],
                index=0,
                help="Kernel coefficient. 'scale' uses 1/(n_features * X.var()), 'auto' uses 1/n_features. Controls how far the influence of a single training example reaches."
            )

        model_params["SVR"] = {
            "kernel": kernel,
            "C": float(C),
            "epsilon": float(epsilon),
            "gamma": gamma
        }

    # Decision Tree
    with tabs[2]:
        st.markdown("**Decision Tree Regressor**")
        col1, col2 = st.columns(2)
        with col1:
            dt_max_depth = st.slider(
                "Max Depth",
                2, 20,
                default_params["Decision Tree"]["max_depth"],
                help="Maximum depth of the tree. Deeper trees can capture more complex patterns but may overfit. Lower values prevent overfitting."
            )
            dt_min_samples_split = st.slider(
                "Min Samples Split",
                2, 50,
                default_params["Decision Tree"]["min_samples_split"],
                help="Minimum number of samples required to split an internal node. Higher values prevent overfitting by requiring more samples before making a split."
            )
        with col2:
            dt_min_samples_leaf = st.slider(
                "Min Samples Leaf",
                1, 30,
                default_params["Decision Tree"]["min_samples_leaf"],
                help="Minimum number of samples required to be at a leaf node. Higher values create smoother models and prevent overfitting by ensuring each leaf represents enough samples."
            )
            dt_random_state = st.number_input(
                "Random State",
                0, 100,
                42,
                help="Seed for random number generation. Use the same value for reproducible results across multiple runs."
            )

        model_params["Decision Tree"] = {
            "max_depth": dt_max_depth,
            "min_samples_split": dt_min_samples_split,
            "min_samples_leaf": dt_min_samples_leaf,
            "random_state": int(dt_random_state)
        }

    # Random Forest
    with tabs[3]:
        st.markdown("**Random Forest Regressor**")
        col1, col2 = st.columns(2)
        with col1:
            rf_n_estimators = st.slider(
                "Number of Trees",
                50, 500,
                default_params["Random Forest"]["n_estimators"],
                50,
                help="Number of decision trees in the forest. More trees generally improve performance but increase training time. Typical values: 100-500."
            )
            rf_max_depth = st.slider(
                "Max Depth",
                2, 30,
                default_params["Random Forest"]["max_depth"],
                help="Maximum depth of each tree. Deeper trees can model more complex patterns but may overfit. None means nodes expand until all leaves are pure."
            )
            rf_min_samples_split = st.slider(
                "Min Samples Split",
                2, 30,
                default_params["Random Forest"]["min_samples_split"],
                help="Minimum samples required to split a node. Higher values prevent overfitting by requiring more evidence before creating splits."
            )
        with col2:
            rf_min_samples_leaf = st.slider(
                "Min Samples Leaf",
                1, 20,
                default_params["Random Forest"]["min_samples_leaf"],
                help="Minimum samples required at each leaf node. Higher values create smoother decision boundaries and prevent overfitting."
            )
            rf_max_features = st.selectbox(
                "Max Features",
                ["sqrt", "log2", None],
                index=0,
                help="Number of features to consider when looking for the best split. 'sqrt' uses sqrt(n_features), 'log2' uses log2(n_features), None uses all features. Lower values increase diversity between trees."
            )
            rf_random_state = st.number_input(
                "Random State",
                0, 100,
                42,
                key="rf_rs",
                help="Seed for random number generation. Use the same value for reproducible results across multiple runs."
            )

        model_params["Random Forest"] = {
            "n_estimators": rf_n_estimators,
            "max_depth": rf_max_depth,
            "min_samples_split": rf_min_samples_split,
            "min_samples_leaf": rf_min_samples_leaf,
            "max_features": rf_max_features,
            "random_state": int(rf_random_state)
        }

    # XGBoost (conditionally shown)
    if XGBOOST_AVAILABLE:
        with tabs[4]:
            st.markdown("**XGBoost Regressor**")
            col1, col2 = st.columns(2)
            with col1:
                xgb_n_estimators = st.slider(
                    "Number of Trees",
                    50, 500,
                    default_params["XGBoost"]["n_estimators"],
                    50,
                    help="Number of boosting rounds (trees). More trees can improve performance but may overfit and increase training time. Combine with lower learning rate for better results."
                )
                xgb_max_depth = st.slider(
                    "Max Depth",
                    2, 10,
                    default_params["XGBoost"]["max_depth"],
                    help="Maximum depth of each tree. Deeper trees capture more complex patterns but may overfit. Typical values: 3-10. Lower values help prevent overfitting."
                )
                xgb_learning_rate = st.slider(
                    "Learning Rate",
                    0.01, 0.3,
                    default_params["XGBoost"]["learning_rate"],
                    0.01,
                    help="Step size shrinkage to prevent overfitting. Lower values (0.01-0.1) make the model more robust but require more trees. Also called 'eta'."
                )
                xgb_subsample = st.slider(
                    "Subsample",
                    0.5, 1.0,
                    default_params["XGBoost"]["subsample"],
                    0.1,
                    help="Fraction of samples used for each tree. Values < 1.0 introduce randomness and prevent overfitting. Typical values: 0.6-0.9."
                )
            with col2:
                xgb_reg_alpha = st.slider(
                    "L1 Regularization (Alpha)",
                    0.0, 1.0,
                    default_params["XGBoost"]["reg_alpha"],
                    0.1,
                    help="L1 regularization term on weights. Increases model simplicity and sparsity. Higher values = more regularization. Use to prevent overfitting."
                )
                xgb_reg_lambda = st.slider(
                    "L2 Regularization (Lambda)",
                    0.0, 2.0,
                    default_params["XGBoost"]["reg_lambda"],
                    0.1,
                    help="L2 regularization term on weights. Smooths feature weights and prevents overfitting. Default is 1.0. Higher values = more regularization."
                )
                xgb_colsample_bytree = st.slider(
                    "Column Subsample",
                    0.5, 1.0,
                    default_params["XGBoost"]["colsample_bytree"],
                    0.1,
                    help="Fraction of features (columns) used when constructing each tree. Values < 1.0 add randomness and prevent overfitting. Typical values: 0.6-0.9."
                )
                xgb_random_state = st.number_input(
                    "Random State",
                    0, 100,
                    42,
                    key="xgb_rs",
                    help="Seed for random number generation. Use the same value for reproducible results across multiple runs."
                )

            model_params["XGBoost"] = {
                "n_estimators": xgb_n_estimators,
                "max_depth": xgb_max_depth,
                "learning_rate": float(xgb_learning_rate),
                "reg_alpha": float(xgb_reg_alpha),
                "reg_lambda": float(xgb_reg_lambda),
                "subsample": float(xgb_subsample),
                "colsample_bytree": float(xgb_colsample_bytree),
                "random_state": int(xgb_random_state),
                "verbosity": 0
            }

    # Train button
    st.subheader("4. Train Models")

    if st.button("🚀 Train All Models", type="primary"):
        if st.session_state.data is None:
            st.error("Please upload data first!")
        else:
            train_models(
                st.session_state.data,
                model_params,
                test_size,
                cv_folds,
                use_capping,
                (lower_percentile, upper_percentile)
            )


def train_models(df, model_params, test_size, cv_folds, use_capping, capping_percentiles):
    """Train all models with given parameters."""
    with st.spinner("Training models..."):
        try:
            # Initialize preprocessor
            preprocessor = FeaturePreprocessor(
                use_capping=use_capping,
                capping_percentiles=capping_percentiles
            )

            # Fit and transform
            X, y = preprocessor.fit_transform(df)

            st.info(f"Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")

            # Initialize trainer
            trainer = ModelTrainer(model_params=model_params)

            # Split data
            trainer.train_test_split_data(X, y, test_size=test_size)

            # Train all models
            results = trainer.train_all_models(cv_folds=cv_folds)

            # Save to session state
            st.session_state.preprocessor = preprocessor
            st.session_state.trainer = trainer
            st.session_state.results = results
            st.session_state.trained = True

            # Save to disk
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            preprocessor.save(model_dir / "preprocessor.json")
            trainer.save(model_dir / "trainer.json")

            st.success("✅ Models trained successfully!")

            # Show quick results
            results_df = trainer.get_results_dataframe()
            st.dataframe(results_df)

        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def prediction_page():
    """Page for making predictions."""
    st.header("🎯 Make Predictions")

    if not st.session_state.trained:
        st.warning("Please train models first on the Data & Training page!")
        return

    # Model selection
    model_name = st.selectbox(
        "Select Model",
        list(st.session_state.trainer.models.keys())
    )

    st.subheader("Enter Property Features")

    # Create input form
    col1, col2, col3 = st.columns(3)

    features = {}

    with col1:
        st.markdown("**Basic Features**")
        features["living_area"] = st.number_input("Living Area (m²)", 20, 1000, 100)
        features["number_of_rooms"] = st.number_input("Number of Rooms", 1, 20, 3)
        features["number_of_facades"] = st.number_input("Number of Facades", 1, 4, 2)
        features["postal_code"] = st.number_input("Postal Code", 1000, 9999, 1000)

    with col2:
        st.markdown("**Property Details**")
        features["subtype_of_property"] = st.selectbox(
            "Property Subtype",
            ["house", "apartment", "villa", "bungalow", "duplex", "studio"]
        )
        features["state_of_building"] = st.selectbox(
            "Building State",
            ["good", "as_new", "to_renovate", "just_renovated"]
        )
        features["garden_surface"] = st.number_input("Garden Surface (m²)", 0, 2000, 0)
        features["terrace_surface"] = st.number_input("Terrace Surface (m²)", 0, 200, 0)

    with col3:
        st.markdown("**Amenities**")
        features["equipped_kitchen"] = int(st.checkbox("Equipped Kitchen", value=True))
        features["furnished"] = int(st.checkbox("Furnished"))
        features["open_fire"] = int(st.checkbox("Open Fire"))
        features["terrace"] = int(st.checkbox("Terrace"))
        features["garden"] = int(st.checkbox("Garden"))
        features["swimming_pool"] = int(st.checkbox("Swimming Pool"))

    # Predict button
    if st.button("🔮 Predict Price", type="primary"):
        try:
            # Convert to DataFrame
            df_pred = pd.DataFrame([features])

            # Preprocess
            X_pred = st.session_state.preprocessor.transform(df_pred)

            # Predict
            y_pred = st.session_state.trainer.predict(model_name, X_pred)

            # Inverse transform
            price = st.session_state.preprocessor.inverse_transform_target(y_pred)[0]

            # Display prediction
            st.success(f"### Predicted Price: €{price:,.2f}")

            # Show predictions from all models
            st.subheader("Predictions from All Models")

            all_predictions = {}
            for name, model in st.session_state.trainer.models.items():
                y = model.predict(X_pred)
                p = st.session_state.preprocessor.inverse_transform_target(y)[0]
                all_predictions[name] = p

            pred_df = pd.DataFrame.from_dict(
                all_predictions,
                orient='index',
                columns=['Predicted Price (€)']
            )
            pred_df['Predicted Price (€)'] = pred_df['Predicted Price (€)'].apply(lambda x: f"€{x:,.2f}")

            st.dataframe(pred_df)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")


def analysis_page():
    """Page for model analysis and overfitting detection."""
    st.header("📈 Model Analysis & Overfitting Detection")

    if not st.session_state.trained:
        st.warning("Please train models first on the Data & Training page!")
        return

    trainer = st.session_state.trainer
    results = st.session_state.results

    # Results table
    st.subheader("Model Performance Metrics")
    results_df = trainer.get_results_dataframe()
    st.dataframe(results_df.style.highlight_max(axis=0, subset=['train_r2', 'test_r2']))

    # Best model
    best_model_name, best_result = trainer.get_best_model(metric="test_r2")
    st.info(f"**Best Model:** {best_model_name} (Test R² = {best_result['test_r2']:.4f})")

    # Overfitting detection
    st.subheader("Overfitting Analysis")

    threshold = st.slider("R² Gap Threshold", 0.0, 0.3, 0.1, 0.01)
    overfitting_status = trainer.detect_overfitting(threshold=threshold)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Train vs Test RMSE
    ax1 = axes[0, 0]
    models = list(results.keys())
    train_rmse = [results[m]['train_rmse'] for m in models]
    test_rmse = [results[m]['test_rmse'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width/2, train_rmse, width, label='Train RMSE', alpha=0.8)
    ax1.bar(x + width/2, test_rmse, width, label='Test RMSE', alpha=0.8)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Train vs Test RMSE')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. R² Comparison
    ax2 = axes[0, 1]
    train_r2 = [results[m]['train_r2'] for m in models]
    test_r2 = [results[m]['test_r2'] for m in models]

    ax2.bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8)
    ax2.bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('R² Score')
    ax2.set_title('Train vs Test R²')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Overfitting Gap
    ax3 = axes[1, 0]
    r2_gaps = [results[m]['r2_gap'] for m in models]
    colors = ['red' if gap > threshold else 'green' for gap in r2_gaps]

    ax3.bar(models, r2_gaps, color=colors, alpha=0.7)
    ax3.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    ax3.set_xlabel('Model')
    ax3.set_ylabel('R² Gap (Train - Test)')
    ax3.set_title('Overfitting Gap Analysis')
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Cross-validation RMSE
    ax4 = axes[1, 1]
    cv_rmse = [results[m]['cv_rmse'] for m in models]
    cv_std = [results[m]['cv_std'] for m in models]

    ax4.bar(models, cv_rmse, yerr=cv_std, capsize=5, alpha=0.7)
    ax4.set_xlabel('Model')
    ax4.set_ylabel('CV RMSE')
    ax4.set_title('Cross-Validation RMSE (with std)')
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Overfitting status table
    st.subheader("Overfitting Status")

    status_data = []
    for model_name, status in overfitting_status.items():
        status_data.append({
            "Model": model_name,
            "Is Overfitting": "⚠️ YES" if status["is_overfitting"] else "✅ NO",
            "R² Gap": f"{status['r2_gap']:.4f}",
            "Train R²": f"{status['train_r2']:.4f}",
            "Test R²": f"{status['test_r2']:.4f}"
        })

    status_df = pd.DataFrame(status_data)
    st.dataframe(status_df)

    # Recommendations
    st.subheader("Recommendations")

    for model_name, status in overfitting_status.items():
        if status["is_overfitting"]:
            st.warning(f"**{model_name}** is overfitting. Consider:")
            st.markdown("""
            - Reducing model complexity (decrease max_depth, n_estimators)
            - Increasing regularization (increase reg_alpha, reg_lambda for XGBoost)
            - Collecting more training data
            - Using feature selection to reduce dimensionality
            """)


def parameters_page():
    """Page to view and export model parameters."""
    st.header("⚙️ Model Parameters")

    if not st.session_state.trained:
        st.warning("Please train models first on the Data & Training page!")
        return

    trainer = st.session_state.trainer

    st.subheader("Current Model Parameters")

    for model_name, params in trainer.model_params.items():
        with st.expander(f"{model_name} Parameters"):
            st.json(params)

    # Export parameters
    st.subheader("Export Configuration")

    if st.button("📥 Download Parameters as JSON"):
        config = {
            "model_parameters": trainer.model_params,
            "preprocessing": {
                "use_capping": st.session_state.preprocessor.use_capping,
                "capping_percentiles": st.session_state.preprocessor.capping_percentiles
            }
        }

        json_str = json.dumps(config, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="model_config.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()
