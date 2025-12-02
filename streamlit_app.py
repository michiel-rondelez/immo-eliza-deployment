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
    """Page for making predictions with live updates."""
    st.header("🎯 Live Price Predictions")

    if not st.session_state.trained:
        st.warning("Please train models first on the Data & Training page!")
        return

    st.markdown("### 🏠 Adjust Property Features")
    st.markdown("*Predictions update automatically as you change values - watch the prices change live!*")

    # Create input form with sliders for better interactivity
    col1, col2, col3 = st.columns(3)

    features = {}

    with col1:
        st.markdown("**📐 Basic Features**")
        features["living_area"] = st.slider("Living Area (m²)", 20, 500, 100, 5,
                                            help="Total living space in square meters")
        features["number_of_rooms"] = st.slider("Number of Rooms", 1, 10, 3,
                                                help="Total number of rooms")
        features["number_of_facades"] = st.slider("Number of Facades", 1, 4, 2,
                                                  help="Number of building facades")
        features["postal_code"] = st.number_input("Postal Code", 1000, 9999, 1000,
                                                   help="Belgian postal code")

    with col2:
        st.markdown("**🏡 Property Details**")
        features["subtype_of_property"] = st.selectbox(
            "Property Subtype",
            ["house", "apartment", "villa", "bungalow", "duplex", "studio"],
            help="Type of property"
        )
        features["state_of_building"] = st.selectbox(
            "Building State",
            ["good", "as_new", "to_renovate", "just_renovated"],
            help="Overall condition of the building"
        )
        features["garden_surface"] = st.slider("Garden Surface (m²)", 0, 500, 0, 10,
                                               help="Garden area in square meters")
        features["terrace_surface"] = st.slider("Terrace Surface (m²)", 0, 100, 0, 5,
                                                help="Terrace area in square meters")

    with col3:
        st.markdown("**✨ Amenities**")
        features["equipped_kitchen"] = int(st.checkbox("Equipped Kitchen", value=True))
        features["furnished"] = int(st.checkbox("Furnished"))
        features["open_fire"] = int(st.checkbox("Open Fire"))
        features["terrace"] = int(st.checkbox("Terrace"))
        features["garden"] = int(st.checkbox("Garden"))
        features["swimming_pool"] = int(st.checkbox("Swimming Pool"))

    # Perform prediction automatically (no button needed)
    try:
        # Convert to DataFrame
        df_pred = pd.DataFrame([features])

        # Preprocess
        X_pred = st.session_state.preprocessor.transform(df_pred)

        # Get predictions from all models
        all_predictions = {}
        for name, model in st.session_state.trainer.models.items():
            y = model.predict(X_pred)
            p = st.session_state.preprocessor.inverse_transform_target(y)[0]
            all_predictions[name] = p

        # Display results
        st.markdown("---")
        st.markdown("## 💰 Predicted Prices")

        # Best model prediction (highlighted)
        best_model_name, best_result = st.session_state.trainer.get_best_model(metric="test_r2")
        best_price = all_predictions[best_model_name]

        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.metric(
                label=f"🏆 Best Model: {best_model_name}",
                value=f"€{best_price:,.0f}",
                help=f"Prediction from the best-performing model (R² = {best_result['test_r2']:.4f})"
            )
        with col2:
            min_price = min(all_predictions.values())
            st.metric("Lowest Estimate", f"€{min_price:,.0f}")
        with col3:
            max_price = max(all_predictions.values())
            st.metric("Highest Estimate", f"€{max_price:,.0f}")
        with col4:
            mean_price = np.mean(list(all_predictions.values()))
            st.metric("Average", f"€{mean_price:,.0f}")

        # Save prediction to JSON
        st.markdown("### 💾 Save Prediction")

        # Prepare prediction data for JSON export
        prediction_data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "input_features": features,
            "predictions": {
                model_name: {
                    "price_eur": float(price),
                    "model_performance": {
                        "test_r2": float(st.session_state.results[model_name]['test_r2']),
                        "test_rmse": float(st.session_state.results[model_name]['test_rmse'])
                    }
                }
                for model_name, price in all_predictions.items()
            },
            "best_model": best_model_name,
            "best_prediction_eur": float(best_price),
            "price_statistics": {
                "mean": float(mean_price),
                "median": float(np.median(list(all_predictions.values()))),
                "min": float(min_price),
                "max": float(max_price),
                "std": float(np.std(list(all_predictions.values())))
            }
        }

        json_str = json.dumps(prediction_data, indent=2)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.download_button(
                label="📥 Download Prediction as JSON",
                data=json_str,
                file_name=f"property_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Download current prediction with all model results"
            )
        with col2:
            with st.expander("Preview JSON"):
                st.code(json_str, language='json')

        # Visualize predictions from all models
        st.markdown("### 📊 Predictions by Model")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart
        models = list(all_predictions.keys())
        prices = list(all_predictions.values())
        colors = ['#FFD700' if m == best_model_name else '#1f77b4' for m in models]

        bars = ax1.barh(models, prices, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Predicted Price (€)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Model', fontsize=11, fontweight='bold')
        ax1.set_title('Price Predictions Comparison', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2,
                    f'€{width:,.0f}', ha='left', va='center', fontsize=9, fontweight='bold')

        # Model performance vs prediction
        model_r2 = [st.session_state.results[m]['test_r2'] for m in models]
        scatter = ax2.scatter(model_r2, prices, s=200, alpha=0.6, c=range(len(models)), cmap='viridis')

        for i, model in enumerate(models):
            ax2.annotate(model, (model_r2[i], prices[i]),
                        fontsize=8, ha='center', va='bottom')

        ax2.set_xlabel('Model Test R²', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Predicted Price (€)', fontsize=11, fontweight='bold')
        ax2.set_title('Model Performance vs Prediction', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Show price range and statistics
        st.markdown("### 📈 Prediction Statistics")
        col1, col2, col3, col4 = st.columns(4)

        median_price = np.median(list(all_predictions.values()))
        std_price = np.std(list(all_predictions.values()))
        price_range = max_price - min_price

        with col1:
            st.metric("Mean Price", f"€{mean_price:,.0f}")
        with col2:
            st.metric("Median Price", f"€{median_price:,.0f}")
        with col3:
            st.metric("Std Deviation", f"€{std_price:,.0f}")
        with col4:
            st.metric("Price Range", f"€{price_range:,.0f}")

        # Multi-Feature Impact Comparison
        st.markdown("---")
        st.markdown("### 🎨 Live Feature Impact Comparison")
        st.markdown("*See how ALL key features affect the price simultaneously*")

        # Calculate impact for multiple features
        model = st.session_state.trainer.models[best_model_name]

        features_to_compare = {
            'living_area': (20, 500, 'Living Area (m²)', features['living_area']),
            'number_of_rooms': (1, 10, 'Rooms', features['number_of_rooms']),
            'garden_surface': (0, 500, 'Garden (m²)', features['garden_surface']),
            'terrace_surface': (0, 100, 'Terrace (m²)', features['terrace_surface'])
        }

        # Create multi-feature comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, (feature_name, (min_val, max_val, display_name, current_val)) in enumerate(features_to_compare.items()):
            ax = axes[idx]

            # Generate test values
            test_values = np.linspace(min_val, max_val, 30)
            impact_prices = []

            for test_val in test_values:
                test_features = features.copy()
                test_features[feature_name] = test_val
                df_test = pd.DataFrame([test_features])
                X_test = st.session_state.preprocessor.transform(df_test)
                y_test = model.predict(X_test)
                p_test = st.session_state.preprocessor.inverse_transform_target(y_test)[0]
                impact_prices.append(p_test)

            # Plot
            ax.plot(test_values, impact_prices, linewidth=2.5, color='#1f77b4')
            ax.axvline(x=current_val, color='#d62728', linestyle='--', linewidth=2,
                      label=f'Current: {current_val}')
            ax.fill_between(test_values, impact_prices, alpha=0.3, color='#1f77b4')

            ax.set_xlabel(display_name, fontsize=10, fontweight='bold')
            ax.set_ylabel('Predicted Price (€)', fontsize=10, fontweight='bold')
            ax.set_title(f'{display_name} Impact', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x:,.0f}'))

            # Add price change annotation
            price_change = max(impact_prices) - min(impact_prices)
            ax.text(0.02, 0.98, f'Range: €{price_change:,.0f}',
                   transform=ax.transAxes, fontsize=9, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        st.pyplot(fig)

        st.info("""
        **🔍 How to use this visualization:**
        - Each chart shows how changing ONE feature affects the predicted price
        - The **blue line** shows the price trend across the feature's range
        - The **red dashed line** marks your current value
        - The **shaded area** emphasizes the price variation
        - **Adjust the sliders above** and watch these charts update in real-time!
        """)

        # Feature Impact Analysis (Single Feature Deep Dive)
        st.markdown("---")
        st.markdown("### 🎯 Deep Dive: Single Feature Analysis")
        st.markdown("*Select a feature for detailed sensitivity analysis*")

        # Define features to analyze with their ranges
        numerical_features = {
            'living_area': (features['living_area'], 20, 500, 'Living Area (m²)'),
            'number_of_rooms': (features['number_of_rooms'], 1, 10, 'Number of Rooms'),
            'number_of_facades': (features['number_of_facades'], 1, 4, 'Number of Facades'),
            'garden_surface': (features['garden_surface'], 0, 500, 'Garden Surface (m²)'),
            'terrace_surface': (features['terrace_surface'], 0, 100, 'Terrace Surface (m²)')
        }

        # Create impact visualization
        selected_feature = st.selectbox(
            "Select feature to analyze",
            list(numerical_features.keys()),
            format_func=lambda x: numerical_features[x][3]
        )

        current_value, min_val, max_val, display_name = numerical_features[selected_feature]

        # Generate range of values with more detail
        test_values = np.linspace(min_val, max_val, 100)
        impact_prices = []

        for test_val in test_values:
            test_features = features.copy()
            test_features[selected_feature] = test_val
            df_test = pd.DataFrame([test_features])
            X_test = st.session_state.preprocessor.transform(df_test)
            y_test = model.predict(X_test)
            p_test = st.session_state.preprocessor.inverse_transform_target(y_test)[0]
            impact_prices.append(p_test)

        # Plot impact with enhanced visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left plot: Price vs Feature
        ax1.plot(test_values, impact_prices, linewidth=3, color='#1f77b4', label='Price Prediction')
        ax1.axvline(x=current_value, color='r', linestyle='--', linewidth=2,
                   label=f'Current Value ({current_value})')
        ax1.axhline(y=best_price, color='g', linestyle='--', linewidth=2, alpha=0.5,
                   label=f'Current Prediction (€{best_price:,.0f})')
        ax1.fill_between(test_values, impact_prices, alpha=0.3, color='#1f77b4')

        ax1.set_xlabel(display_name, fontsize=11, fontweight='bold')
        ax1.set_ylabel('Predicted Price (€)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Impact of {display_name} on Price ({best_model_name})',
                     fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x:,.0f}'))

        # Right plot: Price Change Rate (derivative)
        price_changes = np.diff(impact_prices)
        feature_steps = np.diff(test_values)
        price_rate = price_changes / feature_steps

        ax2.plot(test_values[:-1], price_rate, linewidth=2, color='#ff7f0e', label='Price Change Rate')
        ax2.axvline(x=current_value, color='r', linestyle='--', linewidth=2,
                   label=f'Current Value')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax2.fill_between(test_values[:-1], price_rate, alpha=0.3, color='#ff7f0e')

        ax2.set_xlabel(display_name, fontsize=11, fontweight='bold')
        ax2.set_ylabel('€ per unit increase', fontsize=11, fontweight='bold')
        ax2.set_title(f'Marginal Impact: Price Change Rate', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Calculate sensitivity metrics
        price_change = max(impact_prices) - min(impact_prices)
        avg_rate = price_change / (max_val - min_val)

        # Find where rate of change is highest
        max_rate_idx = np.argmax(np.abs(price_rate))
        max_rate_value = test_values[max_rate_idx]
        max_rate = price_rate[max_rate_idx]

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"""
            **📊 Sensitivity Analysis for {display_name}:**
            - **Total price variation**: €{price_change:,.0f} across full range
            - **Current value**: {current_value} → €{best_price:,.0f}
            - **Average impact**: €{avg_rate:,.0f} per unit increase
            """)
        with col2:
            st.info(f"""
            **🎯 Marginal Impact Insights:**
            - **Peak sensitivity** at {max_rate_value:.1f}: €{abs(max_rate):,.0f} per unit
            - **Current rate** at {current_value}: €{abs(price_rate[np.argmin(np.abs(test_values[:-1] - current_value))]):,.0f} per unit
            - {'📈 Increasing' if max_rate > 0 else '📉 Decreasing'} impact trend
            """)

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def analysis_page():
    """Page for model analysis and overfitting detection."""
    st.header("📈 Model Analysis & Overfitting Detection")

    if not st.session_state.trained:
        st.warning("Please train models first on the Data & Training page!")
        return

    trainer = st.session_state.trainer
    results = st.session_state.results
    preprocessor = st.session_state.preprocessor

    # Model selector for detailed analysis
    st.subheader("🔍 Select Model for Detailed Analysis")
    selected_model = st.selectbox(
        "Choose a model to analyze",
        list(trainer.models.keys()),
        key="analysis_model_select"
    )

    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Performance Overview",
        "⚙️ Parameters & Impact",
        "📈 Predictions Analysis",
        "🎯 Feature Importance"
    ])

    # TAB 1: Performance Overview
    with tab1:
        # Results table
        st.markdown("### Model Performance Metrics")
        results_df = trainer.get_results_dataframe()

        # Format the dataframe for better display
        display_df = results_df.copy()
        for col in display_df.columns:
            if col != 'model_name' and col != 'parameters':
                if 'r2' in col:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                elif 'rmse' in col or 'mae' in col or 'std' in col:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")

        st.dataframe(display_df.style.highlight_max(axis=0, subset=['train_r2', 'test_r2']))

        # Best model
        best_model_name, best_result = trainer.get_best_model(metric="test_r2")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Best Model", best_model_name)
        with col2:
            st.metric("Test R²", f"{best_result['test_r2']:.4f}")
        with col3:
            st.metric("Test RMSE", f"{best_result['test_rmse']:.2f}")

        # Overfitting detection
        st.markdown("### Overfitting Analysis")
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

        ax1.bar(x - width/2, train_rmse, width, label='Train RMSE', alpha=0.8, color='#1f77b4')
        ax1.bar(x + width/2, test_rmse, width, label='Test RMSE', alpha=0.8, color='#ff7f0e')
        ax1.set_xlabel('Model', fontsize=10)
        ax1.set_ylabel('RMSE', fontsize=10)
        ax1.set_title('Train vs Test RMSE', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. R² Comparison
        ax2 = axes[0, 1]
        train_r2 = [results[m]['train_r2'] for m in models]
        test_r2 = [results[m]['test_r2'] for m in models]

        ax2.bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8, color='#2ca02c')
        ax2.bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8, color='#d62728')
        ax2.set_xlabel('Model', fontsize=10)
        ax2.set_ylabel('R² Score', fontsize=10)
        ax2.set_title('Train vs Test R²', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Overfitting Gap
        ax3 = axes[1, 0]
        r2_gaps = [results[m]['r2_gap'] for m in models]
        colors = ['#d62728' if gap > threshold else '#2ca02c' for gap in r2_gaps]

        ax3.bar(models, r2_gaps, color=colors, alpha=0.7)
        ax3.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        ax3.set_xlabel('Model', fontsize=10)
        ax3.set_ylabel('R² Gap (Train - Test)', fontsize=10)
        ax3.set_title('Overfitting Gap Analysis', fontsize=12, fontweight='bold')
        ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Cross-validation RMSE
        ax4 = axes[1, 1]
        cv_rmse = [results[m]['cv_rmse'] for m in models]
        cv_std = [results[m]['cv_std'] for m in models]

        ax4.bar(models, cv_rmse, yerr=cv_std, capsize=5, alpha=0.7, color='#9467bd')
        ax4.set_xlabel('Model', fontsize=10)
        ax4.set_ylabel('CV RMSE', fontsize=10)
        ax4.set_title('Cross-Validation RMSE (with std)', fontsize=12, fontweight='bold')
        ax4.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Overfitting status table
        st.markdown("### Overfitting Status")
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
        st.dataframe(status_df, use_container_width=True)

        # Recommendations
        st.markdown("### 💡 Recommendations")
        for model_name, status in overfitting_status.items():
            if status["is_overfitting"]:
                with st.expander(f"⚠️ {model_name} - Overfitting Detected", expanded=True):
                    st.markdown(f"""
                    **Current R² Gap:** {status['r2_gap']:.4f} (threshold: {threshold})

                    **Recommended Actions:**
                    - 🔧 Reduce model complexity (decrease `max_depth`, `n_estimators`)
                    - 📊 Increase regularization (increase `reg_alpha`, `reg_lambda` for XGBoost, increase `C` for SVR)
                    - 📈 Collect more training data
                    - 🎯 Use feature selection to reduce dimensionality
                    - 🔄 Increase `min_samples_split` and `min_samples_leaf` for tree-based models
                    """)

    # TAB 2: Parameters & Impact
    with tab2:
        st.markdown(f"### ⚙️ Parameters for: **{selected_model}**")

        # Display parameters in a nice format
        params = trainer.model_params[selected_model]
        if params:
            col1, col2 = st.columns(2)
            param_items = list(params.items())
            mid = len(param_items) // 2

            with col1:
                for key, value in param_items[:mid]:
                    st.metric(label=key, value=str(value))

            with col2:
                for key, value in param_items[mid:]:
                    st.metric(label=key, value=str(value))
        else:
            st.info("This model has no tunable hyperparameters.")

        # Performance metrics for selected model
        st.markdown("### 📊 Performance Metrics")
        model_result = results[selected_model]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train RMSE", f"{model_result['train_rmse']:.2f}")
        with col2:
            st.metric("Test RMSE", f"{model_result['test_rmse']:.2f}")
        with col3:
            st.metric("Train R²", f"{model_result['train_r2']:.4f}")
        with col4:
            st.metric("Test R²", f"{model_result['test_r2']:.4f}")

        # Parameter Impact Analysis
        st.markdown("### 📈 Parameter Impact Analysis")
        st.markdown("""
        This section shows how the current parameters affect model behavior:
        """)

        # Create parameter impact visualization
        impact_data = []
        for model_name, model_params in trainer.model_params.items():
            model_res = results[model_name]
            impact_data.append({
                'Model': model_name,
                'Test R²': model_res['test_r2'],
                'Overfitting Gap': model_res.get('r2_gap', 0),
                'CV RMSE': model_res['cv_rmse']
            })

        impact_df = pd.DataFrame(impact_data)

        # Visualize parameter impact
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Test R² vs Overfitting Gap
        ax1 = axes[0]
        scatter = ax1.scatter(impact_df['Overfitting Gap'], impact_df['Test R²'],
                            s=200, alpha=0.6, c=range(len(impact_df)), cmap='viridis')

        for idx, row in impact_df.iterrows():
            ax1.annotate(row['Model'], (row['Overfitting Gap'], row['Test R²']),
                        fontsize=9, ha='center', va='bottom')

        ax1.set_xlabel('Overfitting Gap (R²)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Test R²', fontsize=11, fontweight='bold')
        ax1.set_title('Model Performance vs Overfitting', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({threshold})')
        ax1.legend()

        # Plot 2: CV RMSE comparison
        ax2 = axes[1]
        colors_cv = sns.color_palette('viridis', len(impact_df))
        bars = ax2.barh(impact_df['Model'], impact_df['CV RMSE'], color=colors_cv, alpha=0.7)
        ax2.set_xlabel('Cross-Validation RMSE', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Model', fontsize=11, fontweight='bold')
        ax2.set_title('Cross-Validation Performance', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}', ha='left', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)

        # Show interpretation
        st.markdown("### 🎯 Interpretation Guide")
        st.markdown("""
        - **Top-left quadrant** (low overfitting, high R²): ✅ Ideal - well-generalized model
        - **Top-right quadrant** (high overfitting, high R²): ⚠️ Good train performance but overfitting
        - **Bottom-left quadrant** (low overfitting, low R²): ⚠️ Underfitting - increase model complexity
        - **Bottom-right quadrant** (high overfitting, low R²): ❌ Poor performance overall
        """)

    # TAB 3: Predictions Analysis
    with tab3:
        st.markdown(f"### 📈 Prediction Analysis for: **{selected_model}**")

        model = trainer.models[selected_model]

        # Get predictions
        y_train_pred = model.predict(trainer.X_train)
        y_test_pred = model.predict(trainer.X_test)

        # Calculate residuals
        train_residuals = trainer.y_train - y_train_pred
        test_residuals = trainer.y_test - y_test_pred

        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        # 1. Actual vs Predicted (Train)
        ax1 = axes[0, 0]
        ax1.scatter(trainer.y_train, y_train_pred, alpha=0.5, s=20, color='#1f77b4')
        ax1.plot([trainer.y_train.min(), trainer.y_train.max()],
                [trainer.y_train.min(), trainer.y_train.max()],
                'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Values (Train)', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Predicted Values', fontsize=10, fontweight='bold')
        ax1.set_title(f'Train Set: Actual vs Predicted\nR² = {model_result["train_r2"]:.4f}',
                     fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Actual vs Predicted (Test)
        ax2 = axes[0, 1]
        ax2.scatter(trainer.y_test, y_test_pred, alpha=0.5, s=20, color='#ff7f0e')
        ax2.plot([trainer.y_test.min(), trainer.y_test.max()],
                [trainer.y_test.min(), trainer.y_test.max()],
                'r--', lw=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Values (Test)', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Predicted Values', fontsize=10, fontweight='bold')
        ax2.set_title(f'Test Set: Actual vs Predicted\nR² = {model_result["test_r2"]:.4f}',
                     fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Residuals Distribution (Train)
        ax3 = axes[1, 0]
        ax3.hist(train_residuals, bins=50, alpha=0.7, color='#1f77b4', edgecolor='black')
        ax3.axvline(x=0, color='r', linestyle='--', lw=2)
        ax3.set_xlabel('Residuals (Train)', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax3.set_title(f'Train Residuals Distribution\nMean: {train_residuals.mean():.2f}, Std: {train_residuals.std():.2f}',
                     fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Residuals Distribution (Test)
        ax4 = axes[1, 1]
        ax4.hist(test_residuals, bins=50, alpha=0.7, color='#ff7f0e', edgecolor='black')
        ax4.axvline(x=0, color='r', linestyle='--', lw=2)
        ax4.set_xlabel('Residuals (Test)', fontsize=10, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax4.set_title(f'Test Residuals Distribution\nMean: {test_residuals.mean():.2f}, Std: {test_residuals.std():.2f}',
                     fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Residual plots
        st.markdown("### 📉 Residual Analysis")
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

        # Residuals vs Predicted (Train)
        ax1 = axes2[0]
        ax1.scatter(y_train_pred, train_residuals, alpha=0.5, s=20, color='#1f77b4')
        ax1.axhline(y=0, color='r', linestyle='--', lw=2)
        ax1.set_xlabel('Predicted Values (Train)', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Residuals', fontsize=10, fontweight='bold')
        ax1.set_title('Train: Residuals vs Predicted', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Residuals vs Predicted (Test)
        ax2 = axes2[1]
        ax2.scatter(y_test_pred, test_residuals, alpha=0.5, s=20, color='#ff7f0e')
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Values (Test)', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Residuals', fontsize=10, fontweight='bold')
        ax2.set_title('Test: Residuals vs Predicted', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig2)

        st.markdown("""
        **Residual Analysis Interpretation:**
        - **Good model**: Residuals should be randomly scattered around zero with no clear pattern
        - **Heteroscedasticity**: If residuals fan out, the model's error variance is not constant
        - **Non-linear patterns**: Curved patterns indicate the model is missing non-linear relationships
        - **Normal distribution**: Residuals should be approximately normally distributed around zero
        """)

    # TAB 4: Feature Importance
    with tab4:
        st.markdown(f"### 🎯 Feature Importance for: **{selected_model}**")

        model = trainer.models[selected_model]

        # Check if model supports feature importance
        if hasattr(model, 'feature_importances_'):
            # Get feature names from preprocessor
            if hasattr(preprocessor, 'feature_names_out_'):
                feature_names = preprocessor.feature_names_out_
            else:
                feature_names = [f"Feature {i}" for i in range(trainer.X_train.shape[1])]

            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            # Display top features
            n_top = min(20, len(feature_names))

            fig, ax = plt.subplots(figsize=(12, max(6, n_top * 0.3)))

            colors = sns.color_palette('viridis', n_top)
            bars = ax.barh(range(n_top), importances[indices[:n_top]], color=colors, alpha=0.8)
            ax.set_yticks(range(n_top))
            ax.set_yticklabels([feature_names[i] for i in indices[:n_top]], fontsize=9)
            ax.set_xlabel('Importance', fontsize=11, fontweight='bold')
            ax.set_ylabel('Feature', fontsize=11, fontweight='bold')
            ax.set_title(f'Top {n_top} Feature Importances', fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')

            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f'{width:.4f}', ha='left', va='center', fontsize=8)

            plt.tight_layout()
            st.pyplot(fig)

            # Show interpretation
            st.markdown("""
            **Feature Importance Interpretation:**
            - Higher values indicate features that have more influence on the model's predictions
            - These importances are calculated based on how much each feature decreases impurity (for tree-based models)
            - Focus on the top features when selecting features or explaining model behavior
            """)

        elif hasattr(model, 'coef_'):
            # For linear models, show coefficients
            if hasattr(preprocessor, 'feature_names_out_'):
                feature_names = preprocessor.feature_names_out_
            else:
                feature_names = [f"Feature {i}" for i in range(len(model.coef_))]

            coef = model.coef_
            indices = np.argsort(np.abs(coef))[::-1]

            n_top = min(20, len(feature_names))

            fig, ax = plt.subplots(figsize=(12, max(6, n_top * 0.3)))

            colors = ['#2ca02c' if c > 0 else '#d62728' for c in coef[indices[:n_top]]]
            bars = ax.barh(range(n_top), coef[indices[:n_top]], color=colors, alpha=0.8)
            ax.set_yticks(range(n_top))
            ax.set_yticklabels([feature_names[i] for i in indices[:n_top]], fontsize=9)
            ax.set_xlabel('Coefficient Value', fontsize=11, fontweight='bold')
            ax.set_ylabel('Feature', fontsize=11, fontweight='bold')
            ax.set_title(f'Top {n_top} Feature Coefficients (by magnitude)', fontsize=12, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', lw=1)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')

            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_x = width + (0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])) if width > 0 else width - (0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0]))
                ax.text(label_x, bar.get_y() + bar.get_height()/2,
                       f'{width:.4f}', ha='left' if width > 0 else 'right', va='center', fontsize=8)

            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("""
            **Coefficient Interpretation:**
            - **Positive coefficients** (green): Increase in feature value leads to increase in prediction
            - **Negative coefficients** (red): Increase in feature value leads to decrease in prediction
            - **Magnitude**: Larger absolute values indicate stronger influence on predictions
            - Values shown are for standardized features, so they can be compared directly
            """)
        else:
            st.info(f"Feature importance is not available for {selected_model}. This metric is only supported for tree-based models and linear models.")


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
