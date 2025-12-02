"""
Model training with JSON serialization for all models.
Supports parameter tuning and overfitting detection.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Try to import XGBoost, but make it optional
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: XGBoost not available. XGBoost models will be disabled.")
    print("To install XGBoost, run: pip install xgboost")


class ModelTrainer:
    """
    Train and evaluate multiple regression models.
    All parameters and coefficients stored as JSON.
    """

    DEFAULT_PARAMS = {
        "Linear Regression": {},
        "SVR": {
            "kernel": "rbf",
            "C": 1.0,
            "epsilon": 0.1,
            "gamma": "scale"
        },
        "Decision Tree": {
            "max_depth": 8,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": 42
        },
        "Random Forest": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "random_state": 42
        }
    }

    # Add XGBoost params only if available
    if XGBOOST_AVAILABLE:
        DEFAULT_PARAMS["XGBoost"] = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbosity": 0
        }

    def __init__(self, model_params=None):
        """
        Initialize trainer.

        Args:
            model_params: Dict of model_name -> parameters dict
        """
        self.model_params = model_params or self.DEFAULT_PARAMS.copy()
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _create_model(self, model_name, params):
        """Create model instance from parameters."""
        if model_name == "Linear Regression":
            return LinearRegression(**params)
        elif model_name == "SVR":
            return SVR(**params)
        elif model_name == "Decision Tree":
            return DecisionTreeRegressor(**params)
        elif model_name == "Random Forest":
            return RandomForestRegressor(**params)
        elif model_name == "XGBoost":
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost is not installed. Run: pip install xgboost")
            return XGBRegressor(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _model_to_dict(self, model_name, model):
        """Convert trained model to JSON-serializable dict."""
        if model_name == "Linear Regression":
            return {
                "type": "LinearRegression",
                "coefficients": model.coef_.tolist(),
                "intercept": float(model.intercept_)
            }
        elif model_name == "SVR":
            return {
                "type": "SVR",
                "support_vectors": model.support_vectors_.tolist(),
                "dual_coef": model.dual_coef_.tolist(),
                "intercept": model.intercept_.tolist(),
                "support": model.support_.tolist(),
                "n_support": model.n_support_.tolist()
            }
        elif model_name == "Decision Tree":
            return {
                "type": "DecisionTree",
                "tree": self._tree_to_dict(model.tree_),
                "n_features": int(model.n_features_in_),
                "n_outputs": int(model.n_outputs_)
            }
        elif model_name == "Random Forest":
            return {
                "type": "RandomForest",
                "trees": [self._tree_to_dict(tree.tree_) for tree in model.estimators_],
                "n_features": int(model.n_features_in_)
            }
        elif model_name == "XGBoost":
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost is not available")
            return {
                "type": "XGBoost",
                "booster": model.get_booster().save_raw("json").decode(),
                "n_features": int(model.n_features_in_)
            }

    def _tree_to_dict(self, tree):
        """Convert sklearn tree to dict (simplified for JSON)."""
        return {
            "node_count": int(tree.node_count),
            "max_depth": int(tree.max_depth),
            "children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "value": tree.value.tolist(),
            "impurity": tree.impurity.tolist(),
            "n_node_samples": tree.n_node_samples.tolist()
        }

    def _model_from_dict(self, model_name, data):
        """Reconstruct model from JSON dict."""
        model = self._create_model(model_name, self.model_params[model_name])

        if model_name == "Linear Regression":
            model.coef_ = np.array(data["coefficients"])
            model.intercept_ = data["intercept"]
        elif model_name == "SVR":
            model.support_vectors_ = np.array(data["support_vectors"])
            model.dual_coef_ = np.array(data["dual_coef"])
            model.intercept_ = np.array(data["intercept"])
            model.support_ = np.array(data["support"])
            model.n_support_ = np.array(data["n_support"])
        # Note: Tree-based models are complex to reconstruct from JSON
        # For production, consider using model.predict() during training
        # and storing predictions or re-training

        return model

    def train_test_split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train/test sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_all_models(self, X_train=None, y_train=None, cv_folds=5):
        """
        Train all models and evaluate.

        Args:
            X_train: Training features (uses stored if None)
            y_train: Training target (uses stored if None)
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with results for each model
        """
        X_train = X_train if X_train is not None else self.X_train
        y_train = y_train if y_train is not None else self.y_train

        if X_train is None or y_train is None:
            raise ValueError("No training data available. Call train_test_split_data() first.")

        for model_name, params in self.model_params.items():
            print(f"\nTraining {model_name}...")

            # Create and train model
            model = self._create_model(model_name, params)
            model.fit(X_train, y_train)

            # Store model
            self.models[model_name] = model

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(self.X_test) if self.X_test is not None else None

            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train,
                                       cv=cv_folds, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())

            result = {
                "model_name": model_name,
                "parameters": params,
                "train_rmse": float(train_rmse),
                "train_mae": float(train_mae),
                "train_r2": float(train_r2),
                "cv_rmse": float(cv_rmse),
                "cv_std": float(np.sqrt(-cv_scores).std())
            }

            # Test metrics if available
            if y_test_pred is not None:
                test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
                test_mae = mean_absolute_error(self.y_test, y_test_pred)
                test_r2 = r2_score(self.y_test, y_test_pred)

                result.update({
                    "test_rmse": float(test_rmse),
                    "test_mae": float(test_mae),
                    "test_r2": float(test_r2),
                    "overfitting_gap": float(test_rmse - train_rmse),
                    "r2_gap": float(train_r2 - test_r2)
                })

            self.results[model_name] = result

            print(f"  Train RMSE: {train_rmse:.2f}, R²: {train_r2:.4f}")
            if y_test_pred is not None:
                print(f"  Test RMSE:  {test_rmse:.2f}, R²: {test_r2:.4f}")
                print(f"  Overfitting Gap: {result['overfitting_gap']:.2f}")

        return self.results

    def predict(self, model_name, X):
        """Make predictions with a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet.")
        return self.models[model_name].predict(X)

    def get_best_model(self, metric="test_r2"):
        """Get best model based on metric."""
        if not self.results:
            raise ValueError("No models trained yet.")

        best_model = max(self.results.items(),
                        key=lambda x: x[1].get(metric, -np.inf))
        return best_model[0], best_model[1]

    def detect_overfitting(self, threshold=0.1):
        """
        Detect overfitting in models.

        Args:
            threshold: Max acceptable gap between train and test R²

        Returns:
            Dict of model_name -> overfitting status
        """
        overfitting_status = {}

        for model_name, result in self.results.items():
            if "r2_gap" in result:
                is_overfitting = result["r2_gap"] > threshold
                overfitting_status[model_name] = {
                    "is_overfitting": is_overfitting,
                    "r2_gap": result["r2_gap"],
                    "train_r2": result["train_r2"],
                    "test_r2": result["test_r2"]
                }

        return overfitting_status

    def to_dict(self):
        """Convert trainer state to JSON-serializable dict."""
        return {
            "model_params": self.model_params,
            "results": self.results,
            "models": {
                name: self._model_to_dict(name, model)
                for name, model in self.models.items()
            }
        }

    def save(self, path):
        """Save trainer state as JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved trainer to {path}")

    @classmethod
    def load(cls, path):
        """Load trainer from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        instance = cls(model_params=data["model_params"])
        instance.results = data["results"]

        # Reconstruct models (simplified - only for Linear and SVR)
        for model_name, model_data in data.get("models", {}).items():
            if model_name in ["Linear Regression", "SVR"]:
                instance.models[model_name] = instance._model_from_dict(
                    model_name, model_data
                )

        return instance

    def get_results_dataframe(self):
        """Get results as pandas DataFrame."""
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(self.results, orient='index')
        df.index.name = 'Model'
        return df
