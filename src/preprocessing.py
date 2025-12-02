"""
Preprocessing pipeline with JSON serialization support.
No pickle/joblib - pure JSON for model persistence.
"""
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Cap outliers to percentile limits. JSON-serializable."""

    def __init__(self, lower_percentile=1, upper_percentile=99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        """Learn percentile bounds from data."""
        self.lower_bounds_ = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X):
        """Cap values to learned bounds."""
        X_capped = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        return X_capped

    def to_dict(self):
        """Convert to JSON-serializable dict."""
        return {
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "lower_bounds": self.lower_bounds_.tolist() if self.lower_bounds_ is not None else None,
            "upper_bounds": self.upper_bounds_.tolist() if self.upper_bounds_ is not None else None
        }

    @classmethod
    def from_dict(cls, data):
        """Load from dict."""
        instance = cls(
            lower_percentile=data["lower_percentile"],
            upper_percentile=data["upper_percentile"]
        )
        instance.lower_bounds_ = np.array(data["lower_bounds"]) if data["lower_bounds"] else None
        instance.upper_bounds_ = np.array(data["upper_bounds"]) if data["upper_bounds"] else None
        return instance


def get_region(postal_code):
    """Map postal code to Belgian region."""
    try:
        pc = int(postal_code)
        regions = {
            (1000, 1299): 'Brussels', (1300, 1499): 'Walloon_Brabant',
            (1500, 1999): 'Flemish_Brabant', (2000, 2999): 'Antwerp',
            (3000, 3499): 'Flemish_Brabant', (3500, 3999): 'Limburg',
            (4000, 4999): 'Liege', (5000, 5999): 'Namur',
            (6000, 6599): 'Hainaut', (6600, 6999): 'Luxembourg',
            (7000, 7999): 'Hainaut', (8000, 8999): 'West_Flanders',
            (9000, 9999): 'East_Flanders',
        }
        for (low, high), region in regions.items():
            if low <= pc <= high:
                return region
    except:
        pass
    return "Unknown"


class FeaturePreprocessor:
    """
    Preprocessing pipeline with JSON serialization.
    All transformers stored as JSON-compatible dicts.
    """

    # Feature type definitions
    NUMERIC = {
        "number_of_rooms", "living_area", "number_of_facades",
        "garden_surface", "terrace_surface", "postal_code",
        "total_outdoor", "outdoor_ratio", "luxury_score",
        "area_log", "area_per_room",
    }

    CATEGORICAL = {"subtype_of_property", "state_of_building", "region"}

    BINARY = {"equipped_kitchen", "furnished", "open_fire",
              "terrace", "garden", "swimming_pool"}

    ALL_FEATURES = NUMERIC | CATEGORICAL | BINARY

    def __init__(self, features=None, target="price", log_target=True,
                 use_capping=True, capping_percentiles=(1, 99)):
        """Initialize preprocessor."""
        self.features = set(features) if features else self.ALL_FEATURES
        self.target = target
        self.log_target = log_target
        self.use_capping = use_capping
        self.capping_percentiles = capping_percentiles

        # Auto-assign features to correct type
        self.numeric = list(self.features & self.NUMERIC)
        self.categorical = list(self.features & self.CATEGORICAL)
        self.binary = list(self.features & self.BINARY)

        # Fitted transformers (stored as JSON-compatible dicts)
        self.numeric_imputer_values_ = None
        self.numeric_scaler_mean_ = None
        self.numeric_scaler_scale_ = None
        self.outlier_capper_ = None
        self.categorical_imputer_values_ = None
        self.categorical_encoder_categories_ = None
        self.binary_imputer_value_ = 0
        self.is_fitted_ = False

    def _engineer(self, df):
        """Add engineered features."""
        df = df.copy()

        df["region"] = df["postal_code"].apply(get_region)

        df["total_outdoor"] = (
            df["garden_surface"].fillna(0) + df["terrace_surface"].fillna(0)
        )
        df["outdoor_ratio"] = df["total_outdoor"] / (df["living_area"] + 1)

        df["area_log"] = np.log1p(df["living_area"])
        df["area_per_room"] = df["living_area"] / df["number_of_rooms"].replace(0, 1)

        df["luxury_score"] = (
            df["equipped_kitchen"].fillna(0) +
            df["furnished"].fillna(0) +
            df["open_fire"].fillna(0) +
            df["swimming_pool"].fillna(0) * 2
        )

        return df

    def fit_transform(self, df):
        """Fit and transform training data."""
        X = self._engineer(df)
        y = np.log1p(df[self.target]) if self.log_target else df[self.target]

        transformed_parts = []

        # Process numeric features
        if self.numeric:
            X_num = X[self.numeric].values

            # Impute
            imputer = SimpleImputer(strategy="median")
            X_num = imputer.fit_transform(X_num)
            self.numeric_imputer_values_ = imputer.statistics_.tolist()

            # Cap outliers
            if self.use_capping:
                self.outlier_capper_ = OutlierCapper(
                    lower_percentile=self.capping_percentiles[0],
                    upper_percentile=self.capping_percentiles[1]
                )
                X_num = self.outlier_capper_.fit_transform(X_num)

            # Scale
            scaler = StandardScaler()
            X_num = scaler.fit_transform(X_num)
            self.numeric_scaler_mean_ = scaler.mean_.tolist()
            self.numeric_scaler_scale_ = scaler.scale_.tolist()

            transformed_parts.append(X_num)

        # Process categorical features
        if self.categorical:
            X_cat = X[self.categorical].values

            # Impute
            imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
            X_cat = imputer.fit_transform(X_cat)

            # Encode
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_cat = encoder.fit_transform(X_cat)
            self.categorical_encoder_categories_ = [
                cat.tolist() for cat in encoder.categories_
            ]

            transformed_parts.append(X_cat)

        # Process binary features
        if self.binary:
            X_bin = X[self.binary].fillna(0).values
            transformed_parts.append(X_bin)

        self.is_fitted_ = True
        X_transformed = np.hstack(transformed_parts) if transformed_parts else np.array([]).reshape(len(X), 0)

        return X_transformed, y

    def transform(self, df):
        """Transform new data using fitted parameters."""
        if not self.is_fitted_:
            raise ValueError("Preprocessor not fitted. Call fit_transform() first.")

        X = self._engineer(df)
        transformed_parts = []

        # Process numeric features
        if self.numeric:
            X_num = X[self.numeric].values

            # Impute using stored values
            for i, val in enumerate(self.numeric_imputer_values_):
                mask = np.isnan(X_num[:, i])
                X_num[mask, i] = val

            # Cap outliers
            if self.use_capping and self.outlier_capper_:
                X_num = self.outlier_capper_.transform(X_num)

            # Scale using stored parameters
            X_num = (X_num - np.array(self.numeric_scaler_mean_)) / np.array(self.numeric_scaler_scale_)

            transformed_parts.append(X_num)

        # Process categorical features
        if self.categorical:
            X_cat = X[self.categorical].fillna("Unknown").values

            # One-hot encode using stored categories
            encoded = []
            for i, categories in enumerate(self.categorical_encoder_categories_):
                for cat in categories:
                    encoded.append((X_cat[:, i] == cat).astype(float))

            X_cat = np.column_stack(encoded) if encoded else np.array([]).reshape(len(X), 0)
            transformed_parts.append(X_cat)

        # Process binary features
        if self.binary:
            X_bin = X[self.binary].fillna(0).values
            transformed_parts.append(X_bin)

        X_transformed = np.hstack(transformed_parts) if transformed_parts else np.array([]).reshape(len(X), 0)
        return X_transformed

    def get_target(self, df):
        """Get target variable with same transform as fit_transform."""
        return np.log1p(df[self.target]) if self.log_target else df[self.target]

    def inverse_transform_target(self, y):
        """Inverse transform predictions back to original scale."""
        return np.expm1(y) if self.log_target else y

    def get_feature_names(self):
        """Get feature names after transformation."""
        names = []

        if self.numeric:
            names.extend(self.numeric)

        if self.categorical and self.categorical_encoder_categories_:
            for i, feature in enumerate(self.categorical):
                for category in self.categorical_encoder_categories_[i]:
                    names.append(f"{feature}_{category}")

        if self.binary:
            names.extend(self.binary)

        return names

    def info(self):
        """Show pipeline configuration."""
        print(f"Numeric ({len(self.numeric)}):     {self.numeric}")
        print(f"Categorical ({len(self.categorical)}): {self.categorical}")
        print(f"Binary ({len(self.binary)}):      {self.binary}")
        print(f"Target:            {self.target} (log={self.log_target})")
        print(f"Outlier Capping:   {self.use_capping}")
        if self.use_capping:
            print(f"  Percentiles:     {self.capping_percentiles}")

    def to_dict(self):
        """Convert preprocessor to JSON-serializable dict."""
        return {
            "features": list(self.features),
            "target": self.target,
            "log_target": self.log_target,
            "use_capping": self.use_capping,
            "capping_percentiles": list(self.capping_percentiles),
            "numeric": self.numeric,
            "categorical": self.categorical,
            "binary": self.binary,
            "numeric_imputer_values": self.numeric_imputer_values_,
            "numeric_scaler_mean": self.numeric_scaler_mean_,
            "numeric_scaler_scale": self.numeric_scaler_scale_,
            "outlier_capper": self.outlier_capper_.to_dict() if self.outlier_capper_ else None,
            "categorical_encoder_categories": self.categorical_encoder_categories_,
            "binary_imputer_value": self.binary_imputer_value_,
            "is_fitted": self.is_fitted_
        }

    @classmethod
    def from_dict(cls, data):
        """Load preprocessor from dict."""
        instance = cls(
            features=data.get("features", list(cls.ALL_FEATURES)),
            target=data.get("target", "price"),
            log_target=data.get("log_target", True),
            use_capping=data.get("use_capping", False),
            capping_percentiles=tuple(data.get("capping_percentiles", (1, 99))),
        )

        instance.numeric_imputer_values_ = data.get("numeric_imputer_values")
        instance.numeric_scaler_mean_ = data.get("numeric_scaler_mean")
        instance.numeric_scaler_scale_ = data.get("numeric_scaler_scale")

        if data.get("outlier_capper"):
            instance.outlier_capper_ = OutlierCapper.from_dict(data["outlier_capper"])

        instance.categorical_encoder_categories_ = data.get("categorical_encoder_categories")
        instance.binary_imputer_value_ = data.get("binary_imputer_value", 0)
        instance.is_fitted_ = data.get("is_fitted", False)

        return instance

    def save(self, path):
        """Save preprocessor as JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved preprocessor to {path}")

    @classmethod
    def load(cls, path):
        """Load preprocessor from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
