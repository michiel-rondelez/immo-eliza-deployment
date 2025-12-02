"""
Quick test script to verify the pipeline works correctly.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from preprocessing import FeaturePreprocessor
from models import ModelTrainer


def test_preprocessing():
    """Test preprocessing pipeline."""
    print("Testing preprocessing pipeline...")

    # Create sample data
    data = {
        "number_of_rooms": [3, 4, 2, 5],
        "living_area": [100, 150, 75, 200],
        "number_of_facades": [2, 2, 1, 3],
        "garden_surface": [0, 200, 0, 500],
        "terrace_surface": [15, 0, 10, 20],
        "postal_code": [1000, 2000, 9000, 3000],
        "subtype_of_property": ["apartment", "house", "apartment", "villa"],
        "state_of_building": ["good", "as_new", "to_renovate", "just_renovated"],
        "equipped_kitchen": [1, 1, 0, 1],
        "furnished": [0, 0, 0, 1],
        "open_fire": [0, 1, 0, 1],
        "terrace": [1, 0, 1, 1],
        "garden": [0, 1, 0, 1],
        "swimming_pool": [0, 0, 0, 1],
        "price": [250000, 350000, 180000, 550000]
    }
    df = pd.DataFrame(data)

    # Initialize preprocessor
    prep = FeaturePreprocessor(use_capping=False)

    # Fit and transform
    X, y = prep.fit_transform(df)

    print(f"✅ Preprocessing successful!")
    print(f"   Input shape: {df.shape}")
    print(f"   Output shape: {X.shape}")
    print(f"   Target shape: {y.shape}")

    # Test save/load
    prep.save("test_prep.json")
    prep_loaded = FeaturePreprocessor.load("test_prep.json")
    print(f"✅ Save/load successful!")

    # Clean up
    Path("test_prep.json").unlink()

    return prep, X, y


def test_model_training():
    """Test model training."""
    print("\nTesting model training...")

    # Get preprocessed data
    prep, X, y = test_preprocessing()

    # Initialize trainer with simplified params
    params = {
        "Linear Regression": {},
        "Decision Tree": {
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42
        }
    }

    trainer = ModelTrainer(model_params=params)

    # Split data
    trainer.train_test_split_data(X, y, test_size=0.25)

    # Train models
    results = trainer.train_all_models(cv_folds=2)

    print(f"✅ Model training successful!")
    print(f"   Trained models: {list(results.keys())}")

    for model_name, result in results.items():
        print(f"\n   {model_name}:")
        print(f"     Train R²: {result['train_r2']:.4f}")
        print(f"     Test R²:  {result['test_r2']:.4f}")

    # Test save/load
    trainer.save("test_trainer.json")
    print(f"✅ Model save successful!")

    # Clean up
    Path("test_trainer.json").unlink()

    return trainer


def test_prediction():
    """Test prediction."""
    print("\nTesting prediction...")

    # Setup
    prep, X, y = test_preprocessing()
    trainer = test_model_training()

    # Make prediction
    X_test_sample = X[:1]
    y_pred = trainer.predict("Linear Regression", X_test_sample)

    print(f"✅ Prediction successful!")
    print(f"   Predicted value: {y_pred[0]:.2f}")
    print(f"   Actual value: {y[0]:.2f}")

    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Immo Eliza ML Pipeline")
    print("=" * 50)

    try:
        # Test preprocessing
        test_preprocessing()

        # Test model training
        test_model_training()

        # Test prediction
        test_prediction()

        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        print("=" * 50)
        print("\nYou can now run the application:")
        print("  - Streamlit UI: ./run_streamlit.sh")
        print("  - FastAPI: ./run_api.sh")

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
