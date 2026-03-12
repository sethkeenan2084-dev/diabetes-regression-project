from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    """Stores evaluation results for one trained model."""

    name: str
    mae: float
    rmse: float
    r2: float


class DiabetesRegressionProject:
    """
    Train and compare regression models on scikit-learn's built-in diabetes dataset.

    Attributes
    ----------
    test_size : float
        Proportion of data reserved for testing.
    random_state : int
        Seed used to make the train/test split reproducible.
    output_dir : Path
        Folder where results files will be saved.
    data : pd.DataFrame | None
        The feature dataset after loading.
    target : pd.Series | None
        The regression target after loading.
    X_train, X_test, y_train, y_test : pd.DataFrame | pd.Series | None
        Split datasets used for model training and evaluation.
    models : dict[str, Any]
        Dictionary of model names mapped to initialized regression estimators.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42, output_dir: str = "outputs") -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data: pd.DataFrame | None = None
        self.target: pd.Series | None = None
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None

        self.models: Dict[str, Any] = {
            "Linear Regression": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression()),
                ]
            ),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=300,
                random_state=self.random_state,
            ),
            "Gradient Boosting Regressor": GradientBoostingRegressor(
                random_state=self.random_state,
            ),
        }

    def load_data(self) -> None:
        """Load the built-in diabetes dataset into pandas objects."""
        dataset = load_diabetes(as_frame=True)
        self.data = dataset.data.copy()
        self.target = dataset.target.copy()

    def split_data(self) -> None:
        """Split the dataset into training and testing sets."""
        if self.data is None or self.target is None:
            raise ValueError("Data has not been loaded yet. Run load_data() first.")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data,
            self.target,
            test_size=self.test_size,
            random_state=self.random_state,
        )

    def train_and_evaluate(self) -> pd.DataFrame:
        """Train each model and return a table of evaluation metrics."""
        if any(v is None for v in [self.X_train, self.X_test, self.y_train, self.y_test]):
            raise ValueError("Data has not been split yet. Run split_data() first.")

        results: list[ModelResult] = []

        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)

            results.append(
                ModelResult(
                    name=name,
                    mae=mean_absolute_error(self.y_test, predictions),
                    rmse=mean_squared_error(self.y_test, predictions) ** 0.5,
                    r2=r2_score(self.y_test, predictions),
                )
            )

        results_df = pd.DataFrame([result.__dict__ for result in results])
        results_df = results_df.sort_values(by="r2", ascending=False).reset_index(drop=True)
        return results_df

    def save_results(self, results_df: pd.DataFrame) -> None:
        """Save the comparison table as CSV and JSON."""
        csv_path = self.output_dir / "diabetes_model_results.csv"
        json_path = self.output_dir / "diabetes_model_results.json"

        results_df.to_csv(csv_path, index=False)
        json_path.write_text(results_df.to_json(orient="records", indent=2), encoding="utf-8")

    def run(self) -> pd.DataFrame:
        """Complete the full workflow and return the model comparison table."""
        self.load_data()
        self.split_data()
        results_df = self.train_and_evaluate()
        self.save_results(results_df)
        return results_df


if __name__ == "__main__":
    project = DiabetesRegressionProject()
    results = project.run()

    print("Diabetes Regression Model Comparison")
    print(results.to_string(index=False))

    best_model = results.iloc[0]
    summary = {
        "best_model": best_model["name"],
        "mae": round(float(best_model["mae"]), 4),
        "rmse": round(float(best_model["rmse"]), 4),
        "r2": round(float(best_model["r2"]), 4),
    }
    print("\nBest overall model summary:")
    print(json.dumps(summary, indent=2))
