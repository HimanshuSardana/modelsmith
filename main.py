from rich.console import Console
from rich.text import Text
from questionary import path, select, rawselect
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

csv_file = path("Please select a CSV file").ask()

df = pd.read_csv(csv_file)
column_names = df.columns.tolist()

target_column = select("Please select the target column", choices=column_names).ask()
input_columns = rawselect("Please select the input columns (use space to select multiple)", choices=column_names).ask()

models = ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "Support Vector Machine"]
