import joblib
from rich.console import Console
from rich.text import Text
from questionary import path, select, checkbox 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

console = Console()
csv_file = path("Please select a CSV file").ask()

df = pd.read_csv(csv_file)
column_names = df.columns.tolist()

target_column = select("Please select the target column", choices=column_names).ask()

input_choices = [i for i in column_names if i != target_column]
input_columns = checkbox(
    "Please select the input columns (use space to select multiple)", 
    choices=input_choices
).ask()

if not input_columns:
    console.print(Text("No input columns selected. Exiting...", style="bold red"))
    exit()

X = df[input_columns]
y = df[target_column]

# One-hot encode categorical variables in X
X = pd.get_dummies(X)

# Inform the user of transformed columns
console.print(Text(f"Input features after one-hot encoding: {list(X.columns)}", style="cyan"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "Support Vector Machine"]
model_name = select("Please select the model to train", choices=models).ask()

model_dict = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Support Vector Machine": SVR()
}

model = model_dict[model_name]
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

# Save the model
file_name = f"{model_name.replace(' ', '_').lower()}_model.pkl"
joblib.dump(model, file_name)

console.print(Text(f"{model_name} trained successfully with a score of {score:.2f}", style="bold green"))
console.print(Text(f"Model saved to {file_name}", style="bold green"))

