import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

dagshub.init(repo_owner='aarushi.kankanmeli', repo_name='MLOPs-Experiments-with-MLFlow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/aarushi.kankanmeli/MLOPs-Experiments-with-MLFlow.mlflow")

# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params of RF model
max_depth = 10
n_estimators = 5

# Enable autologging
mlflow.autolog()

# Mention your Experiment below or pass as an argument in mlflow.start_run
mlflow.set_experiment('MLOPs-MLFlow_Exp2')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Creating a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save plot
    plt.savefig('confusion_matrix.png')

    # Log artifacts using MLFlow
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({"Author": "Aarushi", "Project": "Wine Classification"}) #autolog cannot log the tags so we add the tags manually

    print(accuracy)