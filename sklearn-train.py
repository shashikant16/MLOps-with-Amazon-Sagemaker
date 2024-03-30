
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
import pandas as pd, numpy as np, os, argparse

## predict 

# inference function - tells SageMaker how to load the model
def model_fn(model_dir):
    clf = load(os.path.join(model_dir, "model.joblib"))
    return clf    

# Argument parser
def _parse_args():
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--test-file", type=str, default="test.csv")
    # Parse the arguments
    return parser.parse_known_args()

# Main Training Loop
if __name__=="__main__":
    # Process arguments
    args, _ = _parse_args()
    # Load the dataset
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))
    # Separate X and y
    X_train, y_train = train_df.drop(train_df.columns[0], axis=1), train_df[train_df.columns[0]]
    X_test, y_test = test_df.drop(test_df.columns[0], axis=1), test_df[test_df.columns[0]]
    # Define the model and train it
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, n_jobs=-1
    )
    model.fit(X_train, y_train)
    # Evaluate the model performances
    print(f'Model Accuracy: {accuracy_score(y_test, model.predict(X_test))}')
    dump(model, os.path.join(args.model_dir, 'model.joblib'))
