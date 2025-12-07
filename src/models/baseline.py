import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score


def train_baseline_logreg(features_csv: str):
    """
    Train a simple Logistic Regression model on ITA → label.
    Assumes features_csv has at least: 'ita', 'label'
    """
    df = pd.read_csv(features_csv).dropna(subset=["ita", "label"])

    X = df[["ita"]].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial"
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print(f"Macro F1: {macro_f1:.4f}")

    return clf, scaler
