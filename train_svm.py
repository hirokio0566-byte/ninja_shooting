# train_svm.py
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

DATA_PATH = "data/jutsu_dataset.csv"
MODEL_PATH = "models/jutsu_svm.pkl"
os.makedirs("models", exist_ok=True)

# ラベル一覧（collect_dataset.py と一致させる）
JUTSU_LABELS = ["fire", "water", "wind"]


def main():
    data = np.loadtxt(DATA_PATH, delimiter=',')
    if data.ndim == 1:
        data = data[np.newaxis, :]

    X = data[:, :-1]
    y = data[:, -1].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = SVC(kernel="rbf", C=1.0, gamma="scale")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("=== 評価結果 ===")
    print(classification_report(
        y_test, y_pred, target_names=JUTSU_LABELS, digits=3))

    joblib.dump(clf, MODEL_PATH)
    print(f"モデルを保存しました: {MODEL_PATH}")


if __name__ == "__main__":
    main()
