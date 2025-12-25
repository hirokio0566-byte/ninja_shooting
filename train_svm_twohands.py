import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

DATASET_PATH = "dataset_twohands.npz"
MODEL_SAVE_PATH = "models/jutsu_svm_twohands.pkl"

# ここを変えるだけで術を拡張できる
JUTSU_LABELS = ["fire", "water", "wind", "earth", "lightning"]


def main():
    print("=== Training Two-Hands SVM ===")

    # データ読み込み
    data = np.load(DATASET_PATH)
    X = data["X"]
    y = data["y"]

    print(f"Loaded dataset: {X.shape}, labels={len(y)}")

    # SVM学習（RBF）
    clf = SVC(kernel="rbf", C=10, gamma="scale", probability=False)

    clf.fit(X, y)
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)

    print(f"Training accuracy: {acc:.4f}")

    # 保存
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
