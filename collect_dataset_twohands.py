import cv2
import numpy as np
import mediapipe as mp
import os

SAVE_PATH = "dataset_twohands.npz"

# ラベル（ここを書き換えるだけで拡張可能）
JUTSU_LABELS = ["fire", "water", "wind", "earth", "lightning"]
LABEL_TO_ID = {label: i for i, label in enumerate(JUTSU_LABELS)}

mp_hands = mp.solutions.hands

# -------------------------
#  両手特徴量抽出
# -------------------------
def extract_two_hand_features(left_hand, right_hand, image_shape):
    h, w, _ = image_shape

    # 座標を配列に変換
    def lm_to_xy_array(hand):
        coords = []
        for lm in hand.landmark:
            x = lm.x * w
            y = lm.y * h
            coords.append([x, y])
        return np.array(coords)

    left = lm_to_xy_array(left_hand)
    right = lm_to_xy_array(right_hand)

    # 1. 中心を左右手首の中点へ平行移動
    center = (left[0] + right[0]) / 2.0
    left -= center
    right -= center

    # 2. 左右手首間距離でスケール正規化
    dist = np.linalg.norm(left[0] - right[0])
    if dist > 0:
        left /= dist
        right /= dist

    # 3. 回転正規化（左→右が x 軸になるよう回転）
    vec = right[0] - left[0]  # 左→右ベクトル
    angle = -np.arctan2(vec[1], vec[0])  # x軸に合わせる回転角
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

    left = (left @ R.T)
    right = (right @ R.T)

    # 4. Flatten
    feature = np.concatenate([left.flatten(), right.flatten()])
    return feature


# -------------------------
#  メイン（データ収集）
# -------------------------
def main():
    print("=== Two Hands Dataset Collector ===")
    print("両手が同時に映っているときに、保存キーを押すと1フレーム保存されます。")
    print("ラベル選択キー:")
    for idx, label in enumerate(JUTSU_LABELS):
        print(f"  {idx}: {label}")
    print("保存キー: s")
    print("終了: q")

    current_label = JUTSU_LABELS[0]

    X = []
    y = []

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            h, w, _ = frame.shape

            left_hand = None
            right_hand = None

            # ----- 手の検出 -----
            if result.multi_hand_landmarks and result.multi_handedness:
                for lm, handed in zip(result.multi_hand_landmarks, result.multi_handedness):
                    label = handed.classification[0].label
                    if label == "Left":
                        left_hand = lm
                    else:
                        right_hand = lm

            # 検出して描画
            if left_hand:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, left_hand, mp_hands.HAND_CONNECTIONS)
            if right_hand:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, right_hand, mp_hands.HAND_CONNECTIONS)

            # UI表示
            cv2.putText(frame, f"Current label: {current_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # ----- 保存キー処理 -----
            key = cv2.waitKey(1) & 0xFF

            # ラベル変更
            if key in [ord(str(i)) for i in range(len(JUTSU_LABELS))]:
                current_label = JUTSU_LABELS[int(chr(key))]
                print(f"Label changed to {current_label}")

            # 保存
            if key == ord("s"):
                if left_hand is None or right_hand is None:
                    print("⚠ 両手が検出されていません。保存しません。")
                else:
                    feat = extract_two_hand_features(left_hand, right_hand, frame.shape)
                    X.append(feat)
                    y.append(LABEL_TO_ID[current_label])
                    print(f"[保存] {current_label}, 特徴量次元={feat.shape}")

            if key == ord("q"):
                break

            cv2.imshow("Two Hands Collector", frame)

    cap.release()
    cv2.destroyAllWindows()

    # 保存
    print("Saving dataset...")
    X = np.array(X)
    y = np.array(y)
    np.savez(SAVE_PATH, X=X, y=y)
    print(f"Saved to {SAVE_PATH}.  Samples={len(y)}")


if __name__ == "__main__":
    main()
