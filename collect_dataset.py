# collect_dataset.py
import cv2
import mediapipe as mp
import numpy as np
import os

# ラベル一覧（順番が超重要：後でSVMとゲーム側でも同じにする）
JUTSU_LABELS = ["fire", "water", "wind"]  # 火遁・水遁・風遁

DATA_PATH = "data/jutsu_dataset.csv"
os.makedirs("data", exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_features(hand_landmarks, image_shape):
    """
    MediaPipe の hand_landmarks から特徴量ベクトルを作成
    - 21点の (x, y) を画像サイズからピクセル座標に変換
    - 手首(0番)基準で平行移動
    - 全体のスケールで正規化（大きさ依存を減らす）
    """
    h, w, _ = image_shape
    coords = []
    for lm in hand_landmarks.landmark:
        x = lm.x * w
        y = lm.y * h
        coords.append([x, y])
    coords = np.array(coords)  # (21, 2)

    # 手首を基準に平行移動
    wrist = coords[0].copy()
    coords -= wrist

    # スケールで正規化（最大距離で割る）
    max_dist = np.linalg.norm(coords, axis=1).max()
    if max_dist > 0:
        coords /= max_dist

    # (21, 2) -> (42,)
    return coords.flatten()


def main():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        all_samples = []

        print("=== データ収集中 ===")
        print("1: fire（火遁）, 2: water（水遁）, 3: wind（風遁）")
        print("s: 保存して終了, q: 保存せず終了")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            cv2.putText(frame, "1:fire  2:water  3:wind  s:save  q:quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Collect Jutsu Dataset", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("保存せず終了します。")
                break
            if key == ord('s'):
                print("データを保存して終了します。")
                break

            if key in [ord('1'), ord('2'), ord('3')]:
                if result.multi_hand_landmarks:
                    # 先頭の手だけ使う
                    hand_landmarks = result.multi_hand_landmarks[0]
                    feat = extract_features(hand_landmarks, frame.shape)

                    label_idx = int(chr(key)) - 1  # '1'->0, '2'->1, '3'->2
                    label = label_idx

                    sample = np.append(feat, label)
                    all_samples.append(sample)

                    print(f"サンプル追加: label={JUTSU_LABELS[label_idx]}, total={len(all_samples)}")
                else:
                    print("手が検出されていません。もう一度ポーズを取ってください。")

        cap.release()
        cv2.destroyAllWindows()

        if len(all_samples) > 0 and key == ord('s'):
            all_samples = np.array(all_samples)
            if os.path.exists(DATA_PATH):
                # 既存データに追記
                old = np.loadtxt(DATA_PATH, delimiter=',')
                if old.ndim == 1:
                    old = old[np.newaxis, :]
                all_samples = np.vstack([old, all_samples])
            np.savetxt(DATA_PATH, all_samples, delimiter=',')
            print(f"データを保存しました: {DATA_PATH}")
        else:
            print("データは保存されませんでした。")


if __name__ == "__main__":
    main()
