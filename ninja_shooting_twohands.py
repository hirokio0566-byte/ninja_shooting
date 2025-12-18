import cv2
import mediapipe as mp
import numpy as np
import joblib
import random
import time
import os

# ----------------- 設定 ----------------- #

# ここに術の種類を追加すれば、ゲーム全体がそれに追従して拡張される
# 例: ["fire", "water", "wind", "earth", "lightning"]
JUTSU_LABELS = ["fire", "water", "wind"]

# ★両手用SVMモデル（train_svm_twohands.pyで保存したもの）
MODEL_PATH = "models/jutsu_svm_twohands.pkl"

ASSET_DIR = "assets"

# 画面下枠エフェクトの表示時間
BOTTOM_EFFECT_DURATION = 0.4  # 何秒間表示するか

# 弾・敵のスケール（倍率） 0.1～1.0 くらいで調整
BULLET_SCALE = 0.1
ENEMY_SCALE = 0.1

# 手元エフェクト画像の倍率（術ごとに個別指定可能）
EFFECT_SCALE = {
    "fire": 0.3,
    # "water": 0.3,
    # "wind": 0.3,
}

# 術ごとに弾のスケール倍率をさらに個別調整したい場合
# 例: 火だけ小さめにする → "fire": 0.5
BULLET_SCALE_PER_JUTSU = {
    "fire": 0.8,   # fire の弾だけ BULLET_SCALE * 0.8
    "water": 0.5,  # water の弾だけ BULLET_SCALE * 0.5
    # "wind": 1.0,
}

# 敵ごとの当たり判定スケール（1.0 が画像サイズそのまま）
ENEMY_HITBOX_SCALE = {
    "fire": 1.0,
    "water": 1.0,
    "wind": 1.0,
}

# エフェクト表示時間
EFFECT_DURATION = 0.3  # 手元エフェクト
CAST_COOLDOWN = 0.8    # 同じ術の連射制限(秒)

HOLD_TO_CAST_SEC = 0.6   # この秒数ずっと同じ印なら発動
RELEASE_RESET_SEC = 0.15 # 印が途切れたとき何秒でリセット扱いにするか（ガタつき吸収）


# ----------------- 共通ユーティリティ ----------------- #

def load_and_scale(path, scale):
    """
    PNG画像を読み込み、scale 倍に縮小して返す。
    アスペクト比は維持。存在しなければ None を返す。
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    h, w = img.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def overlay_image_alpha(bg, fg, x, y):
    """
    透過PNG (fg) を bg に (x, y) を左上として重ねる。
    x,y が負になったり画像外にはみ出しても安全に動くようにクリッピング。
    fg は BGRA を前提。
    """
    if fg is None:
        return bg

    bh, bw = bg.shape[:2]
    fh, fw = fg.shape[:2]

    # 完全に画面外なら何もしない
    if x >= bw or y >= bh or x + fw <= 0 or y + fh <= 0:
        return bg

    # 背景側の描画範囲
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + fw, bw)
    y2 = min(y + fh, bh)

    # 前景側の対応範囲
    fx1 = x1 - x
    fy1 = y1 - y
    fx2 = fx1 + (x2 - x1)
    fy2 = fy1 + (y2 - y1)

    fg_crop = fg[fy1:fy2, fx1:fx2]
    bg_crop = bg[y1:y2, x1:x2]

    if fg_crop.shape[2] < 4:
        bg[y1:y2, x1:x2] = fg_crop
        return bg

    alpha = fg_crop[:, :, 3] / 255.0
    alpha = alpha[..., np.newaxis]

    bg_crop[:] = (1 - alpha) * bg_crop + alpha * fg_crop[:, :, :3]
    bg[y1:y2, x1:x2] = bg_crop
    return bg


def load_effect_images(labels):
    """
    手元に出すエフェクト画像を読み込む。
    ファイル名規約: assets/<jutsu>_effect.png  (例: fire_effect.png)
    """
    images = {}
    for label in labels:
        path = os.path.join(ASSET_DIR, f"{label}_effect.png")
        scale = EFFECT_SCALE.get(label, 1.0)
        img = load_and_scale(path, scale)
        images[label] = img
    return images


def load_bullet_images(labels):
    """
    弾画像を読み込む。
    ファイル名規約: assets/<jutsu>_bullet.png
    BULLET_SCALE および BULLET_SCALE_PER_JUTSU を使って縮小。
    """
    images = {}
    for label in labels:
        path = os.path.join(ASSET_DIR, f"{label}_bullet.png")
        mul = BULLET_SCALE_PER_JUTSU.get(label, 1.0)
        scale = BULLET_SCALE * mul
        img = load_and_scale(path, scale)
        images[label] = img
    return images


def load_enemy_images(labels):
    """
    敵画像を読み込む。
    ファイル名規約: assets/<jutsu>_enemy.png
    ENEMY_SCALE を使って縮小。
    """
    images = {}
    for label in labels:
        path = os.path.join(ASSET_DIR, f"{label}_enemy.png")
        img = load_and_scale(path, ENEMY_SCALE)
        images[label] = img
    return images


def load_bottom_effect_images(labels):
    """
    画面下枠エフェクト画像を読み込む。
    ファイル名規約: assets/<jutsu>_frame.png
    """
    images = {}
    for label in labels:
        path = os.path.join(ASSET_DIR, f"{label}_frame.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        images[label] = img
    return images


# ----------------- ゲーム用クラス ----------------- #

class Enemy:
    def __init__(self, x, y, speed, element, image=None):
        self.x = x
        self.y = y
        self.speed = speed
        self.element = element
        self.image = image

        # ベースの半径（画像サイズ or デフォルト）
        if image is not None:
            h, w = image.shape[:2]
            base_radius = max(h, w) // 2
        else:
            base_radius = 25  # 画像がないときの標準サイズ

        # 要素ごとの当たり判定倍率を適用
        scale = ENEMY_HITBOX_SCALE.get(self.element, 1.0)
        self.radius = int(base_radius * scale)

    def update(self):
        self.y += self.speed

    def draw(self, frame):
        if self.image is not None:
            h, w = self.image.shape[:2]
            x = int(self.x - w / 2)
            y = int(self.y - h / 2)
            overlay_image_alpha(frame, self.image, x, y)
        else:
            # 画像がないとき用の予備描画（色付き丸）
            color = {
                "fire": (0, 0, 255),
                "water": (255, 0, 0),
                "wind": (0, 255, 0)
            }.get(self.element, (255, 255, 255))
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, color, -1)


class Projectile:
    def __init__(self, x, y, vy, element, image=None):
        self.x = x
        self.y = y
        self.vy = vy
        self.element = element
        self.image = image  # 弾用画像（BGRA想定）

        if image is not None:
            h, w = self.image.shape[:2]
            # ユーザー指定: //2 ではなく //10 を採用
            self.radius = max(h, w) // 10
        else:
            self.radius = 25

    def update(self):
        self.y += self.vy

    def draw(self, frame):
        if self.image is not None:
            h, w = self.image.shape[:2]
            x = int(self.x - w / 2)
            y = int(self.y - h / 2)
            overlay_image_alpha(frame, self.image, x, y)
        else:
            color = {
                "fire": (0, 0, 255),
                "water": (255, 0, 0),
                "wind": (0, 255, 0)
            }.get(self.element, (255, 255, 255))
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, color, -1)


# ----------------- MediaPipe 用ヘルパ ----------------- #

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_two_hand_features(left_hand, right_hand, image_shape):
    """
    collect_dataset_twohands.py と同じロジックで両手特徴量を抽出
    - 左右手首の中点を原点
    - 左右手首距離でスケール正規化
    - 左→右方向を x 軸にそろえる（回転正規化）
    - left, right の順に flatten
    """
    h, w, _ = image_shape

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

    # 4. Flatten（左→右の順）
    feature = np.concatenate([left.flatten(), right.flatten()])
    return feature


def activate_jutsu(jutsu_label, hand_pos, projectiles, projectile_images):
    """
    忍術発動トリガー
    - projectiles に新しい弾を追加
    - 弾には術に対応した画像をセットする
    """
    x, y = hand_pos
    element = jutsu_label  # fire / water / wind / ...

    bullet_img = projectile_images.get(element)
    proj = Projectile(x=x, y=y, vy=-10, element=element, image=bullet_img)
    projectiles.append(proj)


# ----------------- メインゲームループ ----------------- #

def main():
    clf = joblib.load(MODEL_PATH)

    # エフェクト画像（手元）: <jutsu>_effect.png
    effect_images = load_effect_images(JUTSU_LABELS)

    # 弾画像: <jutsu>_bullet.png
    projectile_images = load_bullet_images(JUTSU_LABELS)

    # 敵画像: <jutsu>_enemy.png
    enemy_images = load_enemy_images(JUTSU_LABELS)

    # 画面下フレーム: <jutsu>_frame.png
    bottom_effect_images = load_bottom_effect_images(JUTSU_LABELS)

    cap = cv2.VideoCapture(0)

    enemies = []
    projectiles = []
    last_spawn_time = time.time()
    spawn_interval = 3.0  # 敵出現間隔（秒）
    score = 0

    hold_label = None
    hold_start_time = 0.0
    last_seen_time = 0.0
    hold_fired = False

    last_jutsu = None
    last_cast_time = 0

    # 手元エフェクトの描画位置
    effect_pos = None

    # 画面下フレームエフェクト用
    bottom_effect_start_time = None
    bottom_effect_label = None

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # ★両手検出
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            h, w, _ = frame.shape

            current_jutsu = None
            # 手が見えないとき用のデフォルト照準
            hand_pos = (w // 2, int(h * 0.8))

            left_hand = None
            right_hand = None

            # ----- 手の検出（両手） ----- #
            if result.multi_hand_landmarks and result.multi_handedness:
                for lm, handed in zip(result.multi_hand_landmarks, result.multi_handedness):
                    label = handed.classification[0].label  # "Left" or "Right"
                    if label == "Left":
                        left_hand = lm
                    elif label == "Right":
                        right_hand = lm

            # 描画（見やすさのため両手とも描くが、術発動は両手そろっているときだけ）
            if left_hand:
                mp_drawing.draw_landmarks(
                    frame, left_hand, mp_hands.HAND_CONNECTIONS)
            if right_hand:
                mp_drawing.draw_landmarks(
                    frame, right_hand, mp_hands.HAND_CONNECTIONS)

            # 両手がそろっているときだけ特徴量を作って分類
            if left_hand is not None and right_hand is not None:
                # 手首の中点を照準位置に
                lw = left_hand.landmark[0]
                rw = right_hand.landmark[0]
                cx = int((lw.x + rw.x) / 2 * w)
                cy = int((lw.y + rw.y) / 2 * h)
                hand_pos = (cx, cy)

                feat = extract_two_hand_features(left_hand, right_hand, frame.shape)
                feat = feat.reshape(1, -1)

                pred_label_id = int(clf.predict(feat)[0])
                if 0 <= pred_label_id < len(JUTSU_LABELS):
                    current_jutsu = JUTSU_LABELS[pred_label_id]
            else:
                # 片手だけの場合は術なし（誤発動させない）
                current_jutsu = None

            now = time.time()

            # ----- 忍術発動（クールダウン付き） ----- #
            if current_jutsu is not None:
                last_seen_time = now

                # 印が変わったらホールド開始し直し
                if current_jutsu != hold_label:
                    hold_label = current_jutsu
                    hold_start_time = now
                    hold_fired = False

                # 一定時間ホールドされたら発動（1回だけ）
                hold_elapsed = now - hold_start_time
                if (not hold_fired) and (hold_elapsed >= HOLD_TO_CAST_SEC):
                    # 既存のクールダウンも併用
                    if (hold_label != last_jutsu) or (now - last_cast_time > CAST_COOLDOWN):
                        activate_jutsu(hold_label, hand_pos, projectiles, projectile_images)
                        last_jutsu = hold_label
                        last_cast_time = now
                        effect_pos = hand_pos
                        hold_fired = True

            else:
                # 印が一瞬途切れても即リセットしない（ガタつき吸収）
                if (now - last_seen_time) > RELEASE_RESET_SEC:
                    hold_label = None
                    hold_start_time = 0.0
                    hold_fired = False

            # ----- 敵スポーン ----- #
            if now - last_spawn_time > spawn_interval:
                ex = random.randint(50, w - 50)
                ey = -20
                speed = random.uniform(1.5, 3.0)
                element = random.choice(JUTSU_LABELS)
                enemy_img = enemy_images.get(element)
                enemies.append(Enemy(ex, ey, speed, element, image=enemy_img))
                last_spawn_time = now

            # ----- 敵・弾の更新 ----- #
            for enemy in enemies:
                enemy.update()
            for proj in projectiles:
                proj.update()

            # 画面外のオブジェクトを削除
            enemies = [e for e in enemies if e.y < h + 100]
            projectiles = [p for p in projectiles if p.y > -100]

            # ----- 当たり判定 ----- #
            remaining_enemies = []
            for enemy in enemies:
                hit = False
                for proj in list(projectiles):
                    dx = enemy.x - proj.x
                    dy = enemy.y - proj.y
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist < enemy.radius + proj.radius:
                        if enemy.element == proj.element:
                            hit = True
                            score += 1
                            projectiles.remove(proj)

                            # 敵を倒したら画面下フレームエフェクト発火
                            bottom_effect_start_time = now
                            bottom_effect_label = enemy.element

                            break
                if not hit:
                    remaining_enemies.append(enemy)
            enemies = remaining_enemies

            # ----- 描画 ----- #
            for enemy in enemies:
                enemy.draw(frame)
            for proj in projectiles:
                proj.draw(frame)

            # 手の位置に照準
            cv2.circle(frame, hand_pos, 8, (255, 255, 255), 2)

            # 手元の術エフェクト
            if last_jutsu is not None and effect_pos is not None:
                if now - last_cast_time < EFFECT_DURATION:
                    effect_img = effect_images.get(last_jutsu)
                    if effect_img is not None:
                        eh, ew = effect_img.shape[:2]
                        ex = int(effect_pos[0] - ew / 2)
                        ey = int(effect_pos[1] - eh / 2)
                        frame = overlay_image_alpha(frame, effect_img, ex, ey)
                else:
                    effect_pos = None

            # 画面下のフレームエフェクト（術ごとに切り替え）
            if bottom_effect_start_time is not None and bottom_effect_label is not None:
                if now - bottom_effect_start_time < BOTTOM_EFFECT_DURATION:
                    bottom_img = bottom_effect_images.get(bottom_effect_label)
                    if bottom_img is not None:
                        bh, bw = frame.shape[:2]
                        eh, ew = bottom_img.shape[:2]

                        # 幅を画面いっぱいに合わせる（アスペクト比維持）
                        scale = bw / ew
                        new_w = bw
                        new_h = int(eh * scale)
                        resized_bottom = cv2.resize(
                            bottom_img, (new_w, new_h), interpolation=cv2.INTER_AREA
                        )

                        # 画面の下端に揃えて描画
                        x = 0
                        y = bh - new_h
                        frame = overlay_image_alpha(frame, resized_bottom, x, y)
                else:
                    bottom_effect_start_time = None
                    bottom_effect_label = None

            if hold_label is not None:
                hold_elapsed = max(0.0, now - hold_start_time)
                ratio = min(1.0, hold_elapsed / HOLD_TO_CAST_SEC)

                bar_x, bar_y = 10, 90
                bar_w, bar_h = 200, 16
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * ratio), bar_y + bar_h), (0, 255, 255), -1)
                cv2.putText(frame, f"Hold: {hold_label}", (bar_x, bar_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # UI
            cv2.putText(frame, f"Score: {score}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if current_jutsu:
                cv2.putText(frame, f"Jutsu: {current_jutsu}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Ninja Shooting Game (Two Hands)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
