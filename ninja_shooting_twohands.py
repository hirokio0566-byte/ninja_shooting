import cv2
import mediapipe as mp
import numpy as np
import joblib
import random
import time
import os

# ----------------- 設定 ----------------- #

JUTSU_LABELS = ["fire", "water", "wind", "earth", "lightning"]

MODEL_PATH = "models/jutsu_svm_twohands.pkl"
ASSET_DIR = "assets"

BOTTOM_EFFECT_DURATION = 0.8

BULLET_SCALE = 0.1
ENEMY_SCALE = 0.1

EFFECT_SCALE = {
    "fire": 0.3,
    "water": 0.2,
    "wind": 0.1,
    "lightning": 0.2,
    "earth": 0.3
}

BULLET_SCALE_PER_JUTSU = {
    "fire": 0.8,
    "water": 0.5,
}

ENEMY_SCALE_PER_JUTSU = {
    "fire": 1.2,
    "earth": 0.5
}

ENEMY_HITBOX_SCALE = {
    "fire": 1.0,
    "water": 1.0,
    "wind": 1.0,
}

EFFECT_DURATION = 0.8
CAST_COOLDOWN = 0.8

HOLD_TO_CAST_SEC = 0.6
RELEASE_RESET_SEC = 0.15

EFFECT_OFFSET_X = 60
EFFECT_OFFSET_Y = -10
EFFECT_OFFSET_PER_JUTSU = {
    "fire": (100, -10),
    "water": (100, -10),
    "wind": (100, -10),
    "lightning": (100, -10),
    "earth": (50, 20)
}

FRAME_SCALE = 1.0
FRAME_Y_MARGIN = 0
FRAME_ANCHOR = "bottom"

TIME_LIMIT_SEC = 60
TARGET_SCORE = 18

# ★追加: 画面画像（assets に置く）
START_BG_PATH = os.path.join(ASSET_DIR, "start.png")
CLEAR_BG_PATH = os.path.join(ASSET_DIR, "clear.jpg")
GAMEOVER_BG_PATH = os.path.join(ASSET_DIR, "game_over.jpg")

# ★追加: 難易度切り替え（時間で3段階）
EASY_SEC = 10
HARD_SEC = 20  # 最後の20秒
# Normal は真ん中の 60 - EASY - HARD 秒

# 「普通(=今)」の基準値（これを Normal とする）
NORMAL_SPAWN_INTERVAL = 3.0
NORMAL_SPEED_RANGE = (1.5, 3.0)

# Easy / Hard の調整（好みで調整OK）
EASY_SPAWN_INTERVAL = 4.0       # 出現間隔長め = 敵少なめ
EASY_SPEED_RANGE = (1.0, 2.0)   # 落下遅め

HARD_SPAWN_INTERVAL = 1.8       # 出現間隔短め = 敵多め
HARD_SPEED_RANGE = (2.5, 4.5)   # 落下速め


# ----------------- ★追加: ダメージ（通過）関連の設定 ----------------- #

MISS_LIMIT = 3  # ★ここを 5 にすると「5体通過でGame Over」に変更できる

DAMAGE_FLASH_DURATION = 0.18   # 赤フラッシュの持続秒
DAMAGE_FLASH_ALPHA = 0.45      # 赤フラッシュの濃さ（0~1）

SCREEN_SHAKE_DURATION = 0.22   # 画面揺れの持続秒
SCREEN_SHAKE_INTENSITY = 14    # 揺れの強さ（ピクセル）


# ----------------- 共通ユーティリティ ----------------- #

def load_and_scale(path, scale):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    h, w = img.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def overlay_image_alpha(bg, fg, x, y):
    if fg is None:
        return bg

    bh, bw = bg.shape[:2]
    fh, fw = fg.shape[:2]

    if x >= bw or y >= bh or x + fw <= 0 or y + fh <= 0:
        return bg

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + fw, bw)
    y2 = min(y + fh, bh)

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
    images = {}
    for label in labels:
        path = os.path.join(ASSET_DIR, f"{label}_effect.png")
        scale = EFFECT_SCALE.get(label, 1.0)
        img = load_and_scale(path, scale)
        images[label] = img
    return images


def load_bullet_images(labels):
    images = {}
    for label in labels:
        path = os.path.join(ASSET_DIR, f"{label}_bullet.png")
        mul = BULLET_SCALE_PER_JUTSU.get(label, 1.0)
        scale = BULLET_SCALE * mul
        img = load_and_scale(path, scale)
        images[label] = img
    return images


def load_enemy_images(labels):
    images = {}
    for label in labels:
        path = os.path.join(ASSET_DIR, f"{label}_enemy.png")
        mul = ENEMY_SCALE_PER_JUTSU.get(label, 1.0)
        scale = ENEMY_SCALE * mul
        img = load_and_scale(path, scale)
        images[label] = img
    return images


def load_bottom_effect_images(labels):
    images = {}
    for label in labels:
        path = os.path.join(ASSET_DIR, f"{label}_frame.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        images[label] = img
    return images


def get_effect_offset(jutsu_label):
    return EFFECT_OFFSET_PER_JUTSU.get(jutsu_label, (EFFECT_OFFSET_X, EFFECT_OFFSET_Y))


def overlay_bottom_frame(frame, bottom_img, scale=1.0, y_margin=0, anchor="bottom"):
    if bottom_img is None:
        return frame

    bh, bw = frame.shape[:2]
    ih, iw = bottom_img.shape[:2]

    base_scale = bw / iw
    new_w = int(iw * base_scale)
    new_h = int(ih * base_scale)

    new_w = max(1, int(new_w * scale))
    new_h = max(1, int(new_h * scale))

    resized = cv2.resize(bottom_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x = int((bw - new_w) / 2)

    if anchor == "bottom":
        y = bh - new_h - y_margin
    else:
        y = bh - new_h - y_margin

    return overlay_image_alpha(frame, resized, x, y)


def load_bg_image(path):
    """背景画像を読む（無ければNone）"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def fit_bg_to_frame(bg_img, frame_w, frame_h):
    """背景画像をウィンドウサイズにフィット（アスペクト無視でOK）"""
    if bg_img is None:
        return None
    return cv2.resize(bg_img, (frame_w, frame_h), interpolation=cv2.INTER_AREA)


# ----------------- ★追加: 赤フラッシュ / 画面揺れユーティリティ ----------------- #

def apply_red_flash(frame, alpha=0.45):
    """画面全体に赤フラッシュをかける（BGR）"""
    overlay = np.zeros_like(frame, dtype=np.uint8)
    overlay[:, :] = (0, 0, 255)  # Red in BGR
    out = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
    return out


def apply_screen_shake(frame, intensity=14):
    """画面をランダムに平行移動して揺らす"""
    h, w = frame.shape[:2]
    dx = random.randint(-intensity, intensity)
    dy = random.randint(-intensity, intensity)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shaken = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT101)
    return shaken


def show_image_screen(bg_path, title_text, window_name):
    """
    画像背景の画面を表示（start/clear/gameover用）
    - bg_path が無ければ黒背景にフォールバック
    - Q/ESC で抜ける
    - Start画面の場合は SPACE/ENTER でも開始できるように main から使う
    """
    bg = load_bg_image(bg_path)

    while True:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

        if bg is not None:
            canvas = fit_bg_to_frame(bg, 640, 480)

        # ★タイトルはあれば表示（画像の「後」に描くので前に出る）
        if title_text:
            cv2.putText(canvas, title_text, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 255, 0), 3)

        # ★案内は常に表示
        cv2.putText(canvas, "Press SPACE/ENTER to start", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, "Press Q/ESC to quit", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in [27, ord('q')]:
            return "quit"
        if key in [ord(' '), 13]:  # SPACE or ENTER
            return "start"


def get_difficulty_params(elapsed_sec):
    """
    経過時間に応じて難易度パラメータを返す
    - 0~20: easy
    - middle: normal
    - last 20: hard
    """
    if elapsed_sec < EASY_SEC:
        return "EASY", EASY_SPAWN_INTERVAL, EASY_SPEED_RANGE
    if elapsed_sec >= (TIME_LIMIT_SEC - HARD_SEC):
        return "HARD", HARD_SPAWN_INTERVAL, HARD_SPEED_RANGE
    return "NORMAL", NORMAL_SPAWN_INTERVAL, NORMAL_SPEED_RANGE


# ----------------- クリア/ゲームオーバー画面（画像背景版） ----------------- #

def show_clear_screen(window_name="Ninja Shooting Game (Two Hands)"):
    bg = load_bg_image(CLEAR_BG_PATH)
    while True:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        if bg is not None:
            canvas = fit_bg_to_frame(bg, 640, 480)

        # ★文字を常に描画（背景の有無に関係なく前に出る）
        cv2.putText(canvas, "CLEAR!", (180, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
        cv2.putText(canvas, "Press Q or ESC", (190, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in [27, ord('q')]:
            break


def show_game_over_screen(window_name="Ninja Shooting Game (Two Hands)"):
    bg = load_bg_image(GAMEOVER_BG_PATH)
    while True:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        if bg is not None:
            canvas = fit_bg_to_frame(bg, 640, 480)

        # ★文字を常に描画（背景の有無に関係なく前に出る）
        cv2.putText(canvas, "GAME OVER", (110, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
        cv2.putText(canvas, "Press Q or ESC", (190, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in [27, ord('q')]:
            break


# ----------------- ゲーム用クラス ----------------- #

class Enemy:
    def __init__(self, x, y, speed, element, image=None):
        self.x = x
        self.y = y
        self.speed = speed
        self.element = element
        self.image = image

        if image is not None:
            h, w = image.shape[:2]
            base_radius = max(h, w) // 2
        else:
            base_radius = 25

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
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, (255, 255, 255), -1)


class Projectile:
    def __init__(self, x, y, vy, element, image=None):
        self.x = x
        self.y = y
        self.vy = vy
        self.element = element
        self.image = image

        if image is not None:
            h, w = image.shape[:2]
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
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, (255, 255, 255), -1)


# ----------------- MediaPipe 用ヘルパ ----------------- #

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_two_hand_features(left_hand, right_hand, image_shape):
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

    center = (left[0] + right[0]) / 2.0
    left -= center
    right -= center

    dist = np.linalg.norm(left[0] - right[0])
    if dist > 0:
        left /= dist
        right /= dist

    vec = right[0] - left[0]
    angle = -np.arctan2(vec[1], vec[0])
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

    left = (left @ R.T)
    right = (right @ R.T)

    feature = np.concatenate([left.flatten(), right.flatten()])
    return feature


def activate_jutsu(jutsu_label, hand_pos, projectiles, projectile_images):
    x, y = hand_pos
    element = jutsu_label
    bullet_img = projectile_images.get(element)
    proj = Projectile(x=x, y=y, vy=-10, element=element, image=bullet_img)
    projectiles.append(proj)


# ----------------- メインゲームループ ----------------- #

def main():
    window_name = "Ninja Shooting Game (Two Hands)"

    # ★Start画面（タイトル＆案内文が背景の前に出る）
    start_result = show_image_screen(
        START_BG_PATH,
        title_text="Ninja Shooting Game (Two Hands)",
        window_name=window_name
    )
    if start_result == "quit":
        cv2.destroyAllWindows()
        return

    clf = joblib.load(MODEL_PATH)

    effect_images = load_effect_images(JUTSU_LABELS)
    projectile_images = load_bullet_images(JUTSU_LABELS)
    enemy_images = load_enemy_images(JUTSU_LABELS)
    bottom_effect_images = load_bottom_effect_images(JUTSU_LABELS)

    cap = cv2.VideoCapture(0)

    enemies = []
    projectiles = []
    last_spawn_time = time.time()
    score = 0

    # ★追加: 敵通過（ダメージ）カウント
    miss_count = 0

    hold_label = None
    hold_start_time = 0.0
    last_seen_time = 0.0
    hold_fired = False

    last_jutsu = None
    last_cast_time = 0

    effect_pos = None

    bottom_effect_start_time = None
    bottom_effect_label = None

    game_start_time = time.time()

    # ★追加: ダメージ演出の状態
    damage_flash_until = 0.0
    shake_until = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
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

            elapsed = time.time() - game_start_time
            remaining = max(0.0, TIME_LIMIT_SEC - elapsed)

            # ★難易度パラメータ（時間で自動切替）
            diff_name, spawn_interval, speed_range = get_difficulty_params(elapsed)

            # ★制限時間チェック
            if remaining <= 0:
                if score >= TARGET_SCORE:
                    show_clear_screen(window_name=window_name)
                else:
                    show_game_over_screen(window_name=window_name)
                break

            current_jutsu = None
            hand_pos = (w // 2, int(h * 0.8))

            left_hand = None
            right_hand = None

            if result.multi_hand_landmarks and result.multi_handedness:
                for lm, handed in zip(result.multi_hand_landmarks, result.multi_handedness):
                    label = handed.classification[0].label
                    if label == "Left":
                        left_hand = lm
                    elif label == "Right":
                        right_hand = lm

            if left_hand:
                mp_drawing.draw_landmarks(frame, left_hand, mp_hands.HAND_CONNECTIONS)
            if right_hand:
                mp_drawing.draw_landmarks(frame, right_hand, mp_hands.HAND_CONNECTIONS)

            if left_hand is not None and right_hand is not None:
                lw = left_hand.landmark[0]
                rw = right_hand.landmark[0]
                cx = int((lw.x + rw.x) / 2 * w)
                cy = int((lw.y + rw.y) / 2 * h)
                hand_pos = (cx, cy)

                feat = extract_two_hand_features(left_hand, right_hand, frame.shape).reshape(1, -1)
                pred_label_id = int(clf.predict(feat)[0])
                if 0 <= pred_label_id < len(JUTSU_LABELS):
                    current_jutsu = JUTSU_LABELS[pred_label_id]
            else:
                current_jutsu = None

            now = time.time()

            # ----- ホールド判定で忍術発動 ----- #
            if current_jutsu is not None:
                last_seen_time = now

                if current_jutsu != hold_label:
                    hold_label = current_jutsu
                    hold_start_time = now
                    hold_fired = False

                hold_elapsed = now - hold_start_time
                if (not hold_fired) and (hold_elapsed >= HOLD_TO_CAST_SEC):
                    if (hold_label != last_jutsu) or (now - last_cast_time > CAST_COOLDOWN):
                        activate_jutsu(hold_label, hand_pos, projectiles, projectile_images)
                        last_jutsu = hold_label
                        last_cast_time = now
                        effect_pos = hand_pos
                        hold_fired = True
            else:
                if (now - last_seen_time) > RELEASE_RESET_SEC:
                    hold_label = None
                    hold_start_time = 0.0
                    hold_fired = False

            # ----- 敵スポーン（難易度で変化） ----- #
            if now - last_spawn_time > spawn_interval:
                ex = random.randint(50, w - 50)
                ey = -20
                speed = random.uniform(speed_range[0], speed_range[1])
                element = random.choice(JUTSU_LABELS)
                enemy_img = enemy_images.get(element)
                enemies.append(Enemy(ex, ey, speed, element, image=enemy_img))
                last_spawn_time = now

            # ----- 更新 ----- #
            for enemy in enemies:
                enemy.update()
            for proj in projectiles:
                proj.update()

            # ★追加: 画面下に抜けた敵（通過）をカウントして消す
            remaining_enemies_pass = []
            for e in enemies:
                if e.y >= h + 50:
                    miss_count += 1

                    # ★追加: ダメージ演出（赤フラッシュ＆揺れ）を発火
                    damage_flash_until = max(damage_flash_until, now + DAMAGE_FLASH_DURATION)
                    shake_until = max(shake_until, now + SCREEN_SHAKE_DURATION)
                else:
                    remaining_enemies_pass.append(e)
            enemies = remaining_enemies_pass

            # 弾は従来通り
            projectiles = [p for p in projectiles if p.y > -100]

            # ★追加: 一定数通過で即ゲームオーバー
            if miss_count >= MISS_LIMIT:
                show_game_over_screen(window_name=window_name)
                break

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

                            bottom_effect_start_time = now
                            bottom_effect_label = enemy.element
                            break
                if not hit:
                    remaining_enemies.append(enemy)
            enemies = remaining_enemies

            # ----- frameを最背面に描画 ----- #
            if bottom_effect_start_time is not None and bottom_effect_label is not None:
                if now - bottom_effect_start_time < BOTTOM_EFFECT_DURATION:
                    bottom_img = bottom_effect_images.get(bottom_effect_label)
                    frame = overlay_bottom_frame(
                        frame, bottom_img, scale=FRAME_SCALE,
                        y_margin=FRAME_Y_MARGIN, anchor=FRAME_ANCHOR
                    )
                else:
                    bottom_effect_start_time = None
                    bottom_effect_label = None

            # ----- 敵・弾描画 ----- #
            for enemy in enemies:
                enemy.draw(frame)
            for proj in projectiles:
                proj.draw(frame)

            cv2.circle(frame, hand_pos, 8, (255, 255, 255), 2)

            # ----- 手元エフェクト（横ずらし） ----- #
            if last_jutsu is not None and effect_pos is not None:
                if now - last_cast_time < EFFECT_DURATION:
                    effect_img = effect_images.get(last_jutsu)
                    if effect_img is not None:
                        dx, dy = get_effect_offset(last_jutsu)
                        pos_x = effect_pos[0] + dx
                        pos_y = effect_pos[1] + dy
                        eh, ew = effect_img.shape[:2]
                        ex = int(pos_x - ew / 2)
                        ey = int(pos_y - eh / 2)
                        frame = overlay_image_alpha(frame, effect_img, ex, ey)
                else:
                    effect_pos = None

            # ----- ホールドゲージ ----- #
            if hold_label is not None:
                hold_elapsed = max(0.0, now - hold_start_time)
                ratio = min(1.0, hold_elapsed / HOLD_TO_CAST_SEC)
                bar_x, bar_y = 10, 90
                bar_w, bar_h = 200, 16
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * ratio), bar_y + bar_h), (0, 255, 255), -1)
                cv2.putText(frame, f"Hold: {hold_label}", (bar_x, bar_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # ----- UI ----- #
            cv2.putText(frame, f"Score: {score}/{TARGET_SCORE}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {remaining:.1f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Difficulty: {diff_name}", (250, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # ★追加: Miss 表示（通過した数）
            cv2.putText(frame, f"Miss: {miss_count}/{MISS_LIMIT}", (250, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if current_jutsu:
                cv2.putText(frame, f"Jutsu: {current_jutsu}", (250, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # ----------------- ★追加: 画面揺れ＆赤フラッシュ（最後に適用） ----------------- #
            if now < shake_until:
                frame = apply_screen_shake(frame, intensity=SCREEN_SHAKE_INTENSITY)

            if now < damage_flash_until:
                frame = apply_red_flash(frame, alpha=DAMAGE_FLASH_ALPHA)
            # -------------------------------------------------------------------------- #

            cv2.imshow(window_name, frame)

            # 目標達成 → 即クリア
            if score >= TARGET_SCORE:
                show_clear_screen(window_name=window_name)
                break

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
