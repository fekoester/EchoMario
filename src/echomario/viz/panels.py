from __future__ import annotations

import cv2
import numpy as np


DEFAULT_INPUT_LABELS = [
    "x_norm",
    "y_norm",
    "vx",
    "vy",
    "on_ground",
    "enemy_dx",
    "gap_start_dx",
    "gap_end_dx",
    "obstacle_dx",
    "gap_ahead",
    "goal_dx",
    "time",
    "gap+1",
    "enemy+1",
    "gap+2",
    "enemy+2",
    "gap+3",
    "enemy+3",
    "gap+4",
    "enemy+4",
    "gap+5",
    "enemy+5",
    "gap+6",
    "enemy+6",
    "gap+7",
    "enemy+7",
    "gap+8",
    "enemy+8",
]


def resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height), interpolation=cv2.INTER_NEAREST)


def draw_bar_panel(
    values: np.ndarray,
    labels: list[str],
    title: str,
    width: int = 360,
    height: int = 240,
) -> np.ndarray:
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)

    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return panel

    max_abs = float(np.max(np.abs(values))) + 1e-8
    y0 = 50
    row_h = max(16, (height - 60) // len(values))

    for i, v in enumerate(values):
        y = y0 + i * row_h
        if y + 14 >= height:
            break

        frac = abs(float(v)) / max_abs
        bar_w = int(frac * (width - 170))
        color = (40, 120, 220) if v >= 0 else (220, 80, 60)

        cv2.putText(panel, labels[i][:15], (10, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1)
        cv2.rectangle(panel, (135, y), (135 + bar_w, y + 11), color, -1)
        cv2.putText(panel, f"{v:+.2f}", (145 + bar_w, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (20, 20, 20), 1)

    return panel


def draw_input_panel(
    observation: np.ndarray,
    labels: list[str] | None = None,
    width: int = 420,
    height: int = 300,
) -> np.ndarray:
    if labels is None:
        labels = DEFAULT_INPUT_LABELS

    obs = np.asarray(observation, dtype=np.float32).reshape(-1)

    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(panel, "Input to reservoir", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)

    y0 = 48
    row_h = 17
    max_abs = max(1.0, float(np.max(np.abs(obs))) + 1e-8)

    for i, value in enumerate(obs):
        y = y0 + i * row_h
        if y + row_h >= height:
            break

        label = labels[i] if i < len(labels) else f"u{i}"
        value_f = float(value)

        center_x = 250
        max_bar = 135
        bar_w = int(min(1.0, abs(value_f) / max_abs) * max_bar)

        cv2.putText(panel, label[:16], (10, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1)
        cv2.line(panel, (center_x, y), (center_x, y + 11), (160, 160, 160), 1)

        if value_f >= 0:
            cv2.rectangle(panel, (center_x, y), (center_x + bar_w, y + 11), (40, 120, 220), -1)
        else:
            cv2.rectangle(panel, (center_x - bar_w, y), (center_x, y + 11), (220, 80, 60), -1)

        cv2.putText(panel, f"{value_f:+.2f}", (330, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (20, 20, 20), 1)

    return panel


def draw_activation_panel(x: np.ndarray, width: int = 420, height: int = 190) -> np.ndarray:
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(panel, "Reservoir activation", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)

    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    cols = 50
    rows = int(np.ceil(n / cols))
    cell_w = max(1, (width - 20) // cols)
    cell_h = max(1, (height - 45) // rows)

    vmax = float(np.percentile(np.abs(x), 95)) + 1e-8

    for i, val in enumerate(x):
        r = i // cols
        c = i % cols
        intensity = int(np.clip(abs(val) / vmax, 0, 1) * 255)
        if val >= 0:
            color = (255 - intensity, 255 - intensity, 255)
        else:
            color = (255, 255 - intensity, 255 - intensity)

        y = 40 + r * cell_h
        x0 = 10 + c * cell_w
        if y + cell_h < height:
            panel[y : y + cell_h, x0 : x0 + cell_w] = color

    return panel


def compose_frame(
    game_frame: np.ndarray,
    reservoir_state: np.ndarray,
    action_probs: np.ndarray,
    action_labels: list[str],
    contribution: np.ndarray,
    selected_action: int,
    step: int,
    ep_return: float,
    observation: np.ndarray | None = None,
    input_labels: list[str] | None = None,
) -> np.ndarray:
    game = resize_to_height(game_frame, 480)

    top_k = min(10, len(contribution))
    ids = np.argsort(np.abs(contribution))[-top_k:][::-1]
    contrib_values = contribution[ids]
    contrib_labels = [f"n{int(i)}" for i in ids]

    action_title = "Action output"
    if 0 <= selected_action < len(action_labels):
        action_title = f"Action output | selected: {action_labels[selected_action]}"

    action_panel = draw_bar_panel(
        action_probs,
        action_labels,
        title=action_title,
        width=420,
        height=180,
    )

    contrib_panel = draw_bar_panel(
        contrib_values,
        contrib_labels,
        title="Top decision contributions",
        width=420,
        height=210,
    )

    activation_panel = draw_activation_panel(
        reservoir_state,
        width=420,
        height=170,
    )

    if observation is None:
        input_panel = np.full((250, 420, 3), 245, dtype=np.uint8)
        cv2.putText(input_panel, "Input to reservoir unavailable", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    else:
        input_panel = draw_input_panel(
            observation=observation,
            labels=input_labels,
            width=420,
            height=300,
        )

    right = np.vstack([action_panel, input_panel, contrib_panel, activation_panel])
    right = resize_to_height(right, game.shape[0])

    frame = np.hstack([game, right])

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 36), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    cv2.putText(
        frame,
        f"step={step} return={ep_return:.2f}",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    return frame