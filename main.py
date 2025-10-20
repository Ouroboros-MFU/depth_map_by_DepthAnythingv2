#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, math

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import pipeline
import pysubs2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



def robust_center_value(arr: np.ndarray, cx: int, cy: int, r: int = 2) -> float:
    h, w = arr.shape
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    patch = arr[y0:y1, x0:x1]
    patch = patch[np.isfinite(patch)]
    val = float(np.median(patch)) if patch.size else float("nan")
    if not np.isfinite(val) or val <= 0:
        raise ValueError(f"Некорректное значение в центре калибровки: {val}")
    return val

def roi_values(arr: np.ndarray, cx: int, cy: int, r: int) -> np.ndarray:
    h, w = arr.shape
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (xx - cx)**2 + (yy - cy)**2 <= r*r
    vals = arr[y0:y1, x0:x1][mask]
    return vals[np.isfinite(vals)]

def smooth_disparity(rel_map: np.ndarray, mode: str = "bilateral",
                     median_ksize: int = 5,
                     bilateral_d: int = 9,
                     bilateral_sigma_color: float = 0.1,
                     bilateral_sigma_space: float = 5.0) -> np.ndarray:
    if mode == "none":
        return rel_map.astype(np.float32)
    import cv2
    S = rel_map.astype(np.float32)
    if mode == "median":
        k = max(3, int(median_ksize) | 1)
        return cv2.medianBlur(S, k)
    if mode == "bilateral":
        Smin, Smax = np.percentile(S, [1, 99])
        scale = max(Smax - Smin, 1e-6)
        Sn = (S - Smin) / scale
        out = cv2.bilateralFilter(Sn, d=bilateral_d,
                                  sigmaColor=bilateral_sigma_color,
                                  sigmaSpace=bilateral_sigma_space)
        return out * scale + Smin
    raise ValueError("unknown smoothing mode")

def to_metric_distance(rel_map: torch.Tensor, center_distance_m: float,
                       cx: int, cy: int, mode: str = "disparity",
                       center_radius: int = 2, gamma_fit: bool = False,
                       gamma_range=(0.70, 1.10, 0.02), roi_radius: int = 30) -> np.ndarray:
    rel = rel_map.cpu().numpy().astype(np.float32)
    h, w = rel.shape
    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))
    Sc = robust_center_value(rel, cx, cy, r=center_radius)

    if mode == "depth":
        scale = center_distance_m / Sc
        Z = rel * scale
        return np.clip(Z.astype(np.float32), 0.0, None)

    if not gamma_fit:
        k = center_distance_m * Sc
        Z = k / np.maximum(rel, 1e-12)
        return np.clip(Z.astype(np.float32), 0.0, None)

    vals = roi_values(rel, cx, cy, roi_radius)
    vals = vals[(vals > 0)]
    if vals.size < 50:
        k = center_distance_m * Sc
        Z = k / np.maximum(rel, 1e-12)
        return np.clip(Z.astype(np.float32), 0.0, None)

    if isinstance(gamma_range, (list, tuple)):
        g0, g1, gs = gamma_range
    else:
        parts = [float(x) for x in str(gamma_range).split(",")]
        g0, g1 = parts[0], parts[1]
        gs = parts[2] if len(parts) > 2 else 0.02

    best_g, best_var = None, None
    for g in np.arange(g0, g1 + 1e-9, gs):
        Sg = vals ** g
        k = center_distance_m * np.median(Sg)
        Z_roi = k / np.maximum(Sg, 1e-12)
        v = np.var(Z_roi)
        if best_var is None or v < best_var:
            best_var = v
            best_g = g

    Sg_full = np.power(np.maximum(rel, 1e-12), best_g)
    k = center_distance_m * np.median(vals ** best_g)
    Z = k / Sg_full
    return np.clip(Z.astype(np.float32), 0.0, None)

def rescale_meters(depth_m: np.ndarray, a: float, b: float,
                   source: str = "roi",
                   roi_vals: np.ndarray = None,
                   low_q: float = 5.0, high_q: float = 95.0) -> np.ndarray:
    z = depth_m.astype(np.float32)
    if source == "roi" and roi_vals is not None and roi_vals.size >= 20:
        base = roi_vals
    else:
        base = z[np.isfinite(z)]
    if base.size == 0:
        return z
    vmin, vmax = np.percentile(base, [low_q, high_q])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin < 1e-9:
        return z
    out = a + (z - vmin) * (b - a) / (vmax - vmin)
    return np.clip(out, min(a,b), max(a,b)).astype(np.float32)

# ---------- парсинг H по секундам ----------

FIELD_RE = re.compile(
    r"\b([A-Z]\.?[A-Z]?)\s*([-+]?\d+(?:[.,]\d+)?)\s*(m|см|mm|cm|мм|m/s|км/ч|km/h)?\b",
    re.IGNORECASE
)

def parse_H_from_text(text: str) -> float | None:
    text = text.replace("\n", " ")
    matches = FIELD_RE.findall(text)
    if not matches:
        return None
    H_val = None
    for key, num, unit in matches:
        key_up = key.upper().replace(".", "")
        if key_up == "H":
            num = num.replace(",", ".")
            try:
                v = float(num)
            except:
                continue
            u = (unit or "m").lower()
            if u in ["мм", "mm"]:
                v = v / 1000.0
            elif u in ["см", "cm"]:
                v = v / 100.0
            H_val = v
    return H_val

def build_second_map_from_subs(subs_path: str) -> dict[int, float]:
    subs = pysubs2.load(subs_path)
    sec2H = {}
    for ev in subs:
        sec = int(ev.start // 1000)   # секунда от начала
        H_val = parse_H_from_text(ev.text)
        if H_val is not None:
            sec2H[sec] = H_val        # последнее за секунду перекрывает предыдущее
    return sec2H

# ----------  ----------

class MPLRenderer:
    def __init__(self, width: int, height: int, cmap: str = "plasma_r",
                 show_text: bool = True, text_fmt: str = "t={t:.2f}s  H={H:.2f}m  Dc≈{Dc:.2f}m",
                 show_colorbar: bool = True):
        dpi = 100
        figsize = (width / dpi, height / dpi)
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.ax.axis("off")

        # Основное изображение
        self.im = self.ax.imshow(np.zeros((height, width), dtype=np.float32),
                                 cmap=cmap, vmin=0, vmax=1)

        # Правый вертикальный colorbar (если нужен)
        self.cbar = None
        self.show_colorbar = show_colorbar
        if self.show_colorbar:
            # узкая полоса справа внутри осей
            axins = inset_axes(self.ax, width="2.5%", height="90%", loc="right",
                               bbox_to_anchor=(0.05, 0., 1, 1),
                               bbox_transform=self.ax.transAxes, borderpad=0)
            self.cbar = self.fig.colorbar(self.im, cax=axins, orientation="vertical")
            self.cbar.set_label("Distance to camera (m)")

        # Аннотация
        self.text_obj = None
        self.show_text = show_text
        self.text_fmt = text_fmt

        self.fig.tight_layout(pad=0)
        self.fig.canvas.draw()

    def render(self, depth_m: np.ndarray, vmax: float | None,
               t_sec: float, H_val: float, Dc: float) -> np.ndarray:
        # авто-vmax, если не задано
        if vmax is None or not np.isfinite(vmax):
            finite = depth_m[np.isfinite(depth_m)]
            vmax = float(np.percentile(finite, 99)) if finite.size else 1.0

        self.im.set_data(depth_m)
        self.im.set_clim(0, vmax)

        # обновить colorbar шкалу
        if self.cbar is not None:
            self.cbar.update_normal(self.im)

        # текст аннотации
        if self.show_text:
            txt = self.text_fmt.format(t=t_sec, H=H_val, Dc=Dc)
            if self.text_obj is None:
                self.text_obj = self.ax.text(
                    10, 20, txt, color="white", fontsize=12, ha="left", va="top",
                    bbox=dict(facecolor="black", alpha=0.35, pad=4, edgecolor="none")
                )
            else:
                self.text_obj.set_text(txt)

        self.fig.canvas.draw()
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = self.fig.canvas.get_width_height()
        rgb = buf.reshape((h, w, 3))
        return rgb  # RGB uint8

# ---------- Основной скрипт ----------

def main():
    ap = argparse.ArgumentParser(
        description="Видео - Depth Anything V2 - калибровка по H - визуализация."
    )
    ap.add_argument("video", type=str, help="Путь к видео")
    ap.add_argument("--subs", type=str, required=True, help="Путь к .srt")
    ap.add_argument("--out", type=str, default="result.mp4", help="Выходной MP4.")

    # диапазон времени
    ap.add_argument("--t_start", type=float, default=0.0, help="Начало обработки, сек (включительно).")
    ap.add_argument("--t_end", type=float, default=None, help="Конец обработки, сек (исключая).")

    # устройство/модель
    ap.add_argument("--device", type=str, choices=["cpu","cuda"], default=None)
    ap.add_argument("--mode", type=str, choices=["disparity","depth"], default="disparity")

    # сглаживание/геометрия
    ap.add_argument("--gamma_fit", action="store_true", help="Подбор γ по ROI земли.")
    ap.add_argument("--roi_radius", type=int, default=30)
    ap.add_argument("--smooth", type=str, choices=["none","median","bilateral"], default="bilateral")
    ap.add_argument("--median_ksize", type=int, default=5)
    ap.add_argument("--bilateral_d", type=int, default=9)
    ap.add_argument("--bilateral_sigma_color", type=float, default=0.1)
    ap.add_argument("--bilateral_sigma_space", type=float, default=5.0)
    ap.add_argument("--ground_offset_m", type=float, default=0.0,
                    help="Поправка рельефа: Dc = H - ground_offset_m.")
    ap.add_argument("--tilt_deg", type=float, default=0.0,
                    help="Наклон камеры от вертикали (°): Dc ≈ (H - offset)/cos(tilt).")

    # rescale (в метрах)
    ap.add_argument("--rescale", nargs=2, type=float, metavar=("a","b"),
                    help="Сжать разброс в диапазон [a,b] м (min–max по ROI/кадру).")
    ap.add_argument("--rescale_percentiles", type=str, default="5,95")
    ap.add_argument("--rescale_source", type=str, choices=["roi","global"], default="roi")

    # matplotlib визуализация
    ap.add_argument("--cmap", type=str, default="plasma_r",
                    help="Палитра: 'gray_r','plasma_r','inferno_r','turbo_r', ...")
    ap.add_argument("--annotate", action="store_true",
                    help="Показывать текстовую аннотацию (t, H, Dc) в кадре.")
    ap.add_argument("--annot_format", type=str,
                    default="t={t:.2f}s  H={H:.2f}m  Dc≈{Dc:.2f}m",
                    help="Формат строки аннотации.")
    ap.add_argument("--colorbar", dest="colorbar", action="store_true", help="Показать правый бар высоты.")
    ap.add_argument("--no-colorbar", dest="colorbar", action="store_false", help="Скрыть правый бар высоты.")
    ap.set_defaults(colorbar=True)
    ap.add_argument("--vmax", type=float, default=None,
                    help="Фиксированный верх шкалы (м) для всех кадров. По умолчанию авто 99-й перцентиль на кадр.")

    args = ap.parse_args()

    # 1) Модель
    device = "cuda" if (args.device is None and torch.cuda.is_available()) else (args.device or "cpu")
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device=0 if device == "cuda" else -1,
    )

    # 2) H по секундам из субтитров
    sec2H = build_second_map_from_subs(args.subs)
    if not sec2H:
        raise RuntimeError("Не удалось извлечь поле H из субтитров.")

    # 3) Видео I/O
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Временные рамки
    t_start = max(0.0, float(args.t_start))
    t_end = float(args.t_end) if args.t_end is not None else (total / fps)
    t_end = max(t_end, t_start)
    start_frame = int(round(t_start * fps))
    end_frame = int(round(t_end * fps))

    # Выходной writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, fps, (W, Hh))

    # Renderer
    renderer = MPLRenderer(W, Hh, cmap=args.cmap,
                           show_text=args.annotate, text_fmt=args.annot_format,
                           show_colorbar=args.colorbar)
    cos_tilt = math.cos(math.radians(args.tilt_deg)) if abs(args.tilt_deg) > 1e-6 else 1.0

    # Перейти к стартовому кадру
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    last_H = None
    p_low, p_high = [float(x) for x in args.rescale_percentiles.split(",")] if args.rescale else (None, None)

    while True:
        if frame_idx >= end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = frame_idx / fps

        # H для текущей секунды
        H_agl = sec2H.get(int(t_sec), last_H)
        if H_agl is None:
            H_agl = sec2H.get(int(t_sec)-1, 0.5)
        last_H = H_agl

        # Dc из H (учёт рельефа и наклона)
        Dc = max(H_agl - args.ground_offset_m, 0.05)
        if cos_tilt < 1.0:
            Dc = Dc / max(cos_tilt, 1e-3)

        # кадр -> depth anything
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out = pipe(img)
        if "predicted_depth" in out:
            rel = out["predicted_depth"].squeeze(0).cpu().float().numpy()
        else:
            rel = np.array(out["depth"], dtype=np.float32)

        # сглаживание
        rel = smooth_disparity(rel,
                               mode=args.smooth,
                               median_ksize=args.median_ksize,
                               bilateral_d=args.bilateral_d,
                               bilateral_sigma_color=args.bilateral_sigma_color,
                               bilateral_sigma_space=args.bilateral_sigma_space)

        # в метры до камеры
        depth_m = to_metric_distance(torch.from_numpy(rel),
                                     center_distance_m=Dc,
                                     cx=W//2, cy=Hh//2,
                                     mode=args.mode,
                                     gamma_fit=args.gamma_fit,
                                     roi_radius=args.roi_radius)

        # rescale
        if args.rescale:
            roi = roi_values(depth_m, W//2, Hh//2, args.roi_radius) if args.rescale_source=="roi" else None
            depth_m = rescale_meters(depth_m, args.rescale[0], args.rescale[1],
                                     source=args.rescale_source,
                                     roi_vals=roi, low_q=p_low, high_q=p_high)

        # render + запись
        rgb = renderer.render(depth_m, vmax=args.vmax, t_sec=t_sec, H_val=H_agl, Dc=Dc)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

        if ((frame_idx - start_frame) % int(max(1, fps))) == 0:
            print(f"[{frame_idx}/{end_frame}] t={t_sec:.2f}s  H={H_agl:.2f}m  Dc≈{Dc:.2f}m")

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[OK] Сохранено видео: {args.out}")

if __name__ == "__main__":
    main()
