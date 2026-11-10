import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calc_hist(channel):
    hist = np.zeros(256, dtype=np.int64)
    values, counts = np.unique(channel, return_counts=True)
    hist[values] = counts
    return hist

def calc_cdf(hist):
    cdf = np.zeros(256, dtype=np.int64)
    running = 0
    for k in range(256):
        running += hist[k]
        cdf[k] = running
    return cdf

def equalize_channel(channel):
    hist = calc_hist(channel)
    cdf = calc_cdf(hist)

    total = channel.size
    cdf_min = cdf[cdf > 0][0]
    if total == cdf_min:
        return channel.copy()

    lut = np.round((cdf - cdf_min) / (total - cdf_min) * 255)
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    return lut[channel]

def equalize_rgb(img_bgr):
    b, g, r = cv2.split(img_bgr)
    b_eq = equalize_channel(b)
    g_eq = equalize_channel(g)
    r_eq = equalize_channel(r)
    return cv2.merge([b_eq, g_eq, r_eq])

def equalize_hsv_v(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = equalize_channel(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

def plot_results(original, rgb_eq, hsv_eq, path):
    imgs = [original, rgb_eq, hsv_eq]
    titles = ["Original", "RGB equalized", "HSV (V) equalized"]
    colors = ("b", "g", "r")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(title)
        axes[0, i].axis("off")
        for c, col in enumerate(colors):
            axes[1, i].plot(calc_hist(img[:, :, c]), color=col)
        axes[1, i].set_xlim([0, 255])
        axes[1, i].set_title(title + " histogram")
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def otsu_threshold(gray):
    hist = calc_hist(gray)
    total = gray.size
    levels = np.arange(256)
 
    best_t = 0
    best_score = np.inf
 
    for t in range(256):
        # <= t back group，> t front group
        h0, l0 = hist[:t + 1], levels[:t + 1]
        h1, l1 = hist[t + 1:], levels[t + 1:]
 
        n0, n1 = h0.sum(), h1.sum()
        if n0 == 0 or n1 == 0:
            continue

        mean0 = (l0 * h0).sum() / n0
        mean1 = (l1 * h1).sum() / n1
        var0 = (((l0 - mean0) ** 2) * h0).sum() / n0
        var1 = (((l1 - mean1) ** 2) * h1).sum() / n1
 
        w0, w1 = n0 / total, n1 / total
        score = w0 * var0 + w1 * var1

        if score < best_score:
            best_score = score
            best_t = t
 
    return best_t

def apply_threshold(gray, t):
    # > threshold -> 255, otherwise -> 0
    return np.where(gray > t, 255, 0).astype(np.uint8)

def plot_otsu_hist(gray, t, path):
    hist = calc_hist(gray)
    levels = np.arange(256)
    total = gray.size
 
    scores = np.full(256, np.nan)
    for k in range(256):
        h0, h1 = hist[:k + 1], hist[k + 1:]
        n0, n1 = h0.sum(), h1.sum()
        if n0 == 0 or n1 == 0:
            continue
        m0 = (levels[:k + 1] * h0).sum() / n0
        m1 = (levels[k + 1:] * h1).sum() / n1
        v0 = (((levels[:k + 1] - m0) ** 2) * h0).sum() / n0
        v1 = (((levels[k + 1:] - m1) ** 2) * h1).sum() / n1
        scores[k] = n0 / total * v0 + n1 / total * v1
 
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(levels[:t + 1], hist[:t + 1], width=1, color="tab:blue",
            label="Group 0 (<= T)")
    ax1.bar(levels[t + 1:], hist[t + 1:], width=1, color="tab:orange",
            label="Group 1 (> T)")
    ax1.axvline(t, color="red", linestyle="--", linewidth=2,
                label=f"Otsu threshold T = {t}")
    ax1.set_xlim(0, 255)
    ax1.set_xlabel("Intensity")
    ax1.set_ylabel("Pixel count")
 
    ax2 = ax1.twinx()
    ax2.plot(levels, scores, color="gray", linewidth=1.5,
             label="Within-class variance")
    ax2.set_ylabel("Within-class variance")
 
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Otsu Threshold Histogram")
 
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    img_path = os.path.join(script_dir, f"imgs/lab2", "q1.jpg")
    img = cv2.imread(img_path)
 
    rgb_eq = equalize_rgb(img)
    hsv_eq = equalize_hsv_v(img)

    out_path = os.path.join(script_dir, f"out/lab2", "q1_1.jpg")
    cv2.imwrite(out_path, rgb_eq)
    out_path = os.path.join(script_dir, f"out/lab2", "q1_2.jpg")
    cv2.imwrite(out_path, hsv_eq)
    # out_path = os.path.join(script_dir, f"out/lab2", "q1_comp.jpg")
    # plot_results(img, rgb_eq, hsv_eq, out_path)

    img_path = os.path.join(script_dir, f"imgs/lab2", "q2.jpg")
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = otsu_threshold(gray)
    otsu = apply_threshold(gray, t)
    print(f"threshold of otsu: {t}")
    
    out_path = os.path.join(script_dir, f"out/lab2", "q2.jpg")
    cv2.imwrite(out_path, otsu)
    
    # out_path = os.path.join(script_dir, f"out/lab2", "q2_thres.jpg")
    # plot_otsu_hist(gray, t, out_path)
