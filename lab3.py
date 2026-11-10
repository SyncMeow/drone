import sys
import cv2
import numpy as np
import os

def calc_hist(gray):
    hist = np.zeros(256, dtype=np.int64)
    values, counts = np.unique(gray, return_counts=True)
    hist[values] = counts
    return hist

def otsu_threshold(gray):
    hist = calc_hist(gray)
    total = gray.size
    levels = np.arange(256)
    best_t, best_score = 0, np.inf
    for t in range(256):
        h0, l0 = hist[:t + 1], levels[:t + 1]
        h1, l1 = hist[t + 1:], levels[t + 1:]
        n0, n1 = h0.sum(), h1.sum()
        if n0 == 0 or n1 == 0:
            continue
        m0 = (l0 * h0).sum() / n0
        m1 = (l1 * h1).sum() / n1
        v0 = (((l0 - m0) ** 2) * h0).sum() / n0
        v1 = (((l1 - m1) ** 2) * h1).sum() / n1
        score = n0 / total * v0 + n1 / total * v1
        if score < best_score:
            best_score, best_t = score, t
    return best_t

class UnionFind:
    def __init__(self):
        self.parent = [0]

    def make_label(self):
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb

def connected_components(binary, connectivity=8):
    fg = binary > 0
    h, w = fg.shape
    labels = np.zeros((h, w), dtype=np.int32)
    uf = UnionFind()

    if connectivity == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]
    else:
        offsets = [(-1, 0), (0, -1)]

    for y in range(h):
        for x in range(w):
            if not fg[y, x]:
                continue

            neighbors = []
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] > 0:
                    neighbors.append(labels[ny, nx])

            if not neighbors:
                labels[y, x] = uf.make_label()
            else:
                m = min(neighbors)
                labels[y, x] = m
                for n in neighbors:
                    if n != m:
                        uf.union(m, n)

    n_labels = len(uf.parent)
    lut = np.zeros(n_labels, dtype=np.int32)
    root_to_new = {}
    for lbl in range(1, n_labels):
        root = uf.find(lbl)
        if root not in root_to_new:
            root_to_new[root] = len(root_to_new) + 1
        lut[lbl] = root_to_new[root]

    labels = lut[labels]
    return labels, len(root_to_new)

def colorize(labels, num, seed=0):
    rng = np.random.default_rng(seed)
    colors = rng.integers(50, 256, size=(num + 1, 3), dtype=np.uint8)
    colors[0] = (0, 0, 0)
    return colors[labels]

def connected_components_with_stats(binary, connectivity=8):
    fg = binary > 0
    h, w = fg.shape
    labels = np.zeros((h, w), dtype=np.int32)
    uf = UnionFind()
    stats = [None] # stats[label] = [area, x1, y1, x2, y2]

    if connectivity == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]
    else:
        offsets = [(-1, 0), (0, -1)]

    # row-major visit (from up to down and then from left to right)
    for y, x in np.argwhere(fg):
        neighbors = []
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] > 0:
                neighbors.append(labels[ny, nx])

        if not neighbors:
            lbl = uf.make_label()              
            stats.append([0, x, y, x, y])
        else:
            lbl = min(neighbors)                
            for n in neighbors:
                if n != lbl:
                    uf.union(lbl, n)           
        labels[y, x] = lbl

        s = stats[lbl]
        s[0] += 1
        s[1] = min(s[1], x)
        s[2] = min(s[2], y)
        s[3] = max(s[3], x)
        s[4] = max(s[4], y)

    merged = {}
    for lbl in range(1, len(stats)):
        root = uf.find(lbl)
        a, x1, y1, x2, y2 = stats[lbl]
        if root not in merged:
            merged[root] = [a, x1, y1, x2, y2]
        else:
            m = merged[root]
            m[0] += a
            m[1] = min(m[1], x1)
            m[2] = min(m[2], y1)
            m[3] = max(m[3], x2)
            m[4] = max(m[4], y2)

    lut = np.arange(len(stats), dtype=np.int32)
    for lbl in range(1, len(stats)):
        lut[lbl] = uf.find(lbl)
    labels = lut[labels]

    return labels, merged

def draw_video(video_path, area_threshold, show=True, save_path=None):
    cap = cv2.VideoCapture(video_path)

    backSub = cv2.createBackgroundSubtractorMOG2(history=100)
    shadow_val = backSub.getShadowValue()

    writer = None
    if save_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))

    scale = 0.5
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # scale smaller to run faster
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        fgmask = backSub.apply(small)

        _, nmask = cv2.threshold(fgmask, shadow_val, 255, cv2.THRESH_BINARY)

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # nmask = cv2.morphologyEx(nmask, cv2.MORPH_OPEN, kernel)
        # nmask = cv2.dilate(nmask, kernel, iterations=2)

        _, stats = connected_components_with_stats(nmask)

        for area, x1, y1, x2, y2 in stats.values():
            if area > area_threshold * scale * scale:
                cv2.rectangle(frame,
                              (int(x1 / scale), int(y1 / scale)),
                              (int(x2 / scale), int(y2 / scale)),
                              (0, 255, 0), 2)

        if writer:
            writer.write(frame)
            
        if show:
            cv2.imshow("frame", cv2.resize(frame, None, fx=0.5, fy=0.5))
            if cv2.waitKey(33) & 0xFF == 27:    # press ESC to early exit
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    img_path = os.path.join(script_dir, f"imgs/lab3", "q1.png")
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    t = otsu_threshold(gray)
    binary = np.where(gray > t, 255, 0).astype(np.uint8)

    out_path = os.path.join(script_dir, f"out/lab3", "a1_otsu.png")
    cv2.imwrite(out_path, binary)
    print("Otsu threshold: ", t)

    labels, num = connected_components(binary, connectivity=8)
    print("Number of areas: ", num)

    result = colorize(labels, num)
    out_path = os.path.join(script_dir, f"out/lab3", "a1_cc.png")
    cv2.imwrite(out_path, result)

    video_path = os.path.join(script_dir, f"imgs/lab3", "q2.mp4")
    out_path = os.path.join(script_dir, f"out/lab3", "a2.mp4")
    draw_video(video_path, area_threshold=4000, show=True, save_path=out_path)