import sys
import cv2
import numpy as np

def bilinear_sample(img, xs, ys):
    h, w = img.shape[:2]
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    dx = (xs - x0)[..., None]
    dy = (ys - y0)[..., None]

    img = img.astype(np.float32)
    if img.ndim == 2:
        img = img[..., None]

    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    top = Ia * (1 - dx) + Ib * dx
    bottom = Ic * (1 - dx) + Id * dx
    out = top * (1 - dy) + bottom * dy
    return out

def resize_bilinear_interpolate(img, scale):
    h, w = img.shape[:2]
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    sx, sy = w / new_w, h / new_h
    xs = (np.arange(new_w) + 0.5) * sx - 0.5
    ys = (np.arange(new_h) + 0.5) * sy - 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)

    out = bilinear_sample(img, grid_x, grid_y)
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out[..., 0] if img.ndim == 2 else out


def select_points(img, win="select 4 points"):
    points = []
    show = img.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(show, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(show, str(len(points)), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if len(points) > 1:
                cv2.line(show, points[-2], points[-1], (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(show, points[3], points[0], (0, 255, 0), 2)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, show)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(points) == 4:
            break
        if key == ord('r'):
            points.clear()
            show[:] = img
    cv2.destroyWindow(win)
    return np.float32(points)

def build_mapping(dst_pts, frame_w, frame_h):
    src_pts = np.float32([[0, 0],
                          [frame_w - 1, 0],
                          [frame_w - 1, frame_h - 1],
                          [0, frame_h - 1]])

    M = cv2.getPerspectiveTransform(dst_pts, src_pts)

    x_min, y_min = np.floor(dst_pts.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(dst_pts.max(axis=0)).astype(int)
    xs, ys = np.meshgrid(np.arange(x_min, x_max + 1),
                         np.arange(y_min, y_max + 1))

    ones = np.ones_like(xs)
    pts = np.stack([xs, ys, ones], axis=0).reshape(3, -1).astype(np.float64)
    u, v, w = M @ pts
    u, v = u / w, v / w                               

    inside = (u >= 0) & (u <= frame_w - 1) & (v >= 0) & (v <= frame_h - 1)

    return {
        "bg_x": xs.ravel()[inside], "bg_y": ys.ravel()[inside],
        "src_x": u[inside], "src_y": v[inside],
    }


def project_frame(bg, frame, mapping):
    out = bg.copy()
    vals = bilinear_sample(frame, mapping["src_x"], mapping["src_y"])
    h, w = out.shape[:2]
    keep = (mapping["bg_x"] >= 0) & (mapping["bg_x"] < w) & \
           (mapping["bg_y"] >= 0) & (mapping["bg_y"] < h)
    out[mapping["bg_y"][keep], mapping["bg_x"][keep]] = \
        np.clip(np.round(vals[keep]), 0, 255).astype(np.uint8)
    return out

def main(bg_path, cam_id=0, frame_scale=0.5):
    bg = cv2.imread(bg_path)

    dst_pts = select_points(bg)

    cap = cv2.VideoCapture(cam_id)

    mapping = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_bilinear_interpolate(frame, frame_scale)

        if mapping is None:
            fh, fw = frame.shape[:2]
            mapping = build_mapping(dst_pts, fw, fh)

        result = project_frame(bg, frame, mapping)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1])