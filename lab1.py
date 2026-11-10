import numpy as np
import cv2
import os

def show(img):
    cv2.imshow("Image: ", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save(img, path):
    success = cv2.imwrite(path, img)  
    return success

def keep_blue(path):
    img = cv2.imread(path)
    r = img[:, :, 2].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 0].astype(np.float32)

    mask = (b > 100) & (b > 1.5 * r) & (b > 1.2 * g)
    mask = ~mask

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    img[mask] = gray_bgr[mask]
    return img

def mod_contrast_brightness(path, contrast=50, brightness=20):
    img = cv2.imread(path)
    r = img[:, :, 2].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 0].astype(np.float32)

    blue = (b > 1.5 * r) & (b > 1.2 * g)
    yellow = (r > 1.5 * b) & (g > 1.5 * b)
    mask = blue | yellow

    adj_img = img.astype(np.float32)

    factor = (contrast / 127.0) + 1.0
    adj_img = (adj_img - 127.0) * factor + 127.0 + brightness

    adj_img = np.clip(adj_img, 0, 255).astype(np.uint8)

    img[mask] = adj_img[mask]
    return img

def resize_bilinear_interpolate(path, scale=3.0):
    img = cv2.imread(path)
    H, W, C = img.shape
    new_H, new_W = int(H * scale), int(W * scale)

    dst_x = np.arange(new_W)
    dst_y = np.arange(new_H)
    src_x = (dst_x + 0.5)/scale - 0.5
    src_y = (dst_y + 0.5)/scale - 0.5

    src_x = np.clip(src_x, 0, W-1)
    src_y = np.clip(src_y, 0, H-1)

    xx = src_x[None, :]
    yy = src_y[:, None]

    x1 = np.floor(xx).astype(np.int32)
    y1 = np.floor(yy).astype(np.int32)
    x2 = np.clip(x1+1, 0, W-1)
    y2 = np.clip(y1+1, 0, H-1)

    u = (xx-x1)[:, :, None]
    w = (yy-y1)[:, :, None]

    q11 = img[y1, x1].astype(np.float32)
    q21 = img[y1, x2].astype(np.float32)
    q12 = img[y2, x1].astype(np.float32)
    q22 = img[y2, x2].astype(np.float32)

    out = (1.0-u)*(1.0-w)*q11 + (1.0-u)*w*q12 + u*(1.0-w)*q21 + u*w*q22

    return np.clip(out, 0, 255).astype(np.uint8)

def filtering_sobel(path):
    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)

    kernel_x = np.array([
        [ 1,  0, -1],
        [ 2,  0, -2],
        [ 1,  0, -1]
    ], dtype=np.float64)

    kernel_y = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ], dtype=np.float64)

    gx = cv2.filter2D(blur, cv2.CV_64F, kernel_x)
    gy = cv2.filter2D(blur, cv2.CV_64F, kernel_y)

    g = np.sqrt(gx**2 + gy**2)
    edge_img = np.clip(g, 0, 255).astype(np.uint8)

    return edge_img

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    N = 4
    LAB = 1
    enable_mask = 0b1111
    show_mask = 0b0000

    img_name = ["q1.jpg", "q1.jpg", "q2.jpg", "q3.jpg"]
    out_name = ["a1_1.jpg", "a1_2.jpg", "a2.jpg", "a3.jpg"]
    func = [
        keep_blue,                   # q1-1
        mod_contrast_brightness,     # q1-2
        resize_bilinear_interpolate, # q2
        filtering_sobel              # q3
    ]

    for i in range(N):
        if enable_mask & 1:
            print(f"Running Task {i}...")

            img_path = os.path.join(script_dir, f"imgs/lab{LAB}", img_name[i])
            out_path = os.path.join(script_dir, f"out/lab{LAB}", out_name[i])

            res = func[i](img_path)
            if save(res, out_path): 
                print(f"Saved {out_name[i]} successfully")

            if show_mask & 1: 
                show(res)
            show_mask >>= 1
            
        else:
            print(f"Skipping Task {i}...")

        enable_mask >>= 1
