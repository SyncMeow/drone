import cv2
import numpy as np
import math
from djitellopy import Tello

MARKER_SIZE = 15.0                      
ARUCO_DICT = cv2.aruco.DICT_4X4_250

def load_calib(path="calibrate.xml"):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("intrinsic").mat()
    dist = fs.getNode("distortion").mat()
    fs.release()
    return K, dist

def make_detector():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    params = aruco.DetectorParameters_create()    
    return lambda g: aruco.detectMarkers(g, dictionary, parameters=params)

def draw_axes(img, K, dist, rvec, tvec, length):
    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(img, K, dist, rvec, tvec, length)
    else:
        cv2.aruco.drawAxis(img, K, dist, rvec, tvec, length)

def main():
    K, dist = load_calib()
    detect = make_detector()

    half = MARKER_SIZE / 2
    obj_pts = np.array([[-half,  half, 0],
                        [ half,  half, 0],
                        [ half, -half, 0],
                        [-half, -half, 0]], dtype=np.float32)

    drone = Tello()
    drone.connect()
    drone.streamon()
    frame_read = drone.get_frame_read()

    while True:
        frame = frame_read.frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for c, mid in zip(corners, ids.flatten()):
                ok, rvec, tvec = cv2.solvePnP(obj_pts, c.reshape(4, 2).astype(np.float32),
                                              K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok:
                    continue
                x, y, z = tvec.flatten()
                d = math.sqrt(x * x + y * y + z * z)   

                draw_axes(frame, K, dist, rvec, tvec, MARKER_SIZE)

                cx, cy = c[0].mean(axis=0).astype(int)
                cv2.putText(frame, f"x: {x:.3f}  y: {y:.3f}  z: {z:.3f}",
                            (cx - 150, cy - 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"id: {mid}  dist: {d:.2f} cm",
                            (cx - 150, cy - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

        cv2.imshow("drone", frame)
        if cv2.waitKey(33) & 0xFF == 27:   
            break

    cv2.destroyAllWindows()
    drone.streamoff()

if __name__ == '__main__':
    main()