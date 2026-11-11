import cv2
import numpy as np
from djitellopy import Tello

drone = Tello()
drone.connect()
print("Battery:", drone.get_battery(), "%")
drone.streamon()
frame_read = drone.get_frame_read()

patternSize = (9, 6)

objp = np.zeros((patternSize[0] * patternSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)

objectPoints = []
imagePoints = []

winSize = (11, 11)
zeroZone = (-1, -1)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

print("put the chessboard into the camera")
print("press c over 4 times.")
print("press q to end")

gray = None
try:
    while True:
        frame = frame_read.frame
        if frame is None:
            continue
        frame = frame.copy()   

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret_corner, corners = cv2.findChessboardCorners(gray, patternSize, None)

        show = frame.copy()
        if ret_corner:
            cv2.drawChessboardCorners(show, patternSize, corners, ret_corner)

        cv2.putText(show, f"captured: {len(imagePoints)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("frame", show)
        key = cv2.waitKey(33) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("c"):
            if ret_corner:
                corners_refined = cv2.cornerSubPix(gray, corners, winSize, zeroZone, criteria)
                imagePoints.append(corners_refined)
                objectPoints.append(objp)
                print(f"Success! Currently {len(imagePoints)}")
            else:
                print("No complete set of chessboard corners detected in the frame, so nothing was saved.")

finally:
    drone.streamoff()
    cv2.destroyAllWindows()

if len(imagePoints) > 4:
    print("\nComputing camera parameters...")
    imageSize = gray.shape[::-1]  

    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints, imagePoints, imageSize, None, None
    )
    print("Reprojection error:", ret)

    filename = "calibrate.xml"
    f = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    f.write("intrinsic", cameraMatrix)
    f.write("distortion", distCoeffs)
    f.release()

    print(f"Saved successfully! File name: {filename}")
else:
    print(f"\nNot enough images captured ({len(imagePoints)}); you need at least 5.")