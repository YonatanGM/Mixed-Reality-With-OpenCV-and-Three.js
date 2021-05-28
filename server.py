import numpy as np
import cv2 as cv
import glob
import threading
import json
from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket


server = None
clients = []


class SimpleWSServer(WebSocket):
    def handleConnected(self):
        clients.append(self)
    def handleClose(self):
        clients.remove(self)


def run_server():
    global server
    server = SimpleWebSocketServer('', 9000, SimpleWSServer,
                                   selectInterval=(1000.0 / 60) / 1000)
    server.serveforever()

t = threading.Thread(target=run_server)
t.start()

def draw(img, corners, imgpts):
    corners = corners.astype(int)
    imgpts = imgpts.astype(int)
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    img = cv.circle(img, tuple(corners[24].ravel()), radius=3, color=(255, 0, 0), thickness=3)
    return img

def rot_params(rvecs):
    from math import pi,atan2,asin
    R = cv.Rodrigues(rvecs)[0]
    roll = 180*atan2(-R[2][1], R[2][2])/pi
    pitch = 180*asin(R[2][0])/pi
    yaw = 180*atan2(-R[1][0], R[0][0])/pi
    rot_params= (roll,pitch,yaw)
    return rot_params


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('.\calibration\left*.jpg')
print(images)
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1)
cv.destroyAllWindows()

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
cap = cv.VideoCapture(0)
w = 1000
while True:
    _, img = cap.read()
    scale = w /  img.shape[1]
    h = int(img.shape[0] * scale)
    img = cv.resize(img, (0,0), fx=scale, fy=scale)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7,6),None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img, corners2, imgpts)
        #spot_dir = imgpts[2].ravel() - corners2[0].ravel()
        #spot_dir /= cv.norm(spot_dir)
        x, y = tuple(corners2[24].ravel())
        x /= w
        y /= h 
        roll, pitch, yaw = rot_params(rvecs)
        
        cv.imshow('img',img)
        for client in clients:
            client.sendMessage(str(json.dumps({'x':x, 'y':y, 'roll':roll, 'pitch':pitch, 'yaw':yaw})))

    if cv.waitKey(1) == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()


