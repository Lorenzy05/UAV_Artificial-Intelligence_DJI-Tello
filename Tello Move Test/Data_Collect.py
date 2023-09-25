import cv2
import time
from djitellopy import Tello

Drone = Tello()
Drone.connect()

Drone.streamon()
frame_read = Drone.get_frame_read()
print(Drone.get_battery())

Lt = 1
Rt = 1
Ft = 1

while True:
    img = frame_read.frame
    cv2.imshow("drone", img)

    key = cv2.waitKey(1) & 0xff

    if key == 27: # ESC
        break
    elif key == ord('t'):
        Drone.takeoff()
    elif key == ord('w'):
        Drone.send_rc_control(0, 20, 0, 0)
        time.sleep(0.3)
        Drone.send_rc_control(0, 0, 0, 0)
    elif key == ord('s'):
        Drone.send_rc_control(0, -20, 0, 0)
        time.sleep(0.3)
        Drone.send_rc_control(0, 0, 0, 0)
    elif key == ord('a'):
        Drone.send_rc_control(-20, 0, 0, 0)
        time.sleep(0.3)
        Drone.send_rc_control(0, 0, 0, 0)
    elif key == ord('d'):
        Drone.send_rc_control(20, 0, 0, 0)
        time.sleep(0.3)
        Drone.send_rc_control(0, 0, 0, 0)
    elif key == ord('e'):
        Drone.send_rc_control(0, 0, 0, 20)
    elif key == ord('q'):
        Drone.send_rc_control(0, 0, 0, -20)

    elif key == ord('x'):
        Drone.send_rc_control(0, 0, -80, 0)
        time.sleep(0.3)
        Drone.send_rc_control(0, 0, 0, 0)

    elif key == ord('l'):
        filename = "../File_Paper-Road/Left/" + str(Lt) + ".jpg"
        cv2.imwrite(filename, img)
        Lt += 1
    elif key == ord('r'):
        filename = "../File_Paper-Road/Right/" + str(Rt) + ".jpg"
        cv2.imwrite(filename, img)
        Rt += 1
    elif key == ord('f'):
        filename = "../File_Paper-Road/Forward/" + str(Ft) + ".jpg"
        cv2.imwrite(filename, img)
        Ft += 1

Drone.land()




