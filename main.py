import cv2
import numpy as np
cam= cv2.VideoCapture(0)

def nothing(x):
    pass
_,background=cam.read()

kernel=np.ones((3,3),np.uint8)
kernel2=np.ones((11,11),np.uint8)
kernel3=np.ones((3,3),np.uint8)
while(cam.isOpened()):
    _, frame=cam.read()
    lower = np.array([70, 210, 19])
    upper = np.array([100, 255, 71])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,lower,upper)

    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel2)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel3)

    mask_not=cv2.bitwise_not(mask)
    bg=cv2.bitwise_and(background,background,mask=mask)
    fg=cv2.bitwise_and(frame,frame,mask=mask_not)
    dts=cv2.addWeighted(bg,1,fg,1,0)
    cv2.imshow("orjinal",frame)
    cv2.imshow("mask",mask)
    cv2.imshow("dts",dts)



    if cv2.waitKey(1) & 0xFF == ord("q"):

        break
cam.release()
cv2.destroyAllWindows()