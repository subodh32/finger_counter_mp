import cv2
import mediapipe as mp
import math

def fingerRaised(id, list):

    pt1 = id*4 
    pt2 = id*4 - 2

    x0 = y0 = x1 = y1 = x2 = y2 = 0

    if(list != []):

        x1,y1 = list[pt1][1], list[pt1][2]
        x2,y2 = list[pt2][1], list[pt2][2]

        if(id == 1):
            x2,y2 = list[1][1], list[1][2]
            x0,y0 = list[17][1], list[17][2]
        else:
            x0,y0 = list[0][1] , list[0][2]
        


    dist1 = math.sqrt((x1-x0)**2 + (y1-y0)**2)
    dist2 = math.sqrt((x2-x0)**2 + (y2-y0)**2)

    if(dist1 > dist2):
        return True
    
    return False



vid = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode = False,
    model_complexity = 1,
    min_detection_confidence = 0.75,
    min_tracking_confidence = 0.75,
    max_num_hands = 2
)

Draw = mp.solutions.drawing_utils

while True:
    
    ret,frame = vid.read()
    frame = cv2.resize(frame,(480,480))
    frame = cv2.flip(frame, 1)

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Process = hands.process(frameRGB)

    landmarklist = []

    if Process.multi_hand_landmarks:
        for handlm in Process.multi_hand_landmarks:

            for _id,landmarks in enumerate(handlm.landmark):
                height,width,colorchannel = frame.shape
                x,y = int(landmarks.x*width), int(landmarks.y*height)
                landmarklist.append([_id,x,y])

            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

    
    raised_fingers = 0

    for i in range(1,5 + 1):
        if(fingerRaised(i,landmarklist)):
            raised_fingers += 1

    cv2.putText(frame, "number of fingers" + str(raised_fingers), (240,240),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)

    cv2.imshow('frame',frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

vid.release()
cv2.destroyAllWindows()
