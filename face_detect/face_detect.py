import cv2 as cv
import tool
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from fer import FER#facial emotion

model = 'C:/Users/user/Desktop/Research/vision/face_detect/model/haarcascade_frontalface_default.xml'
# load the pre-trained model
classifier = CascadeClassifier(model)

def face_detect(frame):
    x,y,x2,y2 = 0,0,0,0
    bboxes = classifier.detectMultiScale(frame)
    for box in bboxes:
        x, y, width, height = box
        if width and height>100:
            if width and height<300:
                x2, y2 = x + width, y + height
                print('Face detected')
                return x,y,x2,y2 #return face
    print('Return none')
    return 0,0,0,0#return none


midframe_x,midframe_y = 320,240 #w,h
x_sen,y_sen = 20,20
detector = FER(mtcnn=True)
if __name__ == '__main__':
    print('opening camera......')
    cap = cv.VideoCapture(0)
    print('camera opened')
    while(1):
        _, frame = cap.read()
        #frame = frame.resize((640, 486))
        l = detector.detect_emotions(frame)
        emotion, score = detector.top_emotion(frame)
        x,y,x2,y2 = face_detect(frame)
        mid_x,mid_y = int((x+x2)/2),int((y+y2)/2)
        if mid_x == 0 or mid_y == 0:
            x_command,y_command = 0,0
        else:
            x_command,y_command = mid_x - midframe_x,mid_y - midframe_y # + go left up, - go right down
        if emotion is not None:
            cv.putText(frame,'fear: '+str(l[0]['emotions']['fear']) ,(20,40), cv.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2,cv.LINE_AA)
            cv.putText(frame,'angry: '+str(l[0]['emotions']['angry']) ,(20,60), cv.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2,cv.LINE_AA)
            cv.putText(frame,'disgust: '+str(l[0]['emotions']['disgust']) ,(20,80), cv.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2,cv.LINE_AA)
            cv.putText(frame,'happy: '+str(l[0]['emotions']['happy']) ,(20,100), cv.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2,cv.LINE_AA)
            cv.putText(frame,'sad: '+str(l[0]['emotions']['sad']) ,(20,120), cv.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2,cv.LINE_AA)
            cv.putText(frame,'neutral: '+str(l[0]['emotions']['neutral']) ,(20,140), cv.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2,cv.LINE_AA)
            cv.putText(frame,'surprise: '+str(l[0]['emotions']['surprise']) ,(20,160), cv.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2,cv.LINE_AA)
            cv.putText(frame,str(emotion) ,(x ,y2 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2,cv.LINE_AA)
            
        tool.draw_rect(x,y,x2,y2,frame,(255,0,0))
        cv.putText(frame,'x,y = ' + str(x_command) + ','+ str(y_command) ,(20,20), cv.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2,cv.LINE_AA)
        cv.imshow('frame',frame)
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break

    
cap.release()
cv.destroyAllWindows() 



   


