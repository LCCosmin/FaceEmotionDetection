from deepface import DeepFace
import cv2

def main():
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    
    while True:
        found = False
        _, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x-20, y-20), (x+w+20, y+h+20), (255, 0, 0), 2)
            found = True
            try:
                font = cv2.FONT_HERSHEY_SIMPLEX
                crop_img = img[y:y+h, x:x+w]
                cv2.imshow('test', crop_img)
                emotion = DeepFace.analyze(crop_img, actions=['emotion'],enforce_detection=False)
                #print("Got here1")
                cv2.putText(img, emotion['dominant_emotion'], (x-40,y-40), font, 1, (200,255,155))
                cv2.imshow('img', img)
            except:
                print("No face :(")
        if not False:
            cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()