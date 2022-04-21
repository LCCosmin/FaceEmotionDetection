from deepface import DeepFace
import cv2
from EmotionDetectionModel import EmotionDetectionModel
from utils import normalize

def main():
    
    #Load the HaarCascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    #Open Camera
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    err = 0
    frame = 0
    
    while True:
        _, img = cap.read()
        img = normalize(img)
        faces = face_cascade.detectMultiScale(img, 1.1, 3)
        
        frame = frame + 1
        for (x, y, w, h) in faces:
            try:
                crop_img = img[y:y+h+10, x:x+w+10]
                cv2.rectangle(img, (x-20, y-20), (x+w+20, y+h+20), (255, 0, 0), 2)
                
                model = EmotionDetectionModel()
                model.load_weights('./brain')
                emotion = model.evaluate()
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                #emotion = DeepFace.analyze(crop_img, actions=['emotion'],enforce_detection=False)
                cv2.putText(img, emotion, (x-40,y-40), font, 1, (200,255,155))
                cv2.imshow('App', img)
                print(". Frame no:" + str(frame))
            except:
                print("No face :(. Error no: " + str(err) + ". Frame no: " + str(frame))
                err = err + 1
        if not False:
            cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
def main2():
    model = EmotionDetectionModel()
    model.train()
    
if __name__ == "__main__":
    main2()