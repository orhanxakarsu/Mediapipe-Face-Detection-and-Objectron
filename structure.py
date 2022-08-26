import cv2
import mediapipe as mp


class MediaPipeModels:
    
    def __init__(self):
        self.ObjectModelNames = ['Shoe','Chair','Camera','Cup']
        
        
        self.mpDrawing = mp.solutions.drawing_utils
        self.faceDetection = mp.solutions.face_detection
        self.objectron = mp.solutions.objectron
        
        self.currentObjectModelName = ''
        
        self.faceDetectionModel = self.faceDetection.FaceDetection(model_selection=1,
                                                                   min_detection_confidence=0.5)
        
        self.objectModels =[self.objectron.Objectron(static_image_mode=True,
                                                     max_num_objects=10,
                                                     min_detection_confidence=0.5,
                                                     min_tracking_confidence=0.8,
                                                     model_name=objectName) for objectName in self.ObjectModelNames]
        
        
        
    
    def onlyFaceDetection(self,img,isRgb=False):
        if not isRgb:
            results = self.faceDetectionModel.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            results = self.faceDetectionModel.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        #Tespit edilen yüz varsa gerekli işlemleri gerçekleştir, yoksa gerçekleştirme :
        if results.detections: 
            for detection in results.detections:
                self.mpDrawing.draw_detection(img, detection)
        
        return img


    def onlyObjectDetection(self,img,modelName,isRgb=False):
        if modelName != self.currentObjectModelName:
            self.ObjectDetectionModel = self.objectron.Objectron(static_image_mode=False,
                                                                 max_num_objects=2,
                                                                 min_detection_confidence=0.5,
                                                                 min_tracking_confidence=0.8,
                                                                 model_name=modelName)
            self.currentObjectModelName=modelName
        
        if not isRgb:
            results = self.ObjectDetectionModel.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            results = self.ObjectDetectionModel.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        
        if results.detected_objects:
            
            for detected_object in results.detected_objects:
                self.mpDrawing.draw_landmarks(img,
                                              detected_object.landmarks_2d,
                                              self.objectron.BOX_CONNECTIONS)
                self.mpDrawing.draw_axis(img,
                                         detected_object.rotation,
                                         detected_object.translation)
                
        
        return img
    
    
    
    def faceAndObjectDetection(self,img,modelName,isRgb = False):
        
        if modelName != self.currentObjectModelName:
            self.ObjectDetectionModel = self.objectron.Objectron(static_image_mode=False,
                                                                 max_num_objects=2,
                                                                 min_detection_confidence=0.5,
                                                                 min_tracking_confidence=0.8)
            self.currentObjectModelName=modelName
        
        
        if not isRgb:
            faceResults = self.faceDetectionModel.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            objectResults = self.ObjectDetectionModel.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            faceResults = self.faceDetectionModel.process(img)
            objectResults = self.ObjectDetectionModel.process(img) 
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        
        
        if faceResults.detections: 
            for detection in faceResults.detections:
                self.mpDrawing.draw_detection(img, detection)
        
        
        
        if objectResults.detected_objects:
            
            for detected_object in objectResults.detected_objects:
                self.mpDrawing.draw_landmarks(img,
                                              detected_object.landmarks_2d,
                                              self.objectron.BOX_CONNECTIONS)
                self.mpDrawing.draw_axis(img,
                                         detected_object.rotation,
                                         detected_object.translation)
        return img
        
    
    def detectionOfBooleanValue(self,img,boolList : list):
        if boolList[0]:
            faceResults = self.faceDetectionModel.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        objectList = []
        
        for i in range(1,len(boolList)):
            if boolList[i]:
                objectList.append(self.objectModels[i-1].process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        
        
        if boolList[0]:
            if faceResults.detections: 
                for detection in faceResults.detections:
                    self.mpDrawing.draw_detection(img, detection)
        
        
        
        
        for objectResult in objectList:
            if objectResult.detected_objects:
                for detected_object in objectResult.detected_objects:
                    self.mpDrawing.draw_landmarks(img,
                                                  detected_object.landmarks_2d,
                                                  self.objectron.BOX_CONNECTIONS)
                    self.mpDrawing.draw_axis(img,
                                             detected_object.rotation,
                                             detected_object.translation)
        
        return img
                
        



    
    