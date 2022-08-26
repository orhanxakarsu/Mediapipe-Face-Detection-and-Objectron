#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import tkinter as tk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo 
from yapi1 import *

window = tk.Tk()
window.title('Detection')
window.geometry('600x350')
window.title("Face and Object Detection App")

mdl =MediaPipeModels()
def start_video():
    

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
      success, image = cap.read()
      image = cv2.flip(image, 1)
      img =mdl.detectionOfBooleanValue(image,[face.get(),
                                              shoe.get(),
                                              chair.get(),
                                              camera.get(),
                                              cup.get()])
      
      
      cv2.imshow('Face Or Object Detection',img)
      if cv2.waitKey(5) & 0xFF == 27:
          cv2.destroyAllWindows()
          break


def select_file():

    
    fileName = fd.askopenfilename(
        title='Open a file',
        initialdir='/')
    
    img = cv2.imread(fileName)
    processImg = img =mdl.detectionOfBooleanValue(img,[face.get(),
                                                         shoe.get(),
                                                         chair.get(),
                                                         camera.get(),
                                                         cup.get()])
    cv2.imshow('Face Or Object Detection',processImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
 

 
face = tk.IntVar()
shoe = tk.IntVar()
chair = tk.IntVar()
camera = tk.IntVar()
cup =tk.IntVar()


# Sıralama :Face -> Shoe -> Chair ->Camera -> Cup

faceCheckBox = tk.Checkbutton(window, text='Face Detection',variable=face, onvalue=True, offvalue=False)
faceCheckBox.grid(row=0,column=0,padx=50,pady=50)


shoeCheckBox = tk.Checkbutton(window, text='Shoe Detection',variable=shoe, onvalue=True, offvalue=False)
shoeCheckBox.grid(row=1,column=0,padx=5,pady=5)


chairCheckBox = tk.Checkbutton(window, text='Chair Detection',variable=chair, onvalue=True, offvalue=False)

chairCheckBox.grid(row=1,column=1,padx=5,pady=5)



cameraCheckBox = tk.Checkbutton(window, text='Camera Detection',variable=camera, onvalue=True, offvalue=False)
cameraCheckBox.grid(row=1,column=2,padx=5,pady=5)


cupCheckBox = tk.Checkbutton(window, text='Cup Detection',variable=cup, onvalue=True, offvalue=False)
cupCheckBox.grid(row=1,column=3,padx=5,pady=5)



fileDialogButton = tk.Button(window,text='Select File',command =select_file)
fileDialogButton.grid(row=2,column=1,padx=5,pady=120)


videoButton = tk.Button(window,text='Start Video',command =start_video)
videoButton.grid(row=2,column=2,padx=5,pady=120)






window.mainloop()