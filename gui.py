import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np

#loading the trained model and classify sign
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
model = load_model(r'D:\Python projects\Machine Learning\Skin Cancer Prediction\artifacts\skin_cancer.h5')

classes = {"benign": 0, "malignant": 1}
classes_refined = {v:k for k, v in classes.items()}

#intializing GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Skin Cancer classification')
top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def classify_image(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((224,224))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = model.predict([image])[0][0]
    if pred>0.5:
        sign = classes_refined[1]
    else:
        sign = classes_refined[0]

    label.configure(foreground='#011638', text=sign)

def show_classify_button(file_path):
   classify_b=Button(top,text="Classify Image",command=lambda: classify_image(file_path),padx=10,pady=5)
   classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
   classify_b.place(relx=0.79,rely=0.46)
def upload_image():
    try:
           file_path=filedialog.askopenfilename()
           uploaded=Image.open(file_path)
           uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
           im=ImageTk.PhotoImage(uploaded)
           sign_image.configure(image=im)
           sign_image.image=im
           label.configure(text='')
           show_classify_button(file_path)
    except:
           pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="check your skin image",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()