from PestModel import PestModel
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QStyle
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QMutex, QWaitCondition
from functools import partial
from gui import Ui_MainWindow
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import load_checkpoint
from draw import draw

def handle_exception(exc_type, exc_value, exc_traceback):
    """crash handle func"""
    message = f"An exception of type {exc_type.__name__} occurred.\n{exc_value}"
    QMessageBox.critical(None, "Error", message)

class MyMainForm(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.pb)
        self.pushButton_2.clicked.connect(self.pb2)
        self.pushButton_3.clicked.connect(self.pb3)
        self.pushButton_4.clicked.connect(self.pb4)
        self.pushButton_6.clicked.connect(self.pb6)
        self.pushButton_5.clicked.connect(self.pb5)
        self.pushButton_7.clicked.connect(self.pb7)
        
        self.img_path = None
        self.model = None
        self.img = None
        self.STATUS_PLAYING = 0
        self.STATUS_PAUSE = 1
        self.status = self.STATUS_PAUSE

        with open("classes.txt", "r") as f:
            self.categories = [s.strip() for s in f.readlines()]
        
    def pb(self):
        m, _ = QtWidgets.QFileDialog.getOpenFileName(None,"Image path","./","*.jpg;;*.jpeg")
        if not m:
            pass
        else:
            self.lineEdit.setText(m)
            self.img_path = m
    def pb2(self):
        m, _ = QtWidgets.QFileDialog.getOpenFileName(None,"Weight path","./","*.tar")
        if not m:
            pass
        else:
            self.lineEdit_2.setText(m)
            self.model = PestModel('convnext_base.fb_in22k')
            self.model.eval()
            load_checkpoint(self.model,m)
            
    def pb3(self):
        # UI info
        msg = {'img_path':self.img_path,'model':self.model,'categories':self.categories}
        self.predict_img = Predict_img(msg)
        # refresh UI
        self.predict_img.sinOut.connect(self.display_img)
        
        self.pushButton_3.setEnabled(False)
        self.pushButton_3.setText("Predicting...")
        self.predict_img.start()
        
    def pb4(self):
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(None, 'Save', './', '(*.jpg)')
        if not fileName:
            pass
        else:
            cv2.imwrite(fileName, self.img)
    def pb5(self):
        msg = {'ui':self, 'video_path':self.lineEdit_3.text(),'model':self.model, 'categories':self.categories}
        self.predict_frame = Predict_frame(msg)
        # refresh UI
        self.predict_frame.sinOut.connect(self.display_frame)
        self.predict_frame.sinOut_2.connect(self.over)
        
        self.predict_frame.start()
        self.pushButton_7.setEnabled(True)
    def pb6(self):
        m, _ = QtWidgets.QFileDialog.getOpenFileName(None,"Video path","./","*.mp4;*.mov")
        if not m:
            pass
        else:
            self.lineEdit_3.setText(m)
    def pb7(self):
        if self.status == self.STATUS_PLAYING:
            self.pushButton_7.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.status = self.STATUS_PAUSE
            self.predict_frame.pause()
        else:
            self.pushButton_7.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.status = self.STATUS_PLAYING
            self.predict_frame.resume()
        
    def display_img(self, img):
        self.graphicsView.setImage(img)
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.pushButton_3.setEnabled(True)
        self.pushButton_3.setText("Predict")
    def display_frame(self, frame):
        self.graphicsView_2.setImage(frame)
    def over(self):
        self.pushButton_7.setEnabled(False)
        self.pushButton_7.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.status = self.STATUS_PAUSE
        

class Predict_img(QThread):
    sinOut = pyqtSignal(np.ndarray)
    def __init__(self, msg):
        super(Predict_img, self).__init__()
        self.msg = msg
        self.config = resolve_data_config({}, model=self.msg['model'])
        self.transform = create_transform(**self.config)
    def run(self):
        model = self.msg['model']
        img_path = self.msg['img_path']
        categories = self.msg['categories']
        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0)
        results = model(tensor)
        probabilities = torch.nn.functional.softmax(results[0], dim=0)
        index = torch.argmax(probabilities, dim=0)
        ## add results on image, return img
        text = f'{categories[index]}:{probabilities[index]:.2f}'
        font = 'arial.ttf'
        img = np.array(draw(img,text,font).convert('RGB'))
        self.sinOut.emit(img)

class Predict_frame(QThread):
    sinOut = pyqtSignal(np.ndarray)
    sinOut_2 = pyqtSignal()
    def __init__(self, msg):
        super(Predict_frame, self).__init__()
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.msg = msg
        self.config = resolve_data_config({}, model=self.msg['model'])
        self.transform = create_transform(**self.config)
        self.is_paused = True
    def pause(self):
        self.mutex.lock()
        self.is_paused = True
        self.mutex.unlock()
    def resume(self):
        self.mutex.lock()
        self.is_paused = False
        self.condition.wakeAll()
        self.mutex.unlock()
    def run(self):
        init = True
        model = self.msg['model']
        video_path = self.msg['video_path']
        categories = self.msg['categories']
        #test:http://devimages.apple.com.edgekey.net/streaming/examples/bipbop_4x3/gear2/prog_index.m3u8
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if success:
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                # Run inference on the frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                tensor = self.transform(frame).unsqueeze(0)
                results = model(tensor)
                probabilities = torch.nn.functional.softmax(results[0], dim=0)
                index = torch.argmax(probabilities, dim=0)
                # add results on image, return annotated_frame
                text = f'{categories[index]}:{probabilities[index]:.2f}'
                font = 'arial.ttf'
                annotated_frame = np.array(draw(frame,text,font).convert('RGB'))
                self.sinOut.emit(annotated_frame)
            else:
                # loop over
                self.sinOut_2.emit()
                break
            # pause, recover
            self.mutex.lock()
            if self.is_paused:
                self.condition.wait(self.mutex)
            self.mutex.unlock()

if __name__ == '__main__':
    sys.excepthook = handle_exception
    app = QApplication(sys.argv)
    myWin = MyMainForm()
    myWin.show()
    sys.exit(app.exec_())
