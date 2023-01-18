import cv2
import numpy as np
import pyqtgraph as pg
import webbrowser
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from process import Process
from video_rgb_only import Video_RGB
from video_realsense_file import Video
from webcam import Webcam
from webcam import Camera_RGB
import sys
import timeit
import time
import signal
import threading
from queue import Queue


class Communicate(QObject):
    closeApp = pyqtSignal()


class GUI(QMainWindow, QThread):
    def __init__(self):
        super().__init__()
        self.initUI()  # start the UI when run
        self.input_rgb = Video_RGB()
        self.input_realsense = Video()
        self.input_rgb_camera = Camera_RGB()
        self.input_realsense_camera = Webcam()
        self.input = self.input_rgb  # input of the app
        self.dirname = ""
        self.add_nir_mode = False
        #self.statusBar.showMessage("Input: RGB Only", 5000)
        self.btnOpen.setEnabled(True)
        self.process = Process()
        self.status = False  # If false, not running, if true, running
        self.camera_switch = False
        self.length = 10
        self.running = False
        self.avg_bpms = []
        self.smooth_bpms = []
        self.bpm_count = 0
        self.length = 10
        self.mode = 0

    def initUI(self):
        # set font
        font = QFont()
        font.setFamily('Adobe Gothic Std B')
        font.setBold(True)
        font.setPointSize(14)
        font.setWeight(20)

        # 測試畫面
        self.lblDisplay = QLabel(self)
        self.lblDisplay.setGeometry(20, 70, 640, 480)
        self.lblDisplay.setStyleSheet("background-color: #000000")
        self.lblDisplay.setText("source image")
        self.lblDisplay.setAlignment(QtCore.Qt.AlignCenter)

        # color face image
        self.lblColor = QLabel(self)
        self.lblColor.setGeometry(690, 70, 255, 255)
        self.lblColor.setStyleSheet("background-color: #000000")
        self.lblColor.setText("color face image")
        self.lblColor.setAlignment(QtCore.Qt.AlignCenter)

        # NIR face image
        self.lblNir = QLabel(self)
        self.lblNir.setGeometry(960, 70, 155, 155)
        self.lblNir.setStyleSheet("background-color: #000000")
        self.lblNir.setText("NIR face image")
        self.lblNir.setAlignment(QtCore.Qt.AlignCenter)

        # dynamic plot # Processed Signal 圖表
        self.signal_Plt = pg.PlotWidget(self)
        self.signal_Plt.setGeometry(690, 335, 640, 215)
        self.signal_Plt.setBackground('#ffffff')
        # self.signal_Plt.setOpacity(1)
        self.signal_Plt.setLabel('top', "Processed Signal")

        # Frequency Lable
        self.lblHR = QLabel(self)
        self.lblHR.setGeometry(1130, 90, 300, 50)
        self.lblHR.setStyleSheet("color:white")
        self.lblHR.setFont(QFont("OCR A Std", 13, QFont.Bold))
        self.lblHR.setText(" Current Heart Rate : -- bpm ")

        # Heart Rate Lable
        self.lblHR2 = QLabel(self)
        self.lblHR2.setGeometry(1130, 160, 300, 50)
        self.lblHR2.setStyleSheet("color:white")
        self.lblHR2.setFont(QFont("OCR A Std", 13, QFont.Bold))
        self.lblHR2.setText(" Smooth Heart Rate : -- bpm ")

        # Time Lable
        #now = QDateTime.currentDateTime()
        self.lblTime = QLabel(self)
        self.lblTime.setGeometry(1140, 10, 200, 40)
        self.lblTime.setFont(font)
        self.lblTime.setAlignment(Qt.AlignCenter)
        self.lblTime.setStyleSheet("color:#00FF00")
        self.lblTime.setText("- - . - - . - -")

        # CCU Logo1 button
        self.lblCCU_Logo1 = QLabel(self)
        self.lblCCU_Logo1.setGeometry(70, 10, 270, 50)
        self.lblCCU_Logo1.setStyleSheet(
            "QLabel{border-image: url(./IMG_Source/CCU_Logo.png);}")

        # CCU Logo2 button
        self.lblCCU_Logo2 = QPushButton("< Produced  by  Lab520 >", self)
        self.lblCCU_Logo2.setGeometry(480, 700, 400, 50)
        self.lblCCU_Logo2.setFont(QFont("Hanyi Senty Meadow", 16, QFont.Bold))
        self.lblCCU_Logo2.setStyleSheet("QPushButton{color: white;}")
        self.lblCCU_Logo2.setFlat(True)
        self.lblCCU_Logo2.clicked.connect(self.lblCCU_Logo2_clicked)

        # CCU Logo3 button
        self.lblCCU_Logo3 = QLabel(self)
        self.lblCCU_Logo3.setGeometry(350, 15, 270, 50)
        self.lblCCU_Logo3.setStyleSheet("color:white")
        self.lblCCU_Logo3.setFont(QFont("Adobe 宋体 Std L", 10, QFont.Bold))
        self.lblCCU_Logo3.setText("電機工程 研究所")

        # DataSource Combobox
        self.lblInputTip = QLabel(self)
        self.lblInputTip.setGeometry(30, 580, 100, 40)
        self.lblInputTip.setStyleSheet("color:white")
        self.lblInputTip.setFont(QFont("OCR A Std", 13, QFont.Bold))
        self.lblInputTip.setText(" Input : ")

        self.cbbInput = QComboBox(self)
        self.cbbInput.addItem(" Video ")
        self.cbbInput.addItem(" Camera ")
        self.cbbInput.setCurrentIndex(0)
        self.cbbInput.setGeometry(130, 580, 180, 40)
        self.cbbInput.setFont(font)
        self.cbbInput.setStyleSheet(
            "QComboBox{color:gray;background-image: url(./IMG_Source/Button.jpg); border-radius: 10px; border: 2px groove gray;border-style: outset;}")
        self.cbbInput.activated.connect(self.selectInput)

        # HR Estimator Mode Combobox
        self.lblModeTip = QLabel(self)
        self.lblModeTip.setGeometry(30, 640, 100, 40)
        self.lblModeTip.setStyleSheet("color:white")
        self.lblModeTip.setFont(QFont("OCR A Std", 13, QFont.Bold))
        self.lblModeTip.setText(" Mode : ")

        self.cbbMode = QComboBox(self)
        self.cbbMode.addItem(" RGB ")
        self.cbbMode.addItem(" RGB & NIR ")
        self.cbbMode.addItem(" CIEab & NIR ")
        self.cbbMode.setCurrentIndex(0)
        self.cbbMode.setGeometry(130, 640, 180, 40)
        self.cbbMode.setFont(font)
        self.cbbMode.setStyleSheet(
            "QComboBox{color:gray;background-image: url(./IMG_Source/Button.jpg); border-radius: 10px; border: 2px groove gray;border-style: outset;}")
        self.cbbMode.activated.connect(self.selectMode)

        # HR Estimator length Combobox
        self.lblLengthTip = QLabel(self)
        self.lblLengthTip.setGeometry(340, 580, 130, 40)
        self.lblLengthTip.setStyleSheet("color:white")
        self.lblLengthTip.setFont(QFont("OCR A Std", 13, QFont.Bold))
        self.lblLengthTip.setText(" Estimator length : ")

        self.cbbLength = QComboBox(self)
        self.cbbLength.addItem(" 10 s ")
        self.cbbLength.addItem(" 20 s ")
        self.cbbLength.addItem(" 30 s ")
        self.cbbLength.addItem(" 60 s ")
        self.cbbLength.setCurrentIndex(0)
        self.cbbLength.setGeometry(480, 580, 180, 40)
        self.cbbLength.setFont(font)
        self.cbbLength.setStyleSheet(
            "QComboBox{color:gray;background-image: url(./IMG_Source/Button.jpg); border-radius: 10px; border: 2px groove gray;border-style: outset;}")
        self.cbbLength.activated.connect(self.selectLength)

        # HR Estimator Output Combobox
        self.lblOutputTip = QLabel(self)
        self.lblOutputTip.setGeometry(340, 640, 130, 40)
        self.lblOutputTip.setStyleSheet("color:white")
        self.lblOutputTip.setFont(QFont("OCR A Std", 13, QFont.Bold))
        self.lblOutputTip.setText(" Output : ")

        self.cbbOutput = QComboBox(self)
        self.cbbOutput.addItem(" Single ")
        self.cbbOutput.addItem(" Sequence")
        self.cbbOutput.setCurrentIndex(0)
        self.cbbOutput.setGeometry(480, 640, 180, 40)
        self.cbbOutput.setFont(font)
        self.cbbOutput.setStyleSheet(
            "QComboBox{color:gray;background-image: url(./IMG_Source/Button.jpg); border-radius: 10px; border: 2px groove gray;border-style: outset;}")
        self.cbbOutput.activated.connect(self.selectOutput)

        # Start button
        self.btnStart = QPushButton("Start \n HR Detection", self)
        self.btnStart.setGeometry(1030, 580, 305, 100)
        self.btnStart.setFont(font)
        self.btnStart.setStyleSheet("QPushButton{color: gray ;border-image: url(./IMG_Source/Button.jpg); border-radius: 10px; border: 2px groove gray;border-style: outset;}"
                                    "QPushButton:hover{color: #ffffff;}"
                                    "QPushButton:pressed{color: #000000;}")
        self.btnStart.clicked.connect(self.switch_start_stop)

        # Open button
        self.btnOpen = QPushButton("Open \n Video File", self)
        self.btnOpen.setGeometry(690, 580, 305, 100)
        self.btnOpen.setFont(font)
        self.btnOpen.setStyleSheet("QPushButton{color: gray ;border-image: url(./IMG_Source/Button.jpg); border-radius: 10px; border: 2px groove gray;border-style: outset;}"
                                   "QPushButton:hover{color: #ffffff;}"
                                   "QPushButton:pressed{color: #000000;}")
        self.btnOpen.clicked.connect(self.selectOpenInput)

        # Pro button

        self.btnPro1 = QPushButton(self)
        self.btnPro1.setGeometry(1370, 70, 50, 200)
        self.btnPro1.setStyleSheet("QPushButton{border-image: url(./IMG_Source/Professional_Button.jpg); border-radius: 10px; border: 2px groove gray;border-style: outset;}"
                                   "QPushButton:hover{border-image: url(./IMG_Source/Button.jpg);}"
                                   "QPushButton:pressed{border-image: url(./IMG_Source/Button.jpg);}")
        self.btnPro1.clicked.connect(self.btnPro1_clicked)

        # Pro2 button
        self.btnPro2 = QPushButton("", self)
        self.btnPro2.setGeometry(1370, 280, 50, 200)
        self.btnPro2.setFont(font)

        # Pro3 button
        self.btnPro3 = QPushButton("", self)
        self.btnPro3.setGeometry(1370, 490, 50, 200)
        self.btnPro3.setFont(font)

        # Information button
        self.btnInformation = QPushButton(self)
        self.btnInformation.setGeometry(20, 10, 40, 40)
        self.btnInformation.setStyleSheet("QPushButton{border-image: url(./IMG_Source/Information_Button.png)}"
                                          "QPushButton:hover{background-color: #FFA823;}"
                                          "QPushButton:pressed{background-color: #FFA823;}")
        self.btnInformation.clicked.connect(self.btnInformation_clicked)

        # event close
        self.c = Communicate()
        self.c.closeApp.connect(self.closeEvent)

        # config main window # 視窗大小
        self.setWindowTitle("Heart Rate Monitor")
        self.setGeometry(0, 0, 1415, 760)

        self.fft_Plt = pg.PlotWidget(self)
        self.fft_Plt.setGeometry(20, 770, 257, 139)
        self.fft_Plt.setBackground('#ffffff')
        self.fft_Plt.setLabel('top', "Chosen PP")

        self.trend_Plt = pg.PlotWidget(self)
        self.trend_Plt.setGeometry(283, 770, 257, 139)
        self.trend_Plt.setBackground('#ffffff')
        self.trend_Plt.setLabel('top', "Raw Signal")

        self.test1_Plt = pg.PlotWidget(self)
        self.test1_Plt.setGeometry(546, 770, 257, 139)
        self.test1_Plt.setBackground('#ffffff')
        self.test1_Plt.setLabel('top', "Source 1 PP")

        self.test2_Plt = pg.PlotWidget(self)
        self.test2_Plt.setGeometry(809, 770, 257, 139)
        self.test2_Plt.setBackground('#ffffff')
        self.test2_Plt.setLabel('top', "Source 2 PP")

        self.test3_Plt = pg.PlotWidget(self)
        self.test3_Plt.setGeometry(1072, 770, 257, 139)
        self.test3_Plt.setBackground('#ffffff')
        self.test3_Plt.setLabel('top', "Source 3 PP")

        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(
            QPixmap("./IMG_Source/BackGround.jpg")))
        self.setPalette(palette)

        self.statusBar = QStatusBar()
        self.statusBar.setStyleSheet("color:white")
        self.statusBar.setFont(QFont("OCR A Std", 13, QFont.Bold))
        self.setStatusBar(self.statusBar)

        # event close
        self.c = Communicate()
        self.c.closeApp.connect(self.closeEvent)

        self.center()
        self.show()

    def btnPro1_clicked(self):
        self.setGeometry(0, 0, 1415, 920)
        self.center()
        self.show()

    def btnPro2_clicked(self):
        reply = QMessageBox.question(
            self, "Message", "Are you sure want to quit ?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

    def btnPro3_clicked(self):
        reply = QMessageBox.question(
            self, "Message", "Are you sure want to quit ?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

    def btnInformation_clicked(self):
        webbrowser.open('http://www.dsp.ee.ccu.edu.tw/wnlie/')

    def lblCCU_Logo2_clicked(self):
        webbrowser.open('http://www.dsp.ee.ccu.edu.tw/wnlie/')

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Message", "Are you sure want to quit",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            event.accept()
            self.input.stop()
            cv2.destroyAllWindows()
        else:
            event.ignore()

    def selectOutput(self):
        self.reset()
        if self.cbbOutput.currentIndex() == 0:
            self.btnOpen.setEnabled(True)
            self.statusBar.showMessage("Estimate single bpm", 5000)
        elif self.cbbOutput.currentIndex() == 1:
            self.btnOpen.setEnabled(True)
            self.statusBar.showMessage("Estimate sequence bpm", 5000)

    def selectInput(self):
        self.reset()
        if self.cbbInput.currentIndex() == 0:
            self.btnOpen.setEnabled(True)
            self.btnOpen.setText("Open \n Video File")
            self.statusBar.showMessage("Input: Video", 5000)
        elif self.cbbInput.currentIndex() == 1:
            self.btnOpen.setEnabled(True)
            self.btnOpen.setText("Open Camera ")
            self.statusBar.showMessage("Input: Camera", 5000)

    def selectMode(self):
        self.reset()
        if self.cbbMode.currentIndex() == 0:
            self.btnOpen.setEnabled(True)
            self.add_nir_mode = False
            self.mode = 0
            self.statusBar.showMessage("Mode: RGB", 5000)
        elif self.cbbMode.currentIndex() == 1:
            self.btnOpen.setEnabled(True)
            self.add_nir_mode = True
            self.mode = 1
            self.statusBar.showMessage("Mode: RGB & NIR", 5000)
        elif self.cbbMode.currentIndex() == 1:
            self.btnOpen.setEnabled(True)
            self.add_nir_mode = True
            self.mode = 2
            self.statusBar.showMessage("Mode: CIEab & NIR", 5000)

    # set the length of time for estimate heart rate
    def selectLength(self):
        self.reset()
        if self.cbbLength.currentIndex() == 0:
            self.length = 10
            self.statusBar.showMessage("use 10s estimate", 5000)
        elif self.cbbLength.currentIndex() == 1:
            self.length = 20
            self.statusBar.showMessage("use 20s estimate", 5000)
        elif self.cbbLength.currentIndex() == 2:
            self.length = 30
            self.statusBar.showMessage("use 30s estimate", 5000)
        elif self.cbbLength.currentIndex() == 3:
            self.length = 60
            self.statusBar.showMessage("use 60s estimate", 5000)

    def key_handler(self):
        """
        cv2 window must be focused for keypresses to be detected.
        """
        self.pressed = cv2.waitKey(1) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("[INFO] Exiting")
            self.input.stop()

    def openFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.dirname, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                      "All Files (*);;Python Files (*.py)", options=options)
        self.statusBar.showMessage(" Folder name: " + self.dirname, 5000)

    def selectOpenInput(self):
        if self.cbbInput.currentIndex() == 1:
            if self.cbbMode.currentIndex() == 1:
                self.input = self.input_realsense_camera
            else:
                self.input = self.input_rgb_camera
            print("open camera")
            self.open_camera()
        else:
            if self.cbbMode.currentIndex() == 1:
                self.input = self.input_realsense
            else:
                self.input = self.input_rgb
            print("open file")
            self.openFileDialog()

    def reset(self):
        self.process.reset()
        self.lblDisplay.clear()
        self.lblColor.clear()
        self.lblNir.clear()

    @QtCore.pyqtSlot()
    def main_loop(self):

        # color_frame = None
        # nir_frame = None

        if self.add_nir_mode:
            # self.input.get_frame(color_frame, nir_frame)
            color_frame, nir_frame = self.input.get_frame()  # Read frames from input
        else:
            color_frame = self.input.get_frame()
        gui_img = QImage(color_frame, color_frame.shape[1], color_frame.shape[0], color_frame.strides[0],
                         QImage.Format_RGB888)
        # color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        # cv2.imshow('framergb', color_frame)
        # cv2.imshow('framenir', nir_frame)
        self.lblDisplay.setPixmap(QPixmap(gui_img))  # show frame on GUI

        if self.add_nir_mode:
            bpm, color_face, nir_face = self.process.run(color_frame, nir_frame)
        else:
            bpm, color_face = self.process.run(color_frame)  # Run the main algorithm

        color_face_input = color_face.copy()
        # color_face_input = cv2.resize(color_face_input, (255, 255), interpolation=cv2.INTER_CUBIC)
        color_face_img = QImage(color_face_input, color_face_input.shape[1], color_face_input.shape[0],
                                color_face_input.strides[0], QImage.Format_RGB888)
        self.lblColor.setPixmap(QPixmap(color_face_img))  # Show color face

        if self.add_nir_mode:
            nir_face_input = nir_face.copy()
            # nir_face_input = cv2.resize(nir_face_input, (255, 255), interpolation=cv2.INTER_CUBIC)
            nir_face_img = QImage(nir_face_input, nir_face_input.shape[1], nir_face_input.shape[0],
                                  nir_face_input.strides[0], QImage.Format_Grayscale8)
            self.lblNir.setPixmap(QPixmap(nir_face_img))  # Show nir face

        self.lblHR.setText("Current heart rate: " + str(float("{:.2f}".format(bpm))) + "bpm")
        if len(self.process.bpms) > 25:
            for i in range(5, 0, -1):
                self.smooth_bpms.append(np.mean(self.process.bpms[-5 * i:-5 * (i - 1)]))

        if self.process.bpms.__len__() > 1:
            self.lblHR2.setText(
                "Smoothed heart rate: " + str(
                    float("{:.2f}".format(np.mean(self.process.bpms)))) + " bpm")  # Print bpm value
            self.avg_bpms.append(np.mean(self.process.bpms))

        if self.cbbOutput.currentIndex() == 0 and self.process.count >= self.process.buffer_size + 2:  # Second condition to stop running, this is 10 seconds
            print('Average FPS is: ' + str(self.process.count / (time.time() - self.t0)))
            # print('cost time is: ' + str((time.time() - self.t0)))
            # print("Testing finished")
            print("Result: " + str(np.mean(self.process.bpms)))
            self.status = False
            self.input.stop()
            self.btnStart.setText("Start")
            self.btnOpen.setEnabled(True)

        self.key_handler()  # if not the GUI cant show anything, to make the gui refresh after the end of loop

    def open_camera(self):
        # color_frame = None
        # nir_frame = None
        self.input.start()
        if not self.camera_switch:
            self.camera_switch = True
            while self.camera_switch:
                if self.add_nir_mode:
                    # self.input.get_frame(color_frame, nir_frame)
                    color_frame, nir_frame = self.input.get_frame()  # Read frames from input
                else:
                    color_frame = self.input.get_frame()
                gui_img = QImage(color_frame, color_frame.shape[1], color_frame.shape[0], color_frame.strides[0],
                                 QImage.Format_RGB888)
                self.lblDisplay.setPixmap(QPixmap(gui_img))  # show frame on GUI
                self.key_handler()  # if not the GUI cant show anything, to make the gui refresh after the end of loop
        else:
            self.camera_switch = False

    def estimate_sequence_bpm(self):
        if not self.camera_switch:  # if camera not open
            self.input.start()

        self.t0 = time.time()
        # print('start time is: ' + str(self.t0))
        while self.running:
            self.main_loop()
            self.signal_Plt.clear()
            self.signal_Plt.plot(self.process.RGB_signal_buffer[1], pen='r')  # Plot green signal

            self.fft_Plt.clear()
            self.fft_Plt.plot(self.process.FREQUENCY[:300], self.process.PSD[:300], pen='r')  # Plot fused PSD

            self.trend_Plt.clear()
            self.trend_Plt.plot(self.process.test4, pen='r')  # Plot NIR's PSD

            self.test1_Plt.clear()
            self.test1_Plt.plot(self.process.bpms[-50:], pen='r')  # Plot each component's PSD

            self.test2_Plt.clear()
            self.test2_Plt.plot(self.avg_bpms[-50:], pen='r')  # Plot each component's PSD

            self.test3_Plt.clear()
            #self.test3_Plt.plot(self.smooth_bpms[:], pen='r')

    def estimate_single_bpm(self):
        if not self.status:
            self.reset()
            self.status = True
            self.btnStart.setText("Stop")
            self.lblHR2.clear()
            if not self.camera_switch:
                self.input.start()

            self.t0 = time.time()
            # print('start time is: ' + str(self.t0))
            while self.status:
                self.main_loop()
                self.signal_Plt.clear()
                self.signal_Plt.plot(self.process.RGB_signal_buffer[1], pen='r')  # Plot green signal

                self.fft_Plt.clear()
                self.fft_Plt.plot(self.process.FREQUENCY[:300], self.process.PSD[:300], pen='r')  # Plot fused PSD

                self.trend_Plt.clear()
                self.trend_Plt.plot(self.process.test4, pen='r')  # Plot NIR's PSD

                self.test1_Plt.clear()
                self.test1_Plt.plot(self.process.FREQUENCY[:300], self.process.test1[:300],
                                    pen='r')  # Plot each component's PSD

                self.test2_Plt.clear()
                self.test2_Plt.plot(self.process.FREQUENCY[:300], self.process.test2[:300],
                                    pen='r')  # Plot each component's PSD

                self.test3_Plt.clear()
                self.test3_Plt.plot(self.process.FREQUENCY[:300], self.process.test3[:300], pen='r')
                self.running = False

        elif self.status:
            self.status = False
            self.input.stop()
            self.btnStart.setText("Start")

    def switch_start_stop(self):
        if self.running:  # when click stop
            self.input.stop()
            self.running = False
            self.btnStart.setText("Start")
        else:  # when click start
            self.reset()
            self.btnStart.setText("Stop")
            self.lblHR2.clear()
            self.running = True
            self.run()

    def run(self):

        while self.running:
            print("running")
            self.input.dirname = self.dirname
            self.process.set_mode(self.mode)
            self.process.set_length(self.length)
            if self.cbbInput.currentIndex() == 0 and self.input.dirname == "":
                print("Choose a video first")
                self.statusBar.showMessage("Choose a video first", 5000)
                return

            if self.cbbOutput.currentIndex() == 0:
                self.estimate_single_bpm()
            else:
                self.estimate_sequence_bpm()


def signal_handler(sig, frame):
    print("ctrl c")
    app.quit()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    app = QApplication(sys.argv)
    ex = GUI()
    sys.exit(app.exec_())
