import os
import sys
import csv
import cv2
import time
import pylsl
import socket
import zipfile
import threading
import traceback
import numpy as np
import pyqtgraph as pg
import tobii_research as tr
from io import StringIO
from datetime import datetime
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QPushButton, QDialog, QVBoxLayout, QLabel, QDialogButtonBox
from sensor_streamers.SensorStreamer import SensorStreamer
from visualizers.LinePlotVisualizer import LinePlotVisualizer
from collections import OrderedDict
from utils.print_utils import *
from scipy.signal import lfilter, butter

class SessionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Session Information")
        self.layout = QVBoxLayout(self)

        # Participant name input
        self.layout.addWidget(QLabel("Participant Name:"))
        self.participantNameLineEdit = QtWidgets.QLineEdit(self)
        self.layout.addWidget(self.participantNameLineEdit)

        # Dialog buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)
        
        # Size of window
        self.setGeometry(125, 150, 250, 100)

    def participantName(self):
        return self.participantNameLineEdit.text()

class Worker(QtCore.QThread):
    sensor_data_collect = QtCore.pyqtSignal(np.ndarray, list)
    feature_data_collect = QtCore.pyqtSignal(dict, list)
    
    # 여기서만 tobii_handler 부르기
    def __init__(self):
        super().__init__()
        
        self.channel = 18
        self.point = 100
        self.data = np.zeros((self.point, self.channel))
        
        self.tobii_handler = Tobii_handler()

    def run(self):
        #try:
        while True:
            t = time.time()
            while time.time() - t <= 1:
                self.update_data()
                time.sleep(0.016)
            self.feature_ex()
            
        #except Exception as e:
        #    print(f'Error in worker: {e}')
            
    def update_data(self):
        new_data = self.tobii_handler.gaze_data

        # Shift the data matrix up by one row
        self.data = np.roll(self.data, -1, axis=0)

        # Update the last row with new gaze data
        self.data[-1, :] = new_data
        
        self.sensor_data_collect.emit(self.data, new_data)
        
    def feature_ex(self):
        feature = self.tobii_handler.extract_features()
        sensor = self.tobii_handler.gaze_data
        self.feature_data_collect.emit(feature, sensor)
        
class Worker_E4(QtCore.QThread):
    E4_sensor_data_collect = QtCore.pyqtSignal(list)
    ACC_data_collect = QtCore.pyqtSignal(list)
    BVP_IBI_data_collect = QtCore.pyqtSignal(list)
    GSR_TMP_data_collect = QtCore.pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        
        self.running = True
        
        try:
            self.E4_streamer = E4Streamer(print_status=True, print_debug=False)
            self.E4_streamer.connect()
            self.E4_streamer.run()
            self.start_time = time.time()
            print("Streamer Running Start")
        except:
            print('error in worker_E4')
        
    def stopp(self):
        try:
            self.E4_streamer.stop()
        except:
            pass

    def run(self):
        try:
            while True:
                t = time.time()
                while time.time() - t <= 1:
                    self.update_E4_data()
                    tt = time.time()
                    while time.time() - tt <= 0.25:
                        ttt = time.time()
                        self.update_ACC()
                        while time.time() - ttt <= 0.03125:
                            self.update_BVP_IBI()
                            time.sleep(0.015625)
                    self.update_GSR_TMP()
            
        except Exception as e:
            print(f'Error in worker_E4 : {e}')
            
    def update_E4_data(self):
        msg = []
        try:
            for device_name in self.E4_streamer.get_device_names():
                stream_names = self.E4_streamer.get_stream_names(device_name=device_name)
                for stream_name in stream_names:
                    num_timesteps = self.E4_streamer.get_num_timesteps(device_name, stream_name)
                    msg.append((num_timesteps) / (time.time() - self.start_time))
        except:
            pass
        
        self.E4_sensor_data_collect.emit(msg)
        
    def update_ACC(self):
        msg = []
        try:
            for device_name in self.E4_streamer.get_device_names():
                stream_names = self.E4_streamer.get_stream_names(device_name=device_name)
                for stream_name in stream_names:
                    num_timesteps = self.E4_streamer.get_num_timesteps(device_name, stream_name)
                    msg.append((num_timesteps) / (time.time() - self.start_time))
        except:
            pass
        
        self.ACC_data_collect.emit(msg)
        
    def update_BVP_IBI(self):
        msg = []
        try:
            for device_name in self.E4_streamer.get_device_names():
                stream_names = self.E4_streamer.get_stream_names(device_name=device_name)
                for stream_name in stream_names:
                    num_timesteps = self.E4_streamer.get_num_timesteps(device_name, stream_name)
                    msg.append((num_timesteps) / (time.time() - self.start_time))
        except:
            pass
        
        self.BVP_IBI_data_collect.emit(msg)
        
    def update_GSR_TMP(self):
        msg = []
        try:
            for device_name in self.E4_streamer.get_device_names():
                stream_names = self.E4_streamer.get_stream_names(device_name=device_name)
                for stream_name in stream_names:
                    num_timesteps = self.E4_streamer.get_num_timesteps(device_name, stream_name)
                    msg.append((num_timesteps) / (time.time() - self.start_time))
        except:
            pass
        
        self.GSR_TMP_data_collect.emit(msg)
        
class Worker_time(QtCore.QThread):
    time_signal = QtCore.pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        
    def run(self):
        while True:
            self.time_signal.emit(time.time())
            time.sleep(0.001)
        
class GSRAnalyzer_first:
    def __init__(self, fs, cutoff=0.05):
        """
        Initialize the GSR analyzer.

        Parameters:
        - fs (float): Sampling frequency (Hz)
        - cutoff (float): Cutoff frequency for the low-pass filter (Hz)
        """
        self.fs = fs
        self.cutoff = cutoff
        self.prev_tonic = None
        self.compute_filter_coefficients()

    def compute_filter_coefficients(self):
        """
        Compute the filter coefficients for the low-pass filter.
        """
        tau = 1 / (2 * np.pi * self.cutoff)
        self.b = [1 / (1 + 2 * np.pi * self.cutoff * (1 / self.fs))] # 분자 coefficient
        self.a = [1, (1 - self.b[0])] # 분모 coefficient

    def process_frame(self, gsr_frame):
        """
        Process a single frame of GSR signal and return the tonic and phasic components.

        Parameters:
        - gsr_frame (float): A single GSR value or a small frame of GSR signal

        Returns:
        - gsr_tonic (float): Tonic component of the GSR frame
        - gsr_phasic (float): Phasic component of the GSR frame
        """
        gsr_frame = np.array(gsr_frame)

        if self.prev_tonic is None:
            self.prev_tonic = gsr_frame if gsr_frame.ndim == 0 else gsr_frame[0]
            
        if gsr_frame.ndim == 0:
            gsr_tonic = self.prev_tonic * self.b[0] + gsr_frame * (1 - self.b[0])
            gsr_phasic = gsr_frame - gsr_tonic
            self.prev_tonic = gsr_tonic
        else:
            gsr_tonic = lfilter(self.b, self.a, gsr_frame, zi=[self.prev_tonic])[0]
            gsr_phasic = gsr_frame - gsr_tonic
            self.prev_tonic = gsr_tonic[-1]

        return gsr_tonic, gsr_phasic

##########################################################################
class Application(QtWidgets.QMainWindow):
    def __init__(self, participant_name, host='127.0.0.1', port=5005):
        super(Application, self).__init__()

        self.participant_name = participant_name
        
        self.host = host
        self.port = port
        
        self.eyedata = []

        # Initialize Tobii handling instance
        self.worker = Worker()
        self.worker.start()
        
        self.worker_E4 = Worker_E4()
        self.worker_E4.start()
        
        self.worker_time = Worker_time()
        self.worker_time.start()
        self.current_timer()

        # Initialize TCP client object
        self.tcp_client = TCP_Client(host, port)

        # Initialize thread stop event flag
        self.stop_event = threading.Event()

        # UI Initialization
        self.initUI()

        # Initialize plots
        self.init_plots()
        self.init_feature_plots()
        self.init_E4_feature_plots()
        
        self.gamestarted = ''
    
    def initUI(self):
        self.setWindowTitle("PyQt Graphs")

        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)

        self.layout = QtWidgets.QGridLayout(self.centralWidget)

        # Start button
        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start)
        self.layout.addWidget(self.start_button, 0, 0)
        
        # Graph widget
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graph_widget, 0, 0, 4, 1)
        self.graph_widget.hide()

        # Feature button
        self.start_feature_button = QPushButton("Start eye feature", self)
        self.start_feature_button.clicked.connect(self.start_feature)
        self.layout.addWidget(self.start_feature_button, 0, 1)

        # Feature plots
        self.pupilPlotWidget = pg.PlotWidget(title="Pupil Dilation Rate")
        self.layout.addWidget(self.pupilPlotWidget, 0, 1)
        self.pupilPlotWidget.hide()

        self.blinkPlotWidget = pg.PlotWidget(title="Blink Rate")
        self.layout.addWidget(self.blinkPlotWidget, 0, 2)
        self.blinkPlotWidget.hide()

        self.fixationPlotWidget = pg.PlotWidget(title="Fixation Duration")
        self.layout.addWidget(self.fixationPlotWidget, 1, 1)
        self.fixationPlotWidget.hide()

        self.saccadePlotWidget = pg.PlotWidget(title="Saccade Amplitude")
        self.layout.addWidget(self.saccadePlotWidget, 1, 2)
        self.saccadePlotWidget.hide()

        self.gazePathPlotWidget = pg.PlotWidget(title="Gaze Path Length")
        self.layout.addWidget(self.gazePathPlotWidget, 2, 1)
        self.gazePathPlotWidget.hide()

        self.gazeEntropyPlotWidget = pg.PlotWidget(title="Gaze Transition Entropy")
        self.layout.addWidget(self.gazeEntropyPlotWidget, 2, 2)
        self.gazeEntropyPlotWidget.hide()

        self.saccadeVelocityPlotWidget = pg.PlotWidget(title="Saccade Velocity")
        self.layout.addWidget(self.saccadeVelocityPlotWidget, 3, 1)
        self.saccadeVelocityPlotWidget.hide()

        self.saccadeAccelerationPlotWidget = pg.PlotWidget(title="Saccade Acceleration")
        self.layout.addWidget(self.saccadeAccelerationPlotWidget, 3, 2)
        self.saccadeAccelerationPlotWidget.hide()

        self.peakSaccadicVelocityPlotWidget = pg.PlotWidget(title="Peak Saccadic Velocity")
        self.layout.addWidget(self.peakSaccadicVelocityPlotWidget, 4, 1)
        self.peakSaccadicVelocityPlotWidget.hide()

        self.gazeDispersionPlotWidget = pg.PlotWidget(title="Gaze Dispersion")
        self.layout.addWidget(self.gazeDispersionPlotWidget, 4, 2)
        self.gazeDispersionPlotWidget.hide()
        
        self.gsrTonicPlotWidget = pg.PlotWidget(title="GSR tonic")
        self.layout.addWidget(self.gsrTonicPlotWidget, 5, 1)
        self.gsrTonicPlotWidget.hide()
        
        self.gsrPhasicPlotWidget = pg.PlotWidget(title="GSR phasic")
        self.layout.addWidget(self.gsrPhasicPlotWidget, 5, 2)
        self.gsrPhasicPlotWidget.hide()
        
        # Start E4 button
        self.start_E4_button = QPushButton("Start E4", self)
        self.start_E4_button.clicked.connect(self.start_E4)
        self.layout.addWidget(self.start_E4_button, 5, 0)
        
        self.graph_widget_E4 = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graph_widget_E4, 4, 0, 2, 1)
        self.graph_widget_E4.hide()

        self.start_E4_feature_button = QPushButton("Start E4 feature", self)
        self.start_E4_feature_button.clicked.connect(self.start_E4_feature)
        self.layout.addWidget(self.start_E4_feature_button, 5, 1)

        # TCP connect/disconnect buttons
        self.tcp_connect_button = QPushButton("TCP Connect to Unity 3D", self)
        self.tcp_connect_button.clicked.connect(self.tcp_connect)
        self.layout.addWidget(self.tcp_connect_button, 6, 0)

        self.tcp_disconnect_button = QPushButton("TCP Disconnect from Unity 3D", self)
        self.tcp_disconnect_button.clicked.connect(self.tcp_disconnect)
        self.layout.addWidget(self.tcp_disconnect_button, 6, 1)
        
        self.current = QLabel("Can't see anyway", self)
        self.current.setStyleSheet('background-color: #f5f5f5; color: blue;')
        self.current.setAlignment(QtCore.Qt.AlignCenter) 
        self.layout.addWidget(self.current, 6, 2)
        
        self.gameStartButton = QPushButton("Game start", self)
        self.gameStartButton.clicked.connect(self.gameStart)
        self.layout.addWidget(self.gameStartButton, 7, 0)
        
        self.saveDataButton = QPushButton("Save Data to CSV", self)
        self.saveDataButton.clicked.connect(self.saveData)
        self.layout.addWidget(self.saveDataButton, 7, 1)
        
        # Exit button
        self.exit_button = QPushButton("Exit", self)
        self.exit_button.clicked.connect(self.close)
        self.layout.addWidget(self.exit_button, 7, 2)

        self.setGeometry(100, 100, 1600, 900)
        
    def current_timer(self):
        self.worker_time.time_signal.connect(self.update_time)
        
    def update_time(self, now):
        realtime = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
        self.current.setText((f'{realtime}, UNIX Time : {now:.6f}'))
            
    def feature_timer(self):
        self.worker.feature_data_collect.connect(self.update_feature)

    def start(self):
        
        self.start_button.hide()
        self.graph_widget.show()
        
        self.recorded_data = []

        self.worker.sensor_data_collect.connect(self.update)

        # Start the TCP thread
        self.tcp_thread = threading.Thread(target=self.tcp_thread_works)
        self.tcp_thread.start()

    def init_plots(self):
        self.num_channels = 18  # Total number of channels
        self.num_rows = 6  # Number of rows
        self.num_cols = 3  # Number of columns
        self.num_points = 100
        self.e4_channels = 5
        self.data = np.zeros((self.num_points, self.num_channels)) # Make arrays
        self.data_e4 = np.zeros((self.num_points, self.e4_channels))
        
        self.plots = []
        self.curves = []
        self.plots_e4 = []
        self.curves_e4 = []
        
        self.gazedatalist = ['left gaze point(Dx)', 'left gaze point(Dy)', 'left gaze point(Ux)', 'left gaze point(Uy)', 'left gaze point(Uz)', 'left gaze origin validity',
                        'left gaze point validity', 'left pupil validity', 'left pupil diameter', 'right gaze point(Dx)', 'right gaze point(Dy)', 'right gaze point(Ux)',
                        'right gaze point(Uy)', 'right gaze point(Uz)', 'right gaze origin validity', 'right gaze point validity', 'right pupil validity', 'right pupil diameter']
        e4sensorlist = ['ACC', 'BVP', 'GSR', 'TMP', 'IBI']

        for i in range(self.num_channels):
            # Determine the row and column for the current plot
            row = i // self.num_cols
            col = i % self.num_cols

            plot = self.graph_widget.addPlot(row=row, col=col, title=f'{self.gazedatalist[i]}')
            pen = pg.mkPen(color=(i, self.num_channels), width=2)
            curve = plot.plot(self.data[:, i], pen=pen)
            plot.showGrid(x=True, y=True)
            self.plots.append(plot)
            self.curves.append(curve)
        
        for i in range(self.e4_channels):
            plot_e4 = self.graph_widget_E4.addPlot(row = 0, col = i, title = f'E4 {e4sensorlist[i]}')
            pen = pg.mkPen(color=(i, self.e4_channels), width=2)
            curve_e4 = plot_e4.plot(self.data_e4[:, i], pen=pen)
            plot_e4.showGrid(x=True, y=True)
            self.plots_e4.append(plot_e4)
            self.curves_e4.append(curve_e4)

    def update(self, data, sensor):
        recorded_time = time.time()
        
        # Update the plot curves with new data
        for i, curve in enumerate(self.curves):
           curve.setData(data[:, i])
           
        combined_data = [recorded_time] + sensor
        
        self.recorded_data.append(combined_data)

        # Refresh the graph widget
        self.graph_widget.update()
        
    def start_feature(self):

        self.start_feature_button.hide()
        self.pupilPlotWidget.show()
        self.blinkPlotWidget.show()
        self.fixationPlotWidget.show()
        self.saccadePlotWidget.show()
        self.gazePathPlotWidget.show()
        self.gazeEntropyPlotWidget.show()
        self.saccadeVelocityPlotWidget.show()
        self.saccadeAccelerationPlotWidget.show()
        self.peakSaccadicVelocityPlotWidget.show()
        self.gazeDispersionPlotWidget.show()
        
        self.feature_timer()
        
    def init_feature_plots(self):
        
        # Initialize data lists
        self.pupilData = []
        self.blinkData = []
        self.fixationData = []
        self.saccadeData = []
        self.gazePathData = []
        self.gazeEntropyData = []
        self.saccadeVelocityData = []
        self.saccadeAccelerationData = []
        self.peakSaccadicVelocityData = []
        self.gazeDispersionData = []
        
        self.recorded_data = []
        self.feature_data = []
        self.peak_saccadic_data = []
        self.saccade_amp_data = []
        self.peak_saccadic = 0
        self.last_saccade = 0
        
        # Set up plots with the defined pens
        self.pupilCurve = self.pupilPlotWidget.plot(pen='y')
        self.blinkCurve = self.blinkPlotWidget.plot(pen='r')
        self.fixationCurve = self.fixationPlotWidget.plot(pen='g')
        self.saccadeCurve = self.saccadePlotWidget.plot(pen='b')
        self.gazePathCurve = self.gazePathPlotWidget.plot(pen='m')
        self.gazeEntropyCurve = self.gazeEntropyPlotWidget.plot(pen='y')
        self.saccadeVelocityCurve = self.saccadeVelocityPlotWidget.plot(pen='r')
        self.saccadeAccelerationCurve = self.saccadeAccelerationPlotWidget.plot(pen='g')
        self.peakSaccadicVelocityCurve = self.peakSaccadicVelocityPlotWidget.plot(pen='b')
        self.gazeDispersionCurve = self.gazeDispersionPlotWidget.plot(pen='m')
        
        # Enable grid lines for each plot widget
        plot_widgets = [
            self.pupilPlotWidget, self.blinkPlotWidget, self.fixationPlotWidget,
            self.saccadePlotWidget, self.gazePathPlotWidget, self.gazeEntropyPlotWidget,
            self.saccadeVelocityPlotWidget, self.saccadeAccelerationPlotWidget,
            self.peakSaccadicVelocityPlotWidget, self.gazeDispersionPlotWidget
        ]
        for widget in plot_widgets:
            widget.showGrid(x=True, y=True, alpha=0.3)
        
    def update_feature(self, feature, sensor = None):
        if type(feature) == dict:
            features = feature

            pupil_dilation_rate = features.get('pupil_dilation_rate')
            blink_rate = features.get('blink_rate')
            gaze_path_length = features.get('gaze_path_length')
            gaze_transition_entropy = features.get('gaze_transition_entropy')
            peak_saccadic_velocity = features.get('peak_saccadic_velocity')
            gaze_dispersion = features.get('gaze_dispersion')
            saccade_amp = features.get('saccade_amplitude')
            saccade_vel = features.get('saccade_velocity')
            saccade_accel = features.get('saccade_acceleration')
            fixation_dur = features.get('fixation_duration')

            # Append data to their respective lists
            self.pupilData.append(pupil_dilation_rate)
            self.blinkData.append(blink_rate)
            self.gazePathData.append(gaze_path_length)
            self.gazeEntropyData.append(gaze_transition_entropy)
            self.peakSaccadicVelocityData.append(peak_saccadic_velocity)
            self.gazeDispersionData.append(gaze_dispersion)
            self.saccadeData.append(saccade_amp)
            self.saccadeVelocityData.append(saccade_vel)
            self.saccadeAccelerationData.append(saccade_accel)
            self.fixationData.append(fixation_dur)

            # Update curve data
            self.pupilCurve.setData(self.pupilData)
            self.blinkCurve.setData(self.blinkData)
            self.gazePathCurve.setData(self.gazePathData)
            self.gazeEntropyCurve.setData(self.gazeEntropyData)
            self.peakSaccadicVelocityCurve.setData(self.peakSaccadicVelocityData)
            self.gazeDispersionCurve.setData(self.gazeDispersionData)
            self.saccadeCurve.setData(self.saccadeData)
            self.saccadeVelocityCurve.setData(self.saccadeVelocityData)
            self.saccadeAccelerationCurve.setData(self.saccadeAccelerationData)
            self.fixationCurve.setData(self.fixationData)
        else:
            pass
        
        if not sensor == None:
            recorded_time = time.time()

            feature_data = [features['pupil_dilation_rate'], features['blink_rate'], features['fixation_duration'], features['gaze_path_length'], features['gaze_transition_entropy'],
                            features['saccade_velocity'], features['saccade_acceleration'], features['gaze_dispersion']]

            # Include cognitive engagement level in the recorded data
            combined_feature_data = [recorded_time] + feature_data
            
            self.feature_data.append(combined_feature_data)
            
            if features['saccade_amplitude'] != self.last_saccade:
                t = time.time()
                saccade_data = [t] + [features['saccade_amplitude']]
                self.saccade_amp_data.append(saccade_data)
                self.last_saccade = features['saccade_amplitude']
            
            if not np.isnan(features['saccade_velocity']) and features['saccade_velocity'] > self.peak_saccadic:
                ti = time.time()
                peak_data = [ti] + [features['saccade_velocity']]
                self.peak_saccadic_data.append(peak_data)
                self.peak_saccadic = features['saccade_velocity']
                
    def E4_timer(self):
        self.worker_E4.E4_sensor_data_collect.connect(self.update_E4)
        self.worker_E4.ACC_data_collect.connect(self.update_ACC)
        self.worker_E4.BVP_IBI_data_collect.connect(self.update_BVP_IBI)
        self.worker_E4.GSR_TMP_data_collect.connect(self.update_GSR_TMP)
        self.worker_E4.GSR_TMP_data_collect.connect(self.update_E4_feature)
        
    def start_E4(self):
        
        self.start_E4_button.hide()
        self.graph_widget_E4.show()
        
        self.E4_ACC = []
        self.E4_BVP_IBI = []
        self.E4_GSR_TMP = []

        self.E4_timer()
        
    def update_E4(self, msg):
        
        self.data_e4 = np.roll(self.data_e4, -1, axis = 0)
        
        self.data_e4[-1, :] = msg
        
        for i, curve in enumerate(self.curves_e4):
           curve.setData(self.data_e4[:, i])

        # Refresh the graph widget
        self.graph_widget_E4.update()
        
    def update_ACC(self, msg):
        recorded_time = time.time()
        
        self.E4_ACC.append([recorded_time, msg[0]])
        
    def update_BVP_IBI(self, msg):
        recorded_time = time.time()
        
        self.E4_BVP_IBI.append([recorded_time, msg[1], msg[4]])
        
    def update_GSR_TMP(self, msg):
        recorded_time = time.time()
        
        self.E4_GSR_TMP.append([recorded_time, msg[2], msg[3]])
                
    def start_E4_feature(self):
        self.start_E4_feature_button.hide()
        self.gsrTonicPlotWidget.show()
        self.gsrPhasicPlotWidget.show()
    
    def init_E4_feature_plots(self):
        self.fs = 100
        self.cutoff = 0.05
        
        self.gsr_analyser = GSRAnalyzer_first(self.fs, self.cutoff)
        
        self.tonicData = []
        self.phasicData = []
        
        self.Tonic_Phasic = []
        
        self.tonicCurve = self.gsrTonicPlotWidget.plot(pen='y')
        self.phasicCurve = self.gsrPhasicPlotWidget.plot(pen='r')
        
        plot_widgets = [
            self.gsrTonicPlotWidget, self.gsrPhasicPlotWidget
        ]
        for widget in plot_widgets:
            widget.showGrid(x=True, y=True, alpha=0.3)
        
    def update_E4_feature(self, frame):
        recorded_time = time.time()
        frame = np.array(frame[2])
        self.tonic, self.phasic = self.gsr_analyser.process_frame(frame)
        
        self.tonicData.append(self.tonic)
        self.phasicData.append(self.phasic)
        
        self.tonicCurve.setData(self.tonicData)
        self.phasicCurve.setData(self.phasicData)
        
        self.Tonic_Phasic.append([recorded_time, self.tonic, self.phasic])

    def tcp_connect(self):
        if not self.stop_event.is_set():
            return
        self.stop_event.clear()
        self.tcp_client = TCP_Client(self.host, self.port)
        self.tcp_thread = threading.Thread(target=self.tcp_thread_works)
        self.tcp_thread.start()

    def tcp_disconnect(self):
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.tcp_thread.join()

    def tcp_thread_works(self):
        tobii = Tobii_handler()
        try:
            while not self.stop_event.is_set():
                data = ','.join(map(str, self.data_for_tcp(tobii)))
                if self.tcp_client.is_connected:
                    self.tcp_client.send_msg_to_server(data)
                else:
                    if self.tcp_client.connect_to_server() == "EOC":
                        self.stop_event.set()
                time.sleep(0.00833)
        except Exception as e:
            print(e)
        finally:
            self.tcp_client.close()
    
    def data_for_tcp(self, tobii):
        data_window = 5
        eyex = (tobii.gaze_data[0] + tobii.gaze_data[9]) / 2
        eyey = (tobii.gaze_data[1] + tobii.gaze_data[10]) / 2
        result = [eyex, eyey]
        if len(self.eyedata) < data_window:
            self.eyedata.append([eyex, eyey])
        else:
            self.eyedata.pop(0)
            self.eyedata.append([eyex, eyey])
            neweyex = sum([point[0] for point in self.eyedata]) / data_window
            neweyey = sum([point[1] for point in self.eyedata]) / data_window
            result = [neweyex, neweyey]
        return result

    def closeEvent(self, event):
        self.worker_E4.stopp()
        self.tcp_disconnect()
        super().closeEvent(event)
    
    def gameStart(self):
        self.gamestarted = 'started'
        
        self.recorded_data[-1] = self.recorded_data[-1] + [self.gamestarted]
        self.feature_data[-1] = self.feature_data[-1] + [self.gamestarted]
        self.E4_BVP_IBI[-1] = self.E4_BVP_IBI[-1] + [self.gamestarted]
        self.E4_GSR_TMP[-1] = self.E4_GSR_TMP[-1] + [self.gamestarted]
        self.saccade_amp_data[-1] = self.saccade_amp_data[-1] + [self.gamestarted]
        self.peak_saccadic_data[-1] = self.peak_saccadic_data[-1] + [self.gamestarted]
        self.Tonic_Phasic[-1] = self.Tonic_Phasic[-1] + [self.gamestarted]
        
    def saveData(self):
        if not self.recorded_data:
            print("No data to save.")
            
        datafilezip = zipfile.ZipFile(f'experiment record of {self.participant_name}.zip', 'x')

        fileName = "Eye Sensor data.csv"
        if fileName:
            with open(fileName, 'w', newline='') as file:
                writer = csv.writer(file)
                
                # Update headers to include participant name
                headers = ['Participant Name']
                headers += ['UNIX Time']
                headers += [f'{i}' for i in self.gazedatalist]
                headers += ['game started']
                writer.writerow(headers)

                # Add participant name to each record
                for i in range(len(self.recorded_data)):
                    self.recorded_data[i] = [self.participant_name] + self.recorded_data[i]

                # Write updated data to CSV
                for record in self.recorded_data:
                    writer.writerow(record)
            print(f"Sensor Data saved to {fileName}")
        datafilezip.write(fileName)
        os.remove(f'/python/{fileName}')
            
        fileName = "Eye Feature data.csv"
        if fileName:
            with open(fileName, 'w', newline='') as file:
                writer = csv.writer(file)
                
                # Update headers to include participant name
                headers = ['Participant Name', 'UNIX Time', 'Pupil Dilation Rate', 'Blink Rate', 'Fixation Duration', 'Gaze Path Length', 'Gaze Transition Entropy', 'Saccade Velocity', 'Saccade Acceleration', 'Gaze Dispersion', 'game started']
                writer.writerow(headers)

                # Add participant name to each record
                for i in range(len(self.feature_data)):
                    self.feature_data[i] = [self.participant_name] + self.feature_data[i]
                # Write updated data to CSV
                for record in self.feature_data:
                    writer.writerow(record)
            print(f"Sensor Data saved to {fileName}")
        datafilezip.write(fileName)
        os.remove(f'/python/{fileName}')
            
        '''
        fileName = "E4 ACC data.csv"
        if fileName:
            with open(fileName, 'w', newline='') as file:
                writer = csv.writer(file)
                
                # Update headers to include participant name
                headers = ['Participant Name', 'UNIX Time', 'E4 ACC', 'game started']
                writer.writerow(headers)

                # Add participant name to each record
                for i in range(len(self.E4_ACC)):
                    self.E4_ACC[i] = [self.participant_name] + self.E4_ACC[i] + self.gamestarted

                # Write updated data to CSV
                for record in self.E4_ACC:
                    writer.writerow(record)
            print(f"Sensor Data saved to {fileName}")
        datafilezip.write(fileName)
        os.remove(f'/python/{fileName}')
        '''
            
        fileName = "E4 BVP, IBI data.csv"
        if fileName:
            with open(fileName, 'w', newline='') as file:
                writer = csv.writer(file)
                
                # Update headers to include participant name
                headers = ['Participant Name', 'UNIX Time', 'E4 BVP', 'E4 IBI', 'game started']
                writer.writerow(headers)

                # Add participant name to each record
                for i in range(len(self.E4_BVP_IBI)):
                    self.E4_BVP_IBI[i] = [self.participant_name] + self.E4_BVP_IBI[i]

                # Write updated data to CSV
                for record in self.E4_BVP_IBI:
                    writer.writerow(record)
            print(f"Sensor Data saved to {fileName}")
        datafilezip.write(fileName)
        os.remove(f'/python/{fileName}')
            
        fileName = "E4 GSR, TMP data.csv"
        if fileName:
            with open(fileName, 'w', newline='') as file:
                writer = csv.writer(file)
                
                # Update headers to include participant name
                headers = ['Participant Name', 'UNIX Time', 'E4 GSR', 'E4 TMP', 'game started']
                writer.writerow(headers)

                # Add participant name to each record
                for i in range(len(self.E4_GSR_TMP)):
                    self.E4_GSR_TMP[i] = [self.participant_name] + self.E4_GSR_TMP[i]
                # Write updated data to CSV
                for record in self.E4_GSR_TMP:
                    writer.writerow(record)
            print(f"Sensor Data saved to {fileName}")
        datafilezip.write(fileName)
        os.remove(f'/python/{fileName}')
            
        fileName = "Saccade Amplitude data.csv"
        if fileName:
            with open(fileName, 'w', newline='') as file:
                writer = csv.writer(file)

                headers = ['Participant Name', 'UNIX Time', 'Saccade Amplitude', 'game started']
                writer.writerow(headers)

                for i in range(len(self.saccade_amp_data)):
                    self.saccade_amp_data[i] = [self.participant_name] + self.saccade_amp_data[i]

                for record in self.saccade_amp_data:
                    writer.writerow(record)
            print(f"Saccade Amplitude Data saved to {fileName}")
        datafilezip.write(fileName)
        os.remove(f'/python/{fileName}')
            
        fileName = "Peak Saccadic Velocity data.csv"
        if fileName:
            with open(fileName, 'w', newline='') as file:
                writer = csv.writer(file)

                headers = ['Participant Name', 'UNIX Time', 'Peak Saccadic Velocity', 'game started']
                writer.writerow(headers)

                for i in range(len(self.peak_saccadic_data)):
                    self.peak_saccadic_data[i] = [self.participant_name] + self.peak_saccadic_data[i]

                for record in self.peak_saccadic_data:
                    writer.writerow(record)
            print(f"Peak Saccadic Velocity Data saved to {fileName}")
        datafilezip.write(fileName)
        os.remove(f'/python/{fileName}')
        
        fileName = "Tonic and Phasic GSR data.csv"
        if fileName:
            with open(fileName, 'w', newline='') as file:
                writer = csv.writer(file)

                headers = ['Participant Name', 'UNIX Time', 'Tonic data', 'Phasic data', 'game started']
                writer.writerow(headers)

                for i in range(len(self.Tonic_Phasic)):
                    self.Tonic_Phasic[i] = [self.participant_name] + self.Tonic_Phasic[i]

                for record in self.Tonic_Phasic:
                    writer.writerow(record)
            print(f"Tonic and Phasic GSR Data saved to {fileName}")
        datafilezip.write(fileName)
        os.remove(f'/python/{fileName}')
        
        datafilezip.close()

##########################################################################
class Tobii_handler():
    def __init__(self):
        self.__gaze_data = [0] * 18
        self.__previous_gaze_data = [0] * 18
        self.lock = threading.Lock()

        self.eye_tracker = None
        self.is_ready = False
        self.safe_init_eye_tracker()

        # Blink rate
        self.blink_count = 0
        self.start_time = time.time()
        self.unit_time = time.time()
        self.blink_list = []

        #Pupil dilation
        self.pupil_diameters = []
        self.pupil_time_stamps = []
        self.pupilx = 0

        #Fixation duration
        self.fixation_start_time = None
        self.current_fixation_duration = 0
        self.fixation_durations = []

        # Saccade amplitude
        self.last_gaze_point_1 = None
        self.saccade_amplitudes = []

        # Gaze Path Length
        self.last_gaze_point_2 = None
        self.gaze_path_length = 0

        # Gaze Transition Entropy
        self.gaze_transitions = []
        self.gaze_entropy = 0

        # Lists for gaze points, timestamps, and velocities
        self.gaze_points = []
        self.gaze_timestamps = []
        self.saccade_velocities = []

        # Lists to store gaze points
        self.gaze_points_x = []
        self.gaze_points_y = []
        self.dispersionx = 0

    @property
    def gaze_data(self):
        with self.lock:
            return self.__gaze_data

    def safe_init_eye_tracker(self):
        try:
            self.eye_tracker = tr.find_all_eyetrackers()[0]
        except Exception as e:
            self.is_ready = False
            if e is IndexError:
                print("There are no Tobii eye trackers are detected.")
            else:
                print(e)
        else:
            self.is_ready = True
            self.eye_tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback)

    def gaze_data_callback(self, gaze_data):
        # Before updating __gaze_data, store its current state as previous
        with self.lock:
            self.__previous_gaze_data = self.__gaze_data.copy()

        try:
            self.__gaze_data[0], self.__gaze_data[1] = gaze_data.left_eye.gaze_point.position_on_display_area
            self.__gaze_data[2], self.__gaze_data[3], self.__gaze_data[4] = gaze_data.left_eye.gaze_point.position_in_user_coordinates
            self.__gaze_data[5] = gaze_data.left_eye.gaze_origin.validity
            self.__gaze_data[6] = gaze_data.left_eye.gaze_point.validity
            self.__gaze_data[7] = gaze_data.left_eye.pupil.validity
            self.__gaze_data[8] = gaze_data.left_eye.pupil.diameter

            self.__gaze_data[9], self.__gaze_data[10] = gaze_data.right_eye.gaze_point.position_on_display_area
            self.__gaze_data[11], self.__gaze_data[12], self.__gaze_data[13] = gaze_data.right_eye.gaze_point.position_in_user_coordinates
            self.__gaze_data[14] = gaze_data.right_eye.gaze_origin.validity
            self.__gaze_data[15] = gaze_data.right_eye.gaze_point.validity
            self.__gaze_data[16] = gaze_data.right_eye.pupil.validity
            self.__gaze_data[17] = gaze_data.right_eye.pupil.diameter
            
        except Exception as e:
            print("Error processing gaze data: ", e)

        # Gaze transition entropy
        # Right eye data 추가됨
        current_gaze_point = ((self.__gaze_data[0] + self.__gaze_data[9]) / 2 , (self.__gaze_data[1] + self.__gaze_data[10]) / 2)
        previous_gaze_point = ((self.__previous_gaze_data[0] + self.__previous_gaze_data[9]) / 2, (self.__previous_gaze_data[1] + self.__previous_gaze_data[10]) / 2)
        if previous_gaze_point != (0, 0):  # Check if previous data is valid
            self.gaze_transitions.append((current_gaze_point, previous_gaze_point))
            if len(self.gaze_transitions) > 100:  # Limit size of the list
                self.gaze_transitions.pop(0)

        # Store the current gaze point and timestamp
        this_time = time.time()
        self.gaze_points.append(current_gaze_point)
        self.gaze_timestamps.append(this_time)

    def extract_features(self):
        features = {}
        features['pupil_dilation_rate'] = self.calculate_pupil_dilation_rate()
        features['blink_rate'] = self.calculate_blink_rate()
        features['fixation_duration'] = self.calculate_fixation_duration()
        features['saccade_amplitude'] = self.calculate_saccade_amplitude()
        features['gaze_path_length'] = self.calculate_gaze_path_length()
        features['gaze_transition_entropy'] = self.calculate_gaze_transition_entropy()
        features['saccade_velocity'] = self.calculate_saccade_velocity()
        features['saccade_acceleration'] = self.calculate_saccade_acceleration()
        features['peak_saccadic_velocity'] = self.get_peak_saccadic_velocity()
        features['gaze_dispersion'] = self.calculate_gaze_dispersion()
        return features

    def calculate_pupil_dilation_rate(self):
        #Current time
        current_pupil_time = time.time()

        # Add current average pupil diameter to the list
        current_pupil_diameter = (self.__gaze_data[8] + self.__gaze_data[17]) / 2
        self.pupil_diameters.append(current_pupil_diameter)
        self.pupil_time_stamps.append(current_pupil_time)

        # Define a time window (e.g., 60 seconds)
        time_window = 10

        # Remove old data points that are outside the time window
        while self.pupil_time_stamps and current_pupil_time - self.pupil_time_stamps[self.pupilx] > time_window:
            self.pupil_time_stamps.pop(0)
            self.pupil_diameters.pop(0)
            self.pupilx += 1

        # Calculate the dilation rate if enough data is available
        if len(self.pupil_diameters) > 1:
            # Change in pupil diameter
            delta_pupil_diameter = self.pupil_diameters[-1] - self.pupil_diameters[0]

            # Change in time
            delta_time = self.pupil_time_stamps[-1] - self.pupil_time_stamps[0]

            # Pupil dilation rate (diameter change per second)
            pupil_dilation_rate = delta_pupil_diameter / delta_time if delta_time > 0 else 0

            return pupil_dilation_rate
        else:
            return 1212

    def calculate_blink_rate(self):
        current_time = time.time()
        time_window = 5
        time_unit = 0.5
        
        # Make sample list if blink_list is not long enough
        if len(self.blink_list) < (time_window / time_unit):
            self.blink_list = [0.3] * int(time_window / time_unit)

        # Check for blinks
        # Eye-origin 이 변할 때 blink 한 것으로 간주
        blink_occurred = (self.__previous_gaze_data[5] != self.__gaze_data[5]) or \
                         (self.__previous_gaze_data[14] != self.__gaze_data[14])
        
        if blink_occurred:
            self.blink_count += 1
            
        if current_time - self.unit_time >= time_unit:
            self.blink_list.append(self.blink_count)
            self.blink_count = 0
            self.unit_time = current_time
            
        if current_time - self.start_time >= time_window:
            self.blink_list.pop(0)
            self.start_time += time_unit
            
        blink_rate = sum(self.blink_list) / time_window

        return blink_rate

    def calculate_fixation_duration(self):
        # Define a threshold for gaze stability (e.g., in screen coordinate space)
        stability_threshold = 0.1  # Adjust as needed

        # Calculate distance between consecutive gaze points
        if self.__previous_gaze_data and self.__gaze_data:
            dx = (self.__gaze_data[0] + self.__gaze_data[9] - self.__previous_gaze_data[0] - self.__previous_gaze_data[9]) / 2
            dy = (self.__gaze_data[1] + self.__gaze_data[10] - self.__previous_gaze_data[1] - self.__previous_gaze_data[10]) / 2
            distance = np.sqrt(dx ** 2 + dy ** 2)

            # Check if the gaze is stable
            if distance < stability_threshold:
                if self.fixation_start_time is None:
                    self.fixation_start_time = time.time()  # Mark the start of a fixation
                else:
                    # Update current fixation duration
                    self.current_fixation_duration = time.time() - self.fixation_start_time
            else:
                if self.fixation_start_time is not None:
                    # Fixation ended, record its duration
                    self.fixation_durations.append(self.current_fixation_duration)
                    self.fixation_start_time = None
                    self.current_fixation_duration = 0

        # Return the most recent fixation duration
        return self.fixation_durations[-1] if self.fixation_durations else 0

    def calculate_saccade_amplitude(self):
        # Define a threshold for detecting saccades (e.g., in screen coordinate space)
        saccade_threshold = 0.2  # Adjust as needed

        # Calculate distance between consecutive gaze points
        if self.last_gaze_point_1 and self.__gaze_data:
            dx = (self.__gaze_data[0] + self.__gaze_data[9] - self.last_gaze_point_1[0] - self.last_gaze_point_1[2]) / 2
            dy = (self.__gaze_data[1] + self.__gaze_data[10] - self.last_gaze_point_1[1] - self.last_gaze_point_1[3]) / 2
            distance = np.sqrt(dx ** 2 + dy ** 2)

            # Check if a saccade occurred
            if distance > saccade_threshold:
                self.saccade_amplitudes.append(distance)

        # Update the last gaze point
        self.last_gaze_point_1 = (self.__gaze_data[0], self.__gaze_data[1], self.__gaze_data[9], self.__gaze_data[10])

        # Return the most recent saccade amplitude
        return self.saccade_amplitudes[-1] if self.saccade_amplitudes else 0

    def calculate_gaze_path_length(self):
        # Calculate distance between consecutive gaze points
        if self.last_gaze_point_2 and self.__gaze_data:
            dx = (self.__gaze_data[0] + self.__gaze_data[9] - self.last_gaze_point_2[0] - self.last_gaze_point_2[2]) / 2
            dy = (self.__gaze_data[1] + self.__gaze_data[10] - self.last_gaze_point_2[1] - self.last_gaze_point_2[3]) / 2
            distance = np.sqrt(dx ** 2 + dy ** 2)
            
            if np.isnan(distance):
                distance = 0

            # Add distance to total path length
            self.gaze_path_length += distance

        # Update the last gaze point
        self.last_gaze_point_2 = (self.__gaze_data[0], self.__gaze_data[1], self.__gaze_data[9], self.__gaze_data[10])

        return self.gaze_path_length

    def calculate_gaze_transition_entropy(self):
        # Need more than 1 component to calculate entropy
        if len(self.gaze_transitions) < 2:
            return 0

        # Calculate differences in gaze points
        differences = np.diff(self.gaze_transitions, axis=0)
        # np.linalg.norm - 요소의 n승의 합의 n제곱근을 구합. 기본은 n = 2
        magnitudes = np.linalg.norm(differences, axis=1)
        magnitudes_after = magnitudes[np.isfinite(magnitudes)]

        # Normalize the magnitudes to sum to 1
        if np.sum(magnitudes_after) == 0:
            return 0
        probabilities = magnitudes_after / np.sum(magnitudes_after)

        # Calculate entropy
        self.gaze_entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))  # Adding eps for numerical stability
        return self.gaze_entropy

    def calculate_saccade_velocity(self):
        # Ensure there are at least two points to calculate velocity
        if len(self.gaze_points) < 2:
            return 0

        # Calculate the distance and time difference between the last two gaze points
        dx = self.gaze_points[-1][0] - self.gaze_points[-2][0]
        dy = self.gaze_points[-1][1] - self.gaze_points[-2][1]
        distance = np.sqrt(dx ** 2 + dy ** 2)
        time_difference = self.gaze_timestamps[-1] - self.gaze_timestamps[-2]

        # Calculate and return the velocity
        if time_difference > 0:
            velocity = distance / time_difference
            self.saccade_velocities.append(velocity)
            return velocity
        else:
            return 0

    def calculate_saccade_acceleration(self):
        # Ensure there are at least two velocity points to calculate acceleration
        if len(self.saccade_velocities) < 2:
            return 0

        # Calculate acceleration
        last_velocity = self.saccade_velocities[-1]
        previous_velocity = self.saccade_velocities[-2]
        time_difference = self.gaze_timestamps[-1] - self.gaze_timestamps[-2]

        if time_difference > 0:
            acceleration = (last_velocity - previous_velocity) / time_difference
            return acceleration
        else:
            return 0

    def get_peak_saccadic_velocity(self):
        if self.saccade_velocities:
            # Find the maximum velocity among all saccade velocities
            return np.nanmax(self.saccade_velocities)
        return 0

    def calculate_gaze_dispersion(self):
        time_window = 10
        # Update gaze points lists
        # Right eye data 추가됨
        if np.isnan(self.__gaze_data[0]) and np.isnan(self.__gaze_data[9]):
            self.gaze_points_x.append(0)
        elif np.isnan(self.__gaze_data[0]) and not np.isnan(self.__gaze_data[9]):
            self.gaze_points_x.append(self.__gaze_data[9])
        elif not np.isnan(self.__gaze_data[0]) and np.isnan(self.__gaze_data[9]):
            self.gaze_points_x.append(self.__gaze_data[0])
        else:
            self.gaze_points_x.append((self.__gaze_data[0] + self.__gaze_data[9]) / 2)
        if np.isnan(self.__gaze_data[1]) and np.isnan(self.__gaze_data[10]):
            self.gaze_points_y.append(0)
        elif np.isnan(self.__gaze_data[1]) and not np.isnan(self.__gaze_data[10]):
            self.gaze_points_y.append(self.__gaze_data[10])
        elif not np.isnan(self.__gaze_data[1]) and np.isnan(self.__gaze_data[10]):
            self.gaze_points_y.append(self.__gaze_data[1])
        else:
            self.gaze_points_y.append((self.__gaze_data[1] + self.__gaze_data[10]) / 2)

        if len(self.gaze_timestamps) > 0 and self.gaze_points_x and self.gaze_timestamps[-1] - self.gaze_timestamps[self.dispersionx] > time_window:
            self.gaze_points_x.pop(0)
            self.gaze_points_y.pop(0)
            self.dispersionx += 1
        
        # Ensure there are enough gaze points
        if len(self.gaze_points_x) > 2:
            # Calculate the standard deviation of gaze points
            dispersion_x = np.std(self.gaze_points_x)
            dispersion_y = np.std(self.gaze_points_y)

            # The overall dispersion can be the average of X and Y dispersions
            return (dispersion_x + dispersion_y) / 2
        return 0

##########################################################################
class TCP_Client():
    def __init__(self, host='127.0.0.1', port=5005):
        self.is_connected = None
        self.server_address = (host, port)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect_to_server()

    def connect_to_server(self):
        try:
            self.client_socket.connect(self.server_address)
        except Exception as e:
            self.is_connected = False
            if str(e)[10:15] == "10056":
                # ERROR CODE [10056]: Forced quit occurred from Unity server.
                print("Unity server were forced to quit...<Press Enter to Exit>")
                return "EOC"  # End Of Connection
            elif str(e)[10:15] == "10061":
                # ERROR CODE [10061]: Unity server is not available.
                print("Unity server is not available. Trying to reconnect...")
                return "TRC"  # Try to ReConnect
        else:
            self.is_connected = True
            self.send_msg_to_server("Connected to the python client.")
        return "CTS"  # Connected To Server

    def send_msg_to_server(self, msg: str, encode_type='utf-8'):
        if not self.is_connected:
            print("The client is not connected to Unity server.")
            self.is_connected = False
            return
        try:
            self.client_socket.sendall(msg.encode(encode_type))
        except Exception as e:
            print(e)
            self.is_connected = False
            print("\nTrying to reconnect with Unity server...")
            self.connect_to_server()

    def close(self):
        self.client_socket.close()

##########################################################################
class InitialWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(InitialWindow, self).__init__()
        self.initUI()
        self.applicationWindow = None  # Initialize with None

    def initUI(self):
        self.setWindowTitle("Initial Window")

        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)
        self.layout = QVBoxLayout(self.centralWidget)

        # Button to open the Application window
        open_button = QPushButton("Open Application", self)
        open_button.clicked.connect(self.openApplication)
        self.layout.addWidget(open_button)

        self.setGeometry(100, 100, 300, 200)

    def openApplication(self):
        dialog = SessionDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            participant_name = dialog.participantName()
            self.applicationWindow = Application(participant_name)
            self.applicationWindow.show()
            self.close()
            
class E4Streamer(SensorStreamer):

    ########################
    ###### INITIALIZE ######
    ########################
    def __init__(self,
                 log_player_options=None, visualization_options=None,
                 print_status=True, print_debug=False, log_history_filepath=None):
        SensorStreamer.__init__(self, streams_info=None,
                                visualization_options=visualization_options,
                                log_player_options=log_player_options,
                                print_status=print_status, print_debug=print_debug,
                                log_history_filepath=log_history_filepath)

        ## TODO: Add a tag here for your sensor that can be used in log messages.
        #        Try to keep it under 10 characters long.
        #        For example, 'myo' or 'scale'.
        self._log_source_tag = 'E4'

        ## TODO: Initialize any state that your sensor needs.
        # Initialize counts
        self._num_segments = None

        # Initialize state
        self._buffer = b''
        self._buffer_read_size = 4096
        self._socket = None
        self._E4_sample_index = None  # The current Moticon timestep being processed (each timestep will send multiple messages)
        self._E4_message_start_time_s = None  # When a Moticon message was first received
        self._E4_timestep_receive_time_s = None  # When the first Moticon message for a Moticon timestep was received
        self._device_id1 = '6251CD'
        self._device_id2 = '1B35CD'

        # SELECT DATA TO STREAM
        self._acc = True  # 3-axis acceleration
        self._bvp = True  # Blood Volume Pulse
        self._gsr = True  # Galvanic Skin Response (Electrodermal Activity)
        self._tmp = True  # Temperature
        self._ibi = True  # Ibi Data

        # Specify the Moticon streaming configuration.
        self._E4_network_protocol = 'tcp'
        self._E4_network_ip = '127.0.0.7'
        self._E4_network_port = 28000

        ## TODO: Add devices and streams to organize data from your sensor.
        #        Data is organized as devices and then streams.
        #        For example, a Myo device may have streams for EMG and Acceleration.
        #        If desired, this could also be done in the connect() method instead.
        self.add_stream(device_name='ACC-empatica_e4',
                        stream_name='acc-values',
                        data_type='float32',
                        sample_size=[3],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=32,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'Acceleration data from empatica-e4.'
                             ),
                            ('Units', ''),
                            (SensorStreamer.metadata_data_headings_key,
                             ['acc_x', 'acc_y', 'acc_z']),
                        ]))
        self.add_stream(device_name='BVP-empatica_e4',
                        stream_name='bvp-values',
                        data_type='float32',
                        sample_size=[1],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=64,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'BVP data from empatica-e4.'
                             ),
                            ('Units', ''),
                            (SensorStreamer.metadata_data_headings_key,
                             ['bvp']),
                        ]))
        self.add_stream(device_name='GSR-empatica_e4',
                        stream_name='gsr-values',
                        data_type='float32',
                        sample_size=[1],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=4,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'GSR data from empatica-e4.'
                             ),
                            ('Units', ''),
                            (SensorStreamer.metadata_data_headings_key,
                             ['gsr']),
                        ]))
        self.add_stream(device_name='Tmp-empatica_e4',
                        stream_name='tmp-values',
                        data_type='float32',
                        sample_size=[1],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=4,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'Temperature data from empatica-e4.'
                             ),
                            ('Units', ''),
                            (SensorStreamer.metadata_data_headings_key,
                             ['tmp']),
                        ]))
        self.add_stream(device_name='Ibi-empatica_e4',
                        stream_name='ibi-values',
                        data_type='float32',
                        sample_size=[1],
                        # the size of data saved for each timestep; here, we expect a 2-element vector per timestep
                        sampling_rate_hz=64,  # the expected sampling rate for the stream
                        extra_data_info={},
                        # can add extra information beyond the data and the timestamp if needed (probably not needed, but see MyoStreamer for an example if desired)
                        # Notes can add metadata about the stream,
                        #  such as an overall description, data units, how to interpret the data, etc.
                        # The SensorStreamer.metadata_data_headings_key is special, and is used to
                        #  describe the headings for each entry in a timestep's data.
                        #  For example - if the data was saved in a spreadsheet with a row per timestep, what should the column headings be.
                        data_notes=OrderedDict([
                            ('Description', 'Ibi data from empatica-e4.'
                             ),
                            ('Units', ''),
                            (SensorStreamer.metadata_data_headings_key,
                             ['ibi']),
                        ]))

    #######################################
    # Connect to the sensor.
    # @param timeout_s How long to wait for the sensor to respond.
    def _connect(self, timeout_s=10):
        # Open a socket to the E4 network stream
        ## TODO: Add code for connecting to your sensor.
        #        Then return True or False to indicate whether connection was successful.
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(3)

        print("Connecting to server")
        self._socket.connect((self._E4_network_ip, self._E4_network_port))
        print("Connected to server\n")

        print("Devices available:")
        self._socket.send("device_list\r\n".encode())
        response = self._socket.recv(self._buffer_read_size)

        print(response.decode("utf-8"))

        print("Connecting to device")
        self._socket.send(("device_connect " + self._device_id1 + "\r\n").encode())
        response = self._socket.recv(self._buffer_read_size)
        print(response.decode("utf-8"))

        print("Pausing data receiving")
        self._socket.send("pause ON\r\n".encode())
        response = self._socket.recv(self._buffer_read_size)
        print(response.decode("utf-8"))

        if self._acc:
            print("Suscribing to ACC")
            self._socket.send(("device_subscribe " + 'acc' + " ON\r\n").encode())
            response = self._socket.recv(self._buffer_read_size)
            print(response.decode("utf-8"))
        if self._bvp:
            print("Suscribing to BVP")
            self._socket.send(("device_subscribe " + 'bvp' + " ON\r\n").encode())
            response = self._socket.recv(self._buffer_read_size)
            print(response.decode("utf-8"))
        if self._gsr:
            print("Suscribing to GSR")
            self._socket.send(("device_subscribe " + 'gsr' + " ON\r\n").encode())
            response = self._socket.recv(self._buffer_read_size)
            print(response.decode("utf-8"))
        if self._tmp:
            print("Suscribing to Tmp")
            self._socket.send(("device_subscribe " + 'tmp' + " ON\r\n").encode())
            response = self._socket.recv(self._buffer_read_size)
            print(response.decode("utf-8"))
        if self._ibi:
            print("Suscribing to Ibi")
            self._socket.send(("device_subscribe " + 'ibi' + " ON\r\n").encode())
            response = self._socket.recv(self._buffer_read_size)
            print(response.decode("utf-8"))

        print("Resuming data receiving")
        self._socket.send("pause OFF\r\n".encode())
        response = self._socket.recv(self._buffer_read_size)
        print(response.decode("utf-8"))

        self._log_status('Successfully connected to the E4 streamer.')

        if self._acc:
            infoACC = pylsl.StreamInfo('acc', 'ACC', 3, 32, 'int32', 'ACC-empatica_e4');
            global outletACC
            outletACC = pylsl.StreamOutlet(infoACC)
        if self._bvp:
            infoBVP = pylsl.StreamInfo('bvp', 'BVP', 1, 64, 'float32', 'BVP-empatica_e4');
            global outletBVP
            outletBVP = pylsl.StreamOutlet(infoBVP)
        if self._gsr:
            infoGSR = pylsl.StreamInfo('gsr', 'GSR', 1, 4, 'float32', 'GSR-empatica_e4');
            global outletGSR
            outletGSR = pylsl.StreamOutlet(infoGSR)
        if self._tmp:
            infoTmp = pylsl.StreamInfo('tmp', 'Tmp', 1, 4, 'float32', 'Tmp-empatica_e4');
            global outletTmp
            outletTmp = pylsl.StreamOutlet(infoTmp)
        if self._ibi:
            infoIBI = pylsl.StreamInfo('ibi','IBI', 1, 64, 'float32','Ibi-empatica_e4');
            global outletIbi
            outletIbi = pylsl.StreamOutlet(infoIBI)

        return True

    #######################################
    ###### INTERFACE WITH THE SENSOR ######
    #######################################

    ## TODO: Add functions to control your sensor and acquire data.
    #        [Optional but probably useful]

    # A function to read a timestep of data for the first stream.
    def _read_data(self):
        # For example, may want to return the data for the timestep
        #  and the time at which it was received.
        try:
            # print("Starting LSL streaming")

            rawdata = self._socket.recvfrom(self._buffer_read_size)
            # print(rawdata)
            response = rawdata[0].decode("utf-8")
            # print(type(response))
            # print(response)
            samples = response.split("\n")
            # print("SAMPLEs")
            # print(samples)
            streamer_list = []
            time_s_list = []
            data_list = []

            for i in range(len(samples) - 1):
                stream_type = samples[i].split()[0]
                if stream_type == "E4_Acc":
                    time_s = float(samples[i].split()[1].replace(',', '.'))
                    data = [int(samples[i].split()[2].replace(',', '.')), int(samples[i].split()[3].replace(',', '.')),
                            int(samples[i].split()[4].replace(',', '.'))]
                    # print(data)
                    outletACC.push_sample(data, timestamp=time_s)
                    # print('1')
                    # print(stream_type, time_s, data)
                    streamer_list.append(stream_type)
                    time_s_list.append(time_s)
                    data_list.append(data)
                if stream_type == "E4_Bvp":
                    time_s = float(samples[i].split()[1].replace(',', '.'))
                    data = float(samples[i].split()[2].replace(',', '.'))
                    outletBVP.push_sample([data], timestamp=time_s)
                    # print('2')
                    # print(stream_type, time_s, data)
                    streamer_list.append(stream_type)
                    time_s_list.append(time_s)
                    data_list.append(data)
                if stream_type == "E4_Gsr":
                    time_s = float(samples[i].split()[1].replace(',', '.'))
                    data = float(samples[i].split()[2].replace(',', '.'))
                    outletGSR.push_sample([data], timestamp=time_s)
                    # print('3')
                    # print(stream_type, time_s, data)
                    streamer_list.append(stream_type)
                    time_s_list.append(time_s)
                    data_list.append(data)
                if stream_type == "E4_Temperature":
                    time_s = float(samples[i].split()[1].replace(',', '.'))
                    data = float(samples[i].split()[2].replace(',', '.'))
                    outletTmp.push_sample([data], timestamp=time_s)
                    # print('4')
                    # print(stream_type, time_s, data)
                    streamer_list.append(stream_type)
                    time_s_list.append(time_s)
                    data_list.append(data)
                if stream_type == "E4_Ibi":
                    time_s = float(samples[i].split()[1].replace(',', '.'))
                    data = float(samples[i].split()[2].replace(',', '.'))
                    outletIbi.push_sample([data], timestamp=time_s)
                    # print('4')
                    # print(stream_type, time_s, data)
                    streamer_list.append(stream_type)
                    time_s_list.append(time_s)
                    data_list.append(data)

            return (streamer_list, time_s_list, data_list)

        except:
            self._log_error('\n\n***ERROR reading from E4Streamer:\n%s\n' % traceback.format_exc())
            time.sleep(1)
            return (None, None, None)

    #####################
    ###### RUNNING ######
    #####################

    ## TODO: Continuously read data from your sensor.
    # Loop until self._running is False.
    # Acquire data from your sensor as desired, and for each timestep
    #  call self.append_data(device_name, stream_name, time_s, data).
    def _run(self):
        try:
            print("Streaming...")
            while self._running:

                # Read and store data for stream 1.
                (stream_type, time_s, data) = self._read_data()
                if time_s is not None:
                    for i in range (len(stream_type)):
                        if stream_type[i] == "E4_Acc":
                            self.append_data('ACC-empatica_e4', 'acc-values', time_s[i], data[i])
                        if stream_type[i] == "E4_Bvp":
                            self.append_data('BVP-empatica_e4', 'bvp-values', time_s[i], data[i])
                        if stream_type[i] == "E4_Gsr":
                            self.append_data('GSR-empatica_e4', 'gsr-values', time_s[i], data[i])
                        if stream_type[i] == "E4_Temperature":
                            self.append_data('Tmp-empatica_e4', 'tmp-values', time_s[i], data[i])
                        if stream_type[i] == "E4_Ibi":
                            self.append_data('Ibi-empatica_e4', 'ibi-values', time_s[i], data[i])
        except KeyboardInterrupt:  # The program was likely terminated
            pass
        except:
            self._log_error('\n\n***ERROR RUNNING E4Streamer:\n%s\n' % traceback.format_exc())
        finally:
            ## TODO: Disconnect from the sensor if desired.
            self._socket.close()

    # Clean up and quit
    def quit(self):
        ## TODO: Add any desired clean-up code.
        self._log_debug('E4Streamer quitting')
        self._socket.close()
        SensorStreamer.quit(self)

    ###########################
    ###### VISUALIZATION ######
    ###########################

    # Specify how the streams should be visualized.
    # Return a dict of the form options[device_name][stream_name] = stream_options
    #  Where stream_options is a dict with the following keys:
    #   'class': A subclass of Visualizer that should be used for the specified stream.
    #   Any other options that can be passed to the chosen class.
    def get_default_visualization_options(self, visualization_options=None):
        # Start by not visualizing any streams.
        processed_options = {}
        for (device_name, device_info) in self._streams_info.items():
            processed_options.setdefault(device_name, {})
            for (stream_name, stream_info) in device_info.items():
                processed_options[device_name].setdefault(stream_name, {'class': None})

        ## TODO: Specify whether some streams should be visualized.
        #        Examples of a line plot and a heatmap are below.
        #        To not visualize data, simply omit the following code and just leave each streamer mapped to the None class as shown above.
        # Use a line plot to visualize the weight.
        processed_options['ACC-empatica_e4']['acc-values'] = \
            {'class': LinePlotVisualizer, #HeatmapVisualizer
             'single_graph': True,   # Whether to show each dimension on a subplot or all on the same plot.
             'plot_duration_s': 15,  # The timespan of the x axis (will scroll as more data is acquired).
             'downsample_factor': 1, # Can optionally downsample data before visualizing to improve performance.
             }
        processed_options['BVP-empatica_e4']['bvp-values'] = \
            {'class': LinePlotVisualizer,  # HeatmapVisualizer
             'single_graph': True,  # Whether to show each dimension on a subplot or all on the same plot.
             'plot_duration_s': 15,  # The timespan of the x axis (will scroll as more data is acquired).
             'downsample_factor': 1,  # Can optionally downsample data before visualizing to improve performance.
             }
        processed_options['GSR-empatica_e4']['gsr-values'] = \
            {'class': LinePlotVisualizer,  # HeatmapVisualizer
             'single_graph': True,  # Whether to show each dimension on a subplot or all on the same plot.
             'plot_duration_s': 15,  # The timespan of the x axis (will scroll as more data is acquired).
             'downsample_factor': 1,  # Can optionally downsample data before visualizing to improve performance.
             }
        processed_options['Tmp-empatica_e4']['tmp-values'] = \
            {'class': LinePlotVisualizer,  # HeatmapVisualizer
             'single_graph': True,  # Whether to show each dimension on a subplot or all on the same plot.
             'plot_duration_s': 15,  # The timespan of the x axis (will scroll as more data is acquired).
             'downsample_factor': 1,  # Can optionally downsample data before visualizing to improve performance.
             }
        processed_options['Ibi-empatica_e4']['ibi-values'] = \
            {'class': LinePlotVisualizer,  # HeatmapVisualizer
             'single_graph': True,  # Whether to show each dimension on a subplot or all on the same plot.
             'plot_duration_s': 15,  # The timespan of the x axis (will scroll as more data is acquired).
             'downsample_factor': 1,  # Can optionally downsample data before visualizing to improve performance.
             }

        # Override the above defaults with any provided options.
        if isinstance(visualization_options, dict):
            for (device_name, device_info) in self._streams_info.items():
                if device_name in visualization_options:
                    device_options = visualization_options[device_name]
                    # Apply the provided options for this device to all of its streams.
                    for (stream_name, stream_info) in device_info.items():
                        for (k, v) in device_options.items():
                            processed_options[device_name][stream_name][k] = v

        return processed_options

##########################################################################
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    initialWin = InitialWindow()
    initialWin.show()
    sys.exit(app.exec_())