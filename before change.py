import sys
import csv
import cv2
import time
import math
import socket
import zipfile
import threading
import matplotlib.pyplot as plt
import tobii_research as tr
import pyqtgraph as pg
import numpy as np
from datetime import datetime
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QPushButton, QDialog, QVBoxLayout, QLabel, QDialogButtonBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer


##########################################################################
class FeaturePlotWindow(QtWidgets.QMainWindow):
    def __init__(self, tobii_handler, participant_name):  # Add participant_name parameter
        super(FeaturePlotWindow, self).__init__()
        self.tobii_handler = tobii_handler # tobii_handler 받을 필요 없음
        self.participant_name = participant_name
        self.recording = False
        self.recorded_data = []
        self.conginitive_data = []
        self.markEvent_data = []
        self.peak_saccadic_data = []
        self.saccade_amp_data = []
        self.peak_saccadic = 0
        self.last_saccade = 0
        self.initUI()
        self.initPlots()
        self.startUpdateTimer()

    def initUI(self):
        self.setWindowTitle("Feature Plots")

        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)
        gridLayout = QtWidgets.QGridLayout(self.centralWidget)


        # Create plot widgets
        self.pupilPlotWidget = pg.PlotWidget(title="Pupil Dilation Rate")
        self.blinkPlotWidget = pg.PlotWidget(title="Blink Rate")
        self.fixationPlotWidget = pg.PlotWidget(title="Fixation Duration")
        self.saccadePlotWidget = pg.PlotWidget(title="Saccade Amplitude")
        self.gazePathPlotWidget = pg.PlotWidget(title="Gaze Path Length")
        self.gazeEntropyPlotWidget = pg.PlotWidget(title="Gaze Transition Entropy")
        self.saccadeVelocityPlotWidget = pg.PlotWidget(title="Saccade Velocity")
        self.saccadeAccelerationPlotWidget = pg.PlotWidget(title="Saccade Acceleration")
        self.peakSaccadicVelocityPlotWidget = pg.PlotWidget(title="Peak Saccadic Velocity")
        self.gazeDispersionPlotWidget = pg.PlotWidget(title="Gaze Dispersion")


        # Add widgets to grid layout
        gridLayout.addWidget(self.pupilPlotWidget, 0, 0)
        gridLayout.addWidget(self.blinkPlotWidget, 0, 1)
        gridLayout.addWidget(self.fixationPlotWidget, 1, 0)
        gridLayout.addWidget(self.saccadePlotWidget, 1, 1)
        gridLayout.addWidget(self.gazePathPlotWidget, 2, 0)
        gridLayout.addWidget(self.gazeEntropyPlotWidget, 2, 1)
        gridLayout.addWidget(self.saccadeVelocityPlotWidget, 3, 0)
        gridLayout.addWidget(self.saccadeAccelerationPlotWidget, 3, 1)
        gridLayout.addWidget(self.peakSaccadicVelocityPlotWidget, 4, 0)
        gridLayout.addWidget(self.gazeDispersionPlotWidget, 4, 1)

        # Create a matplotlib figure for the 3D SVM plot
        self.svmCanvas = FigureCanvas(Figure())
        self.svmAx = self.svmCanvas.figure.add_subplot(projection='3d')  # Create 3D subplot
        self.svmAx.set_title("SVM Visualization")  # Set title for SVM plot
        gridLayout.addWidget(self.svmCanvas, 0, 2, 3, 1)  # Add to layout

        # Create a matplotlib figure for the Gradient Boosting Tree plot
        self.gbCanvas = FigureCanvas(Figure())
        self.gbAx = self.gbCanvas.figure.add_subplot(projection='3d')
        self.gbAx.set_title("Gradient Boosting Tree Visualization")  # Set title for GBT plot
        gridLayout.addWidget(self.gbCanvas, 3, 2, 2, 1)  # Add to layout

        # Adjust geometry for wider window to accommodate 3D plots
        self.setGeometry(100, 50, 1200, 1000)  # Adjusted for extra plots

        # Add checkboxes for selecting the visualization
        self.svmCheckbox = QtWidgets.QCheckBox("Show SVM Visualization", self.centralWidget)
        self.gbCheckbox = QtWidgets.QCheckBox("Show Gradient Boosting Tree Visualization", self.centralWidget)

        gridLayout.addWidget(self.svmCheckbox, 5, 2)  # Position below the 3D plot
        gridLayout.addWidget(self.gbCheckbox, 6, 2)
        
        self.svmCheckbox.setCheckState(2)
        self.gbCheckbox.setCheckState(2)

        self.svmCheckbox.stateChanged.connect(self.updatePlotVisibility)
        self.gbCheckbox.stateChanged.connect(self.updatePlotVisibility)

        # SVM Feature Selection Checkboxes
        self.svmBlinkCheckbox = QtWidgets.QCheckBox("Blink Data (SVM)", self.centralWidget)
        self.svmSaccadeCheckbox = QtWidgets.QCheckBox("Saccade Data (SVM)", self.centralWidget)
        self.svmVelocityCheckbox = QtWidgets.QCheckBox("Saccade Velocity (SVM)", self.centralWidget)
        gridLayout.addWidget(self.svmBlinkCheckbox, 7, 2)
        gridLayout.addWidget(self.svmSaccadeCheckbox, 8, 2)
        gridLayout.addWidget(self.svmVelocityCheckbox, 9, 2)

        # Gradient Boosting Feature Selection Checkboxes
        self.gbBlinkCheckbox = QtWidgets.QCheckBox("Blink Data (GB)", self.centralWidget)
        self.gbSaccadeCheckbox = QtWidgets.QCheckBox("Saccade Data (GB)", self.centralWidget)
        self.gbVelocityCheckbox = QtWidgets.QCheckBox("Saccade Velocity (GB)", self.centralWidget)
        gridLayout.addWidget(self.gbBlinkCheckbox, 10, 2)
        gridLayout.addWidget(self.gbSaccadeCheckbox, 11, 2)
        gridLayout.addWidget(self.gbVelocityCheckbox, 12, 2)

        # Connect checkboxes directly without interdependent logic
        self.svmBlinkCheckbox.stateChanged.connect(self.updatePlots)
        self.svmSaccadeCheckbox.stateChanged.connect(self.updatePlots)
        self.svmVelocityCheckbox.stateChanged.connect(self.updatePlots)

        self.gbBlinkCheckbox.stateChanged.connect(self.updatePlots)
        self.gbSaccadeCheckbox.stateChanged.connect(self.updatePlots)
        self.gbVelocityCheckbox.stateChanged.connect(self.updatePlots)

        # Buttons for recording and saving data
        self.startRecordingButton = QPushButton("Start Recording", self)
        self.startRecordingButton.clicked.connect(self.startRecording)
        gridLayout.addWidget(self.startRecordingButton, 13, 0)

        self.stopRecordingButton = QPushButton("Stop Recording", self)
        self.stopRecordingButton.clicked.connect(self.stopRecording)
        gridLayout.addWidget(self.stopRecordingButton, 13, 1)

        self.saveDataButton = QPushButton("Save Data to CSV", self)
        self.saveDataButton.clicked.connect(self.saveData)
        gridLayout.addWidget(self.saveDataButton, 13, 2)

        # Cognitive Engagement Slider
        self.cognitiveEngagementLabel = QLabel("Cognitive Engagement Annotation:")
        self.cognitiveEngagementSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.cognitiveEngagementSlider.setMinimum(1)
        self.cognitiveEngagementSlider.setMaximum(3)
        self.cognitiveEngagementSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.cognitiveEngagementSlider.setTickInterval(1)
        self.cognitiveEngagementSlider.setValue(2)
        self.cognitiveEngagementSlider.valueChanged.connect(self.engagement)

        # Slider labels
        self.lowLabel = QLabel("Low")
        self.highLabel = QLabel("High")

        # Layout for slider and labels
        self.sliderLayout = QtWidgets.QHBoxLayout()
        self.sliderLayout.addWidget(self.lowLabel)
        self.sliderLayout.addWidget(self.cognitiveEngagementSlider)
        self.sliderLayout.addWidget(self.highLabel)

        # Create a widget to hold the layout
        self.sliderWidget = QtWidgets.QWidget()
        self.sliderWidget.setLayout(self.sliderLayout)

        gridLayout.addWidget(self.cognitiveEngagementLabel, 14, 0)
        gridLayout.addWidget(self.sliderWidget, 14, 1)

        # Add Event Mark button
        self.eventMarkButton = QPushButton("Mark Event", self)
        self.eventMarkButton.clicked.connect(self.markEvent)
        gridLayout.addWidget(self.eventMarkButton, 15, 0)

        # Initialize a flag to track event marking
        self.eventMarked = False
        
    def engagement(self, value):
        t = time.time()
        if value == '1':
            cog = 'Low'
        elif value == '2':
            cog = 'middle'
        else:
            cog = 'High'
        data = [t] + [cog]
        self.conginitive_data.append(data)
        
    def startRecording(self):
        self.recording = True
        self.recorded_data = []

    def stopRecording(self):
        self.recording = False

    def markEvent(self):
        t = time.time()
        self.eventMarked = True
        data = [t] + [self.eventMarked]
        self.markEvent_data.append(data)
        self.eventMarked = False

    def saveData(self):
        if not self.recorded_data:
            print("No data to save.")
        
        if not self.conginitive_data:
            print("No cognitive data to save.")
        
        if not self.markEvent_data:
            print("No markevent data to save.")

        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Sensor data CSV File", "", "CSV Files (*.csv)", options=options)
        if fileName:
            with open(fileName, 'w', newline='') as file:
                writer = csv.writer(file)

                # Calculate the number of gaze data columns
                num_gaze_data_columns = len(self.recorded_data[0]) - 9  # Adjust 10 based on the number of feature columns + 1 for UNIX time

                # Update headers to include participant name
                headers = ['Participant Name', 'UNIX Time', 'Pupil Dilation Rate', 'Blink Rate', 'Fixation Duration', 'Gaze Path Length', 'Gaze Transition Entropy', 'Saccade Velocity', 'Saccade Acceleration', 'Gaze Dispersion']
                headers += [f'Gaze Data {i + 1}' for i in range(num_gaze_data_columns)]
                writer.writerow(headers)

                # Add participant name to each record
                for i in range(len(self.recorded_data)):
                    self.recorded_data[i] = [self.participant_name] + self.recorded_data[i]

                # Write updated data to CSV
                for record in self.recorded_data:
                    writer.writerow(record)
                    
            print(f"Sensor Data saved to {fileName}")
                    
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Cognitive data CSV File", "", "CSV Files (*.csv)", options=options)
        if fileName:
            with open(fileName, 'w', newline='') as file:
                writer = csv.writer(file)

                headers = ['Participant Name', 'UNIX Time', 'Cognitive Engagement Level']
                writer.writerow(headers)

                for i in range(len(self.conginitive_data)):
                    self.conginitive_data[i] = [self.participant_name] + self.conginitive_data[i]

                for record in self.conginitive_data:
                    writer.writerow(record)

            print(f"Cognitive Data saved to {fileName}")
            
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Eventmarked data CSV File", "", "CSV Files (*.csv)", options=options)
        if fileName:
            with open(fileName, 'w', newline='') as file:
                writer = csv.writer(file)

                headers = ['Participant Name', 'UNIX Time', 'Event Mark']
                writer.writerow(headers)

                for i in range(len(self.markEvent_data)):
                    self.markEvent_data[i] = [self.participant_name] + self.markEvent_data[i]

                for record in self.markEvent_data:
                    writer.writerow(record)

            print(f"Eventmarked Data saved to {fileName}")
            
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Saccade Amplitude data CSV File", "", "CSV Files (*.csv)", options=options)
        if fileName:
            with open(fileName, 'w', newline='') as file:
                writer = csv.writer(file)

                headers = ['Participant Name', 'UNIX Time', 'Saccade Amplitude']
                writer.writerow(headers)

                for i in range(len(self.saccade_amp_data)):
                    self.saccade_amp_data[i] = [self.participant_name] + self.saccade_amp_data[i]

                for record in self.saccade_amp_data:
                    writer.writerow(record)

            print(f"Saccade Amplitude Data saved to {fileName}")
            
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Peak Saccadic Velocity data CSV File", "", "CSV Files (*.csv)", options=options)
        if fileName:
            with open(fileName, 'w', newline='') as file:
                writer = csv.writer(file)

                headers = ['Participant Name', 'UNIX Time', 'Peak Saccadic Velocity']
                writer.writerow(headers)

                for i in range(len(self.peak_saccadic_data)):
                    self.peak_saccadic_data[i] = [self.participant_name] + self.peak_saccadic_data[i]

                for record in self.peak_saccadic_data:
                    writer.writerow(record)

            print(f"Peak Saccadic Velocity Data saved to {fileName}")

    def initPlots(self):
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

        # Initialize data lists for SVM and Gradient Boosting Tree
        self.svmDataX = []
        self.svmDataY = []
        self.svmDataZ = []
        self.svmDataLabels = []

        self.gbDataX = []
        self.gbDataY = []
        self.gbDataZ = []
        self.gbDataLabels = []

    def startUpdateTimer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updatePlots)
        self.timer.start(1000)

    def updatePlots(self):
        features = self.tobii_handler.extract_features() # 쓰레드에서 dict 받는 걸로 변경

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

        # Create mesh for the plots (50 * 50)
        xx, yy = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
        zz = np.sin(np.sqrt(xx ** 2 + yy ** 2))

        # Update SVM and Gradient Boosting Tree plots
        if self.svmCheckbox.isChecked():
            self.updateSVMPlot(xx, yy, zz, self.blinkData, self.saccadeData, self.saccadeVelocityData)
        if self.gbCheckbox.isChecked():
            self.updateGradientBoostingPlot(xx, yy, zz, self.blinkData, self.saccadeData, self.saccadeVelocityData)

        if self.recording:
            recorded_time = time.time()

            feature_data = [features['pupil_dilation_rate'], features['blink_rate'], features['fixation_duration'], features['gaze_path_length'], features['gaze_transition_entropy'],
                            features['saccade_velocity'], features['saccade_acceleration'], features['gaze_dispersion']]
            
            peak_data = [features['peak_saccadic_velocity']]

            raw_sensor_data = self.tobii_handler.gaze_data

            # Include cognitive engagement level in the recorded data
            combined_data = [recorded_time] + feature_data + raw_sensor_data

            self.recorded_data.append(combined_data)
            
            if features['saccade_amplitude'] != self.last_saccade:
                t = time.time()
                saccade_data = [t] + [features['saccade_amplitude']]
                self.saccade_amp_data.append(saccade_data)
                self.last_saccade = features['saccade_amplitude']
            
            if features['peak_saccadic_velocity'] != self.peak_saccadic and not math.isnan(features['peak_saccadic_velocity']):
                ti = time.time()
                peak_data = [ti] + [features['peak_saccadic_velocity']]
                self.peak_saccadic_data.append(peak_data)
                self.peak_saccadic = features['peak_saccadic_velocity']

    def updatePlotVisibility(self):
        # Toggle visibility based on checkboxes
        self.svmCanvas.setVisible(self.svmCheckbox.isChecked())
        self.gbCanvas.setVisible(self.gbCheckbox.isChecked())

    def updateSVMPlot(self, xx, yy, zz, blink_rate, saccade_amplitude, saccade_velocity):
        # Select features based on checkboxes
        selected_features = []
        if self.svmBlinkCheckbox.isChecked():
            selected_features.append(blink_rate)
        if self.svmSaccadeCheckbox.isChecked():
            selected_features.append(saccade_amplitude)
        if self.svmVelocityCheckbox.isChecked():
            selected_features.append(saccade_velocity)
            
        # Handle the case when no checkboxes are selected
        if not selected_features:
            # Skip the update
            return

        X_svm = np.column_stack(selected_features)
        print(selected_features)
        if self.svmVelocityCheckbox.isChecked():
            imputer = SimpleImputer(strategy='mean')
            X_svm = imputer.fit_transform(X_svm)
        Y_svm = [0] * len(X_svm)
            
        for i in range(len(Y_svm)):
            k = 0
            for j in range(len(selected_features)):
                if X_svm[i][j] in blink_rate and X_svm[i][j] < 0.5:
                    k += 1
                if X_svm[i][j] in saccade_amplitude and not X_svm[i - 1][j] == X_svm[i][j]:
                    k += 1
                if X_svm[i][j] in saccade_velocity and X_svm[i][j] < 2:
                    k += 1
            if len(selected_features) == 1:
                Y_svm[i] = 1 if k == 1 else 0
            elif len(selected_features) == 2:
                Y_svm[i] = 1 if k == 2 else 0
            else:
                Y_svm[i] = 1 if k >= 2 else 0
        
        # Fit SVM model
        model = svm.SVC(kernel='linear')
        try:
            model.fit(X_svm, np.array(Y_svm))
        except ValueError:
            print(f'Not enough classes yet')
            return
        except Exception as e:
            print(f'Error: {e}')

        # Prepare meshgrid based on the number of selected features
        if len(selected_features) == 1:
            # If only one feature is selected, use a 1D grid
            Z = model.decision_function(xx.ravel().reshape(-1, 1))
        elif len(selected_features) == 2:
            # If two features are selected, use a 2D grid
            Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
        else:
            # Handle cases with more than 2 features (if applicable)
            # This part might require additional logic depending on your data
            Z = model.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
            Z = Z.reshape(xx.shape)

        # Clear and update plot
        self.svmAx.clear()
        if len(selected_features) == 1:
            self.svmAx.plot(xx.ravel(), Z, color='r')  # Adjust this line as needed for 1D plot
        elif len(selected_features) == 2:
            self.svmAx.plot_surface(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
        else:
            self.svmAx.scatter(xx, yy, zz, c = Z, cmap='viridis', alpha=0.5)

        # Scatter plot and labels for the SVM
        for i, point in enumerate(X_svm):
            label = "Not Cognitive Engagement" if Y_svm[i] == 0 else "Cognitive Engagement"
            
            # Plot scatter points
            if len(selected_features) == 1:
                self.svmAx.scatter(point[0], 0, zs=0, color='black')  # Adjust for 1D plot
                self.svmAx.text(point[0], 0, 0.5, label)
            elif len(selected_features) == 2:
                self.svmAx.scatter(point[0], point[1], zs=0, color='black')  # zs sets the z coordinate
                self.svmAx.text(point[0], point[1], 0.5, label)
            else:
                self.svmAx.scatter(point[0], point[1], point[2], color='black')  # zs sets the z coordinate
                self.svmAx.text(point[0], point[1], point[2], label)

        self.svmCanvas.draw()

    def updateGradientBoostingPlot(self, xx, yy, zz, blink_data, saccade_data, saccade_velocity_data):
        # Select features based on checkboxes
        selected_features_gb = []
        if self.gbBlinkCheckbox.isChecked():
            selected_features_gb.append(blink_data)
        if self.gbSaccadeCheckbox.isChecked():
            selected_features_gb.append(saccade_data)
        if self.gbVelocityCheckbox.isChecked():
            selected_features_gb.append(saccade_velocity_data)

        # Handle the case when no checkboxes are selected
        if not selected_features_gb:
            # Skip the update
            return

        X_gb = np.column_stack(selected_features_gb)
        if self.gbVelocityCheckbox.isChecked():
            imputer = SimpleImputer(strategy='mean')
            X_gb = imputer.fit_transform(X_gb)
        Y_gb = [0] * len(X_gb)
            
        for i in range(len(Y_gb)):
            k = 0
            for j in range(len(selected_features_gb)):
                if X_gb[i][j] in blink_data and X_gb[i][j] < 0.5:
                    k += 1
                if X_gb[i][j] in saccade_data and not X_gb[i - 1][j] == X_gb[i][j]:
                    k += 1
                if X_gb[i][j] in saccade_velocity_data and X_gb[i][j] < 2:
                    k += 1
            if len(selected_features_gb) == 1:
                Y_gb[i] = 1 if k == 1 else 0
            elif len(selected_features_gb) == 2:
                Y_gb[i] = 1 if k >= 1 else 0
            else:
                Y_gb[i] = 1 if k >= 2 else 0

        # Fit Gradient Boosting model
        gb_model = GradientBoostingClassifier()
        try:
            gb_model.fit(X_gb, np.array(Y_gb))
        except ValueError:
            print(f'Not enough classes yet')
        except Exception as e:
            print(f'Error: {e}')


        # Prepare meshgrid based on the number of selected features
        if len(selected_features_gb) == 1:
            # If only one feature is selected, use a 1D grid
            Z_gb = gb_model.decision_function(xx.ravel().reshape(-1, 1))
        elif len(selected_features_gb) == 2:
            # If two features are selected, use a 2D grid
            Z_gb = gb_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z_gb = Z_gb.reshape(xx.shape)
        else:
            # Handle cases with more than 2 features (if applicable)
            # This part might require additional logic depending on your data
            Z_gb = gb_model.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
            Z_gb = Z_gb.reshape(xx.shape)

        # Clear previous plot and create a new one
        self.gbAx.clear()
        if len(selected_features_gb) == 1:
            self.gbAx.plot(xx.ravel(), Z_gb, color='r')  # Adjust this line as needed for 1D plot
        elif len(selected_features_gb) == 2:
            self.gbAx.plot_surface(xx, yy, Z_gb, cmap=plt.cm.coolwarm, alpha=0.5)
        else:
            self.gbAx.scatter(xx, yy, zz, c = Z_gb, cmap='viridis', alpha=0.5)

        # Scatter plot and labels for Gradient Boosting
        for i, point in enumerate(X_gb):
            label_gb = "Not Cognitive Engagement" if Y_gb[i] == 0 else "Cognitive Engagement"
            
            # Plot scatter points
            if len(selected_features_gb) == 1:
                self.gbAx.scatter(point[0], 0, zs=0, color='black')  # Adjust for 1D plot
                self.gbAx.text(point[0], 0, 0.5, label_gb)
            elif len(selected_features_gb) == 2:
                self.gbAx.scatter(point[0], point[1], zs=0, color='black')  # zs sets the z coordinate
                self.gbAx.text(point[0], point[1], 0.5, label_gb)  # Adjust z position as needed
            else:
                self.gbAx.scatter(point[0], point[1], point[2], color='black')  # zs sets the z coordinate
                self.gbAx.text(point[0], point[1], point[2], label_gb)  # Adjust z position as needed

        self.gbCanvas.draw()

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
    sensor_data_collect = QtCore.pyqtSignal(list)
    feature_data_collect = QtCore.pyqtSignal(dict)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.tobii_handler = Tobii_handler()

    def run(self):
        self.channels = 18
        self.points = 100
        self.data = np.zeros((self.points, self.channels))
        
        self.timer_sensor = QtCore.QTimer()
        self.timer_sensor.timeout.connect(self.update_data)
        self.timer_sensor.start()
        
        self.timer_feature = QtCore.Qtimer()
        self.timer_feature.timeout.connect(self.feature_ex)
        self.timer_feature.start(1000)
    
    def update_data(self):
        new_data = self.tobii_handler.gaze_data

        # Shift the data matrix up by one row
        self.data = np.roll(self.data, -1, axis=0)

        # Update the last row with new gaze data
        self.data[-1, :] = new_data
        
        self.data_collect.emit(self.data)
        
    def feature_ex(self):
        self.feature = self.tobii_handler.extract_features()
        self.feature_data_collect.emit(self.feature)

##########################################################################
class Application(QtWidgets.QMainWindow):
    def __init__(self, participant_name, host='127.0.0.1', port=5005):
        super(Application, self).__init__()

        self.participant_name = participant_name
        
        self.host = host
        self.port = port
        
        self.eyedata = []

        # Initialize Tobii handling instance
        self.tobii_handler = Tobii_handler()

        # Initialize TCP client object
        self.tcp_client = TCP_Client(host, port)

        # Initialize thread stop event flag
        self.stop_event = threading.Event()

        # UI Initialization
        self.initUI()

        # Initialize plots
        self.init_plots()
    
    def initUI(self):
        self.setWindowTitle("PyQt Graphs")

        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)

        self.layout = QVBoxLayout(self.centralWidget)

        # Start button
        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start)
        self.layout.addWidget(self.start_button)
        
        # Graph widget
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graph_widget)
        self.graph_widget.hide()

        # Button to open the Feature Plot window
        self.openFeaturePlotButton = QPushButton("Open Feature Plot", self)
        self.openFeaturePlotButton.clicked.connect(self.openFeaturePlotWindow)
        self.layout.addWidget(self.openFeaturePlotButton)

        # Exit button
        self.exit_button = QPushButton("Exit", self)
        self.exit_button.clicked.connect(self.close)
        self.layout.addWidget(self.exit_button)

        # TCP connect/disconnect buttons
        self.tcp_connect_button = QPushButton("TCP Connect to Unity 3D", self)
        self.tcp_connect_button.clicked.connect(self.tcp_connect)
        self.layout.addWidget(self.tcp_connect_button)

        self.tcp_disconnect_button = QPushButton("TCP Disconnect from Unity 3D", self)
        self.tcp_disconnect_button.clicked.connect(self.tcp_disconnect)
        self.layout.addWidget(self.tcp_disconnect_button)

        self.setGeometry(100, 100, 800, 600)

    def start(self):
        
        self.start_button.hide()
        self.graph_widget.show()

        # Start the update timer
        # Call 'update' for every 50 ms
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

        # Start the TCP thread
        self.tcp_thread = threading.Thread(target=self.tcp_thread_works)
        self.tcp_thread.start()

    def init_plots(self):
        self.num_channels = 18  # Total number of channels
        self.num_rows = 6  # Number of rows
        self.num_cols = 3  # Number of columns
        self.num_points = 100
        self.data = np.zeros((self.num_points, self.num_channels)) # Make arrays
        

        self.plots = []
        self.curves = []

        for i in range(self.num_channels):
            # Determine the row and column for the current plot
            row = i // self.num_cols
            col = i % self.num_cols

            plot = self.graph_widget.addPlot(row=row, col=col, title=f'Channel {i + 1}')
            pen = pg.mkPen(color=(i, self.num_channels), width=2)
            curve = plot.plot(self.data[:, i], pen=pen)
            plot.showGrid(x=True, y=True)
            self.plots.append(plot)
            self.curves.append(curve)
            
    def data_thread(self):
        worker = Worker(self)
        worker.data_collect.connect(self.update)
        worker.start(50)

    def update(self):
        # Fetch new gaze data
        new_data = self.tobii_handler.gaze_data

        # Shift the data matrix up by one row
        self.data = np.roll(self.data, -1, axis=0)

        # Update the last row with new gaze data
        self.data[-1, :] = new_data

        # Update the plot curves with new data
        for i, curve in enumerate(self.curves):
           curve.setData(self.data[:, i])

        # Refresh the graph widget
        self.graph_widget.update()
        
        # features 변수 삭제

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
        try:
            while not self.stop_event.is_set():
                data = ','.join(map(str, self.data_for_tcp()))
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
            
    def data_for_tcp(self):
        data_window = 5
        eyex = (self.tobii_handler.gaze_data[0] + self.tobii_handler.gaze_data[9]) / 2
        eyey = (self.tobii_handler.gaze_data[1] + self.tobii_handler.gaze_data[10]) / 2
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
        self.tcp_disconnect()
        super().closeEvent(event)

    def openFeaturePlotWindow(self):
        # Pass participant_name to FeaturePlotWindow
        self.featurePlotWindow = FeaturePlotWindow(self.tobii_handler, self.participant_name) # self.tobii_handler 대신 쓰레드 인스턴스 넣기
        self.featurePlotWindow.show()
        


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
        self.dispersion_time_window = 30  # Define a time window in seconds

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
        time_window = 0.5

        # Remove old data points that are outside the time window
        while self.pupil_time_stamps and current_pupil_time - self.pupil_time_stamps[0] > time_window:
            self.pupil_time_stamps.pop(0)
            self.pupil_diameters.pop(0)

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
            return 0

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

        # Normalize the magnitudes to sum to 1
        if np.sum(magnitudes) == 0:
            return 0
        probabilities = magnitudes / np.sum(magnitudes)

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
            peak_velocity = max(self.saccade_velocities)
            return peak_velocity
        return 0

    def calculate_gaze_dispersion(self):
        # Update gaze points lists
        # Right eye data 추가됨
        self.gaze_points_x.append((self.__gaze_data[0] + self.__gaze_data[9]) / 2)
        self.gaze_points_y.append((self.__gaze_data[1] + self.__gaze_data[10]) / 2)

        while self.gaze_points_x and self.gaze_timestamps[-1] - self.gaze_points_x[0] > self.dispersion_time_window:
            self.gaze_points_x.pop(0)
            self.gaze_points_y.pop(0)
        
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


##########################################################################
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    initialWin = InitialWindow()
    initialWin.show()
    sys.exit(app.exec_())