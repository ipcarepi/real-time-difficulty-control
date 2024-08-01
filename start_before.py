import sys
import csv
import cv2
import time
import socket
import threading
import tobii_research as tr
import pyqtgraph as pg
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QPushButton, QDialog, QVBoxLayout, QLabel, QDialogButtonBox

from sensor_streamers.SensorStreamer import SensorStreamer
from visualizers.LinePlotVisualizer import LinePlotVisualizer
from visualizers.HeatmapVisualizer import HeatmapVisualizer

import socket
import numpy as np
import time
from collections import OrderedDict
import traceback
import pylsl

from utils.print_utils import *

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
    sensor_data_collect = QtCore.pyqtSignal(np.ndarray)
    
    # 여기서만 tobii_handler 부르기
    def __init__(self):
        super().__init__()
        
        self.channel = 18
        self.point = 100
        self.data = np.zeros((self.point, self.channel))
        
        self.tobii_handler = Tobii_handler()

    def run(self):
        try:
            while True:
                self.update_data()
                time.sleep(0.05)
            
        except Exception as e:
            print(f'Error in worker: {e}')
            
    def update_data(self):
        new_data = self.tobii_handler.gaze_data

        # Shift the data matrix up by one row
        self.data = np.roll(self.data, -1, axis=0)

        # Update the last row with new gaze data
        self.data[-1, :] = new_data
        
        self.sensor_data_collect.emit(self.data)

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
        
        # Start E4 button
        self.start_E4_button = QPushButton("Start E4", self)
        self.start_E4_button.clicked.connect(self.start_E4)
        self.layout.addWidget(self.start_E4_button)
        
        self.graph_widget_E4 = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graph_widget_E4)
        self.graph_widget_E4.hide()

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

        self.setGeometry(100, 100, 800, 900)

    def start(self):
        
        self.start_button.hide()
        self.graph_widget.show()

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
        self.data_e41 = np.zeros((self.num_points, self.e4_channels))
        self.data_e42 = np.zeros((self.num_points, self.e4_channels))
        
        self.plots = []
        self.curves = []
        
        self.plots_e41 = []
        self.curves_e41 = []
        
        self.plots_e42 = []
        self.curves_e42 = []

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
        
        for i in range(self.e4_channels):
            plot_e4 = self.graph_widget_E4.addPlot(row = 6, col = i, title = f'E4 1 {i + 1}')
            pen = pg.mkPen(color=(i, self.e4_channels), width=2)
            curve_e4 = plot_e4.plot(self.data_e41[:, i], pen=pen)
            plot_e4.showGrid(x=True, y=True)
            self.plots_e41.append(plot_e4)
            self.curves_e41.append(curve_e4)
            
            plot_e4 = self.graph_widget_E4.addPlot(row = 7, col = i, title = f'E4 2 {i + 1}')
            pen = pg.mkPen(color=(i, self.e4_channels), width=2)
            curve_e4 = plot_e4.plot(self.data_e42[:, i], pen=pen)
            plot_e4.showGrid(x=True, y=True)
            self.plots_e42.append(plot_e4)
            self.curves_e42.append(curve_e4)
        

    def update(self, data):
        # Update the plot curves with new data
        for i, curve in enumerate(self.curves):
           curve.setData(data[:, i])

        # Refresh the graph widget
        self.graph_widget.update()

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
        try:
            self.E4_streamer.stop()
        except:
            pass
        self.tcp_disconnect()
        super().closeEvent(event)
        
    def start_E4(self):
        
        self.start_E4_button.hide()
        self.graph_widget_E4.show()
        
        # Connect to the device(s).
        self.E4_streamer = E4Streamer(print_status=True, print_debug=False)
        self.E4_streamer.connect()
        self.E4_streamer.run()
        self.start_time = time.time()
        print("Streamer Running Start")

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_E4)
        self.timer.start(2000)  # 2초마다 업데이트
        
    def update_E4(self):
        msg = []
        try:
            for device_name in self.E4_streamer.get_device_names():
                stream_names = self.E4_streamer.get_stream_names(device_name=device_name)
                for stream_name in stream_names:
                    num_timesteps = self.E4_streamer.get_num_timesteps(device_name, stream_name)
                    msg.append((num_timesteps) / (time.time() - self.start_time))
        except:
            pass
        
        self.data_e41 = np.roll(self.data_e41, -1, axis = 0)
        
        self.data_e41[-1, :] = msg
        
        for i, curve in enumerate(self.curves_e41):
           curve.setData(self.data_e41[:, i])
           

        # Refresh the graph widget
        self.graph_widget_E4.update()

##########################################################################
class Tobii_handler():
    def __init__(self):
        self.__gaze_data = [0] * 18
        self.lock = threading.Lock()

        self.eye_tracker = None
        self.is_ready = False
        self.safe_init_eye_tracker()

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
        self._device_id1 = '6337CD'
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

            # print("This is")
            #
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