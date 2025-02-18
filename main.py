# C:\github\cx330_radar\main.py

from real_time_process import UdpListener, DataProcessor
from radar_config import SerialConfig, DCA1000Config
from recorders import RecordingSession

from queue import Queue
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtGui, QtCore
import time, torch, sys, os, numpy as np
from serial.tools import list_ports
import iwr6843_tlv.detected_points as readpoint
import globalvar as gl

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import matplotlib.pyplot as plt
from colortrans import pg_get_cmap

# 尝试引入 OpenCV（相机）
try:
    import cv2
    CV2_OK = True
except Exception:
    CV2_OK = False

from UI_interface import Ui_MainWindow, Qt_pet

# ================== 参数区（与 cfg 对齐的显示尺度） ==================
FRAME_PERIOD_S = 0.1        # 100 ms 一帧
RANGE_RES_M    = 0.0488     # ~4.88 cm
VEL_RES_MPS    = 0.0806     # ~0.08 m/s
ANG_RES_DEG    = 2.0        # 角度栅格间隔（与 DSP.py 设置一致）
ANG_FOV_DEG    = (-90.0, 90.0)

# ================== 全局句柄/状态 ==================
ENABLE_CAMERA    = True
CAMERA_PROBE_MAX = 6

datasetfile = 'dataset'
datasetsencefile = ' '
gesturedict = {
    '0': 'backward', '1': 'dbclick', '2': 'down', '3': 'front',
    '4': 'Left', '5': 'Right', '6': 'up', '7': 'NO'
}
_flagdisplay = False

# 雷达与绘图对象
img_rdi = img_rai = img_rti = img_dti = img_rei = None
plot_rdi = plot_rai = plot_rti = plot_dti = plot_rei = None
view_gesture = None
logtxt = None

# 相机
_cam = None
_cam_timer = None
_cam_index = None

# 录制会话（raw.bin + gui.mp4）
_session = None
_gui_rec_timer = None

# 其他全局
MainWindow = None
radar_ctrl = None
processor = None

# 与处理链保持一致（在 __main__ 里真正赋值）
NUM_TX = 3
NUM_RX = 4
NUM_CHIRPS = 64
NUM_ADC_SAMPLES = 64
radar_config = None
frame_length = None


# ================ 工具函数 ==================
def printlog(string, fontcolor):
    """把信息打印到右下角 printlog 文本框。"""
    global logtxt
    if logtxt is None:
        return
    logtxt.moveCursor(QtGui.QTextCursor.End)
    gettime = time.strftime("%H:%M:%S", time.localtime())
    logtxt.append(f"<font color={fontcolor}>{gettime}-->{string}</font>")


def show_with_scale(img_item, arr, x_scale, y_scale, levels=None):
    """
    把 numpy 数组显示到 ImageItem，并用仿射把像素缩放到物理坐标尺度。
    * 不处理平移（平移在外面用 img_item.setPos 来做）
    """
    if levels is None:
        img_item.setImage(arr, autoLevels=True)
    else:
        img_item.setImage(arr, levels=levels, autoLevels=False)
    tr = QtGui.QTransform()
    tr.scale(x_scale, y_scale)
    img_item.setTransform(tr)


def _init_plot_with_axes(target_gw: pg.GraphicsView, x_label: str, x_unit: str,
                         y_label: str, y_unit: str):
    """在给定 GraphicsView 里放一个 PlotItem（带坐标轴）并返回 (plot, img)。"""
    plot = pg.PlotItem()
    plot.setMenuEnabled(False)
    plot.setMouseEnabled(x=False, y=False)
    plot.hideButtons()
    plot.setLabel('bottom', x_label, units=x_unit)
    plot.setLabel('left', y_label, units=y_unit)
    plot.getAxis('bottom').setStyle(tickTextOffset=2)
    plot.getAxis('left').setStyle(tickTextOffset=2)
    img = pg.ImageItem(border=None)
    plot.addItem(img)
    target_gw.setCentralItem(plot)
    return plot, img


def _probe_camera_index(max_idx=6):
    if not CV2_OK:
        return None
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            return i
    return None


def _camera_start(label_widget: QtWidgets.QLabel):
    """把相机帧嵌入到 GUI 的 label_widget（Realtime Camera 区域）。"""
    global _cam, _cam_timer, _cam_index
    if not ENABLE_CAMERA or not CV2_OK:
        printlog("OpenCV 不可用，跳过相机。", "red")
        return
    idx = _probe_camera_index(CAMERA_PROBE_MAX)
    if idx is None:
        printlog("未检测到可用摄像头。", "red")
        return
    _cam_index = idx
    _cam = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not _cam.isOpened():
        printlog("摄像头打开失败。", "red")
        return

    _cam_timer = QtCore.QTimer()
    _cam_timer.timeout.connect(lambda: _camera_tick(label_widget))
    _cam_timer.start(33)  # ~30 fps
    printlog(f"Camera embedded on index {_cam_index}", "green")


def _camera_tick(label_widget: QtWidgets.QLabel):
    if _cam is None:
        return
    ok, frame = _cam.read()
    if not ok:
        return
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = frame.shape
    qimg = QtGui.QImage(frame.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
    pix = QtGui.QPixmap.fromImage(qimg)
    label_widget.setPixmap(
        pix.scaled(
            label_widget.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
    )


def _capture_gui_frame():
    """把当前主窗口截图转成 BGR ndarray，写入 GuiRecorder。"""
    global _session, MainWindow
    if _session is None or _session.gui is None:
        return
    if MainWindow is None:
        return

    try:
        pixmap = MainWindow.grab()
        qimg = pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB888)
        w = qimg.width()
        h = qimg.height()
        bytes_per_line = qimg.bytesPerLine()
        ptr = qimg.bits()
        ptr.setsize(h * bytes_per_line)
        arr = np.frombuffer(ptr, np.uint8).reshape(h, bytes_per_line // 3, 3)
        frame_rgb = arr[:, :w, :]
        frame_bgr = frame_rgb[:, :, ::-1].copy()
        _session.gui.write_frame(frame_bgr)
    except Exception as e:
        # 录屏失败不致命，只在终端提示一次即可
        print("GUI capture error:", e)


# ================== 推理/数据相关（基本保持原逻辑） ==================
def loadmodel():
    global model
    if (modelfile.currentText() != '--select--'
            and modelfile.currentText() != ''):
        model_info = torch.load(
            modelfile.currentText(),
            map_location='cpu'
        )
        model = []
        model.load_state_dict(model_info['state_dict'])
        printlog('加载' + modelfile.currentText() + '模型成功!', fontcolor='blue')
    else:
        printlog("请加载模型!", fontcolor='red')


def cleartjpg():
    view_gesture.setPixmap(QtGui.QPixmap("gesture_icons/7.jpg"))


def Judge_gesture(a, b, c, d, e):
    global _flagdisplay
    fanhui = 7   # 占位
    view_gesture.setPixmap(QtGui.QPixmap(f"gesture_icons/{fanhui}.jpg"))
    QtCore.QTimer.singleShot(2000, cleartjpg)
    _flagdisplay = True
    printlog("输出:" + gesturedict[str(fanhui)], fontcolor='blue')
    return gesturedict[str(fanhui)]


def getradarparameters():
    if (radarparameters.currentIndex() > -1 and
            radarparameters.currentText() != '--select--'):
        radarparameters.setToolTip(radarparameters.currentText())
        configParameters = readpoint.IWR6843AOP_TLV()._initialize(
            config_file=radarparameters.currentText()
        )
        rangeResolutionlabel.setText(
            str(configParameters["rangeResolutionMeters"]) + 'cm'
        )
        dopplerResolutionlabel.setText(
            str(configParameters["dopplerResolutionMps"]) + 'm/s'
        )
        maxRangelabel.setText(str(configParameters["maxRange"]) + 'm')
        maxVelocitylabel.setText(str(configParameters["maxVelocity"]) + 'm/s')


# ================== 实时绘图 ==================
def update_figure():
    """
    从队列取出最新数据并更新 5 个视图：
      - RTI: Range–Time
      - DTI: Doppler–Time
      - RDI: Range–Doppler
      - REI: Range–Elevation
      - RAI: Range–Azimuth

    修正点：
      * DTI / RDI 在速度轴上对称显示 [-vmax, +vmax]
      * RAI / REI 在角度轴上完整显示 [-90°, +90°]，不再只看到正半轴
    """
    # 若队列还没准备好，就稍后再试
    if (RTIData.empty() or RDIData.empty() or DTIData.empty()
            or RAIData.empty() or REIData.empty()):
        QtCore.QTimer.singleShot(10, update_figure)
        return

    # 读取队列
    rti_buf = RTIData.get()
    rdi_buf = RDIData.get()
    dti_buf = DTIData.get()
    rai_buf = RAIData.get()
    rei_buf = REIData.get()

    # ---------- 图像显示（像素 -> 物理） ----------
    # RTI: x=Time(s), y=Range(m)
    rti_slice = rti_buf.sum(2)[0:1024:16, :]   # [time_idx, range_bin]
    show_with_scale(
        img_rti,
        rti_slice,
        FRAME_PERIOD_S,          # x: time
        RANGE_RES_M,             # y: range
        levels=[0, 1e4]
    )

    # RDI: x=Range(m), y=Velocity(m/s)
    # rdi_buf: [frames, range, doppler, virt_ant]
    rdi_frame = rdi_buf.sum(0)[:, :, 0]        # [range, doppler]
    rdi_for_plot = rdi_frame.T                # [doppler, range]
    show_with_scale(
        img_rdi,
        rdi_for_plot,
        RANGE_RES_M,             # x: range
        VEL_RES_MPS,             # y: velocity
        levels=[2e4, 4e5]
    )

    # DTI: x=Time(s), y=Velocity(m/s)
    show_with_scale(
        img_dti,
        dti_buf,
        FRAME_PERIOD_S,          # x: time
        VEL_RES_MPS,             # y: velocity
        levels=[0, 1000]
    )

    # RAI: x=Range(m), y=Angle(deg)
    # 取最近几帧的累加以减小抖动
    rai_img = rai_buf[4:12, :, :].sum(0)
    show_with_scale(
        img_rai,
        rai_img,
        RANGE_RES_M,
        ANG_RES_DEG,
        levels=[0, 8]
    )

    # REI: x=Range(m), y=Angle(deg)
    rei_img = rei_buf[4:12, :, :].sum(0).T
    show_with_scale(
        img_rei,
        rei_img,
        RANGE_RES_M,
        ANG_RES_DEG,
        levels=[0, 8]
    )

    # ---------- 坐标范围 + 速度/角度轴对齐 ----------

    # 距离上限
    max_range_m = NUM_ADC_SAMPLES * RANGE_RES_M

    # 速度总跨度  = NUM_CHIRPS * VEL_RES_MPS
    vel_span = NUM_CHIRPS * VEL_RES_MPS
    vmax = vel_span / 2.0        # 速度范围 [-vmax, +vmax]

    # 角度轴固定 FoV
    plot_rai.setRange(xRange=(0.0, max_range_m), yRange=ANG_FOV_DEG)
    plot_rei.setRange(xRange=(0.0, max_range_m), yRange=ANG_FOV_DEG)

    # RTI
    t_rti = rti_slice.shape[0] * FRAME_PERIOD_S
    plot_rti.setRange(
        xRange=(0.0, max(0.1, t_rti)),
        yRange=(0.0, max_range_m)
    )

    # DTI（y 轴速度，对称）+ 把图像中心移到 0 m/s
    t_dti = dti_buf.shape[0] * FRAME_PERIOD_S
    plot_dti.setRange(
        xRange=(0.0, max(0.1, t_dti)),
        yRange=(-vmax, vmax)
    )
    img_dti.setPos(0, -vmax)     # 图像下边界对齐 -vmax

    # RDI（y 轴速度，对称）+ 把图像中心移到 0 m/s
    plot_rdi.setRange(
        xRange=(0.0, max_range_m),
        yRange=(-vmax, vmax)
    )
    img_rdi.setPos(0, -vmax)

    # RAI / REI：角度从 -90 到 +90，图像原点需要平移到 -90°
    img_rai.setPos(0, ANG_FOV_DEG[0])   # y=-90 对应数组第 0 行
    img_rei.setPos(0, ANG_FOV_DEG[0])

    # ---------- 手势逻辑 (原逻辑保留) ----------
    if gl.get_value('usr_gesture'):
        RT_feature = RTIData.get().sum(2)[0:1024:16, :]
        DT_feature = DTIData.get()
        RDT_feature = RDIData.get()[:, :, :, 0]
        ART_feature = RAIData.get()
        ERT_feature = REIData.get()
        if Recognizebtn.isChecked():
            time_start = time.time()
            result = Judge_gesture(
                RT_feature, DT_feature, RDT_feature, ART_feature, ERT_feature
            )
            time_end = time.time()
            printlog(
                '识别时间:' + str(time_end - time_start) +
                's, 识别结果:' + str(result),
                fontcolor='blue'
            )
        gl.set_value('usr_gesture', False)

    QtCore.QTimer.singleShot(1, update_figure)


# ================== UI 绑定 ==================
def updatacomstatus(cbox):
    cbox.clear()
    for p in list(list_ports.comports()):
        cbox.addItem(str(p.device))


def setserialport(cbox, com):
    global CLIport_name, Dataport_name
    if cbox.currentIndex() > -1:
        port = cbox.currentText()
        if com == "CLI":
            CLIport_name = port
        else:
            Dataport_name = port


def sendconfigfunc():
    if len(CLIport_name) != 0 and radarparameters.currentText() != '--select--':
        openradar(radarparameters.currentText(), CLIport_name)
        printlog('发送成功', 'green')
    else:
        printlog('发送失败', 'red')


def setintervaltime():
    gl.set_value('timer_2s', True)
    QtCore.QTimer.singleShot(2000, setintervaltime)


def setcolor():
    if (color_.currentText() != '--select--'
            and color_.currentText() != ''):
        if color_.currentText() == 'customize':
            pgColormap = pg_get_cmap(color_.currentText())
        else:
            cmap = plt.cm.get_cmap(color_.currentText())
            pgColormap = pg_get_cmap(cmap)
        lut = pgColormap.getLookupTable(0.0, 1.0, 256)
        for img in (img_rdi, img_rai, img_rti, img_dti, img_rei):
            img.setLookupTable(lut)


def savedatasetsencefile():
    pass  # 保留你的原逻辑，暂时空实现


# ================== 雷达启动 & 录制 ==================
def _start_session_and_threads(config_file):
    """
    1) 创建 / 重置 RecordingSession（raw.bin + gui.mp4）
    2) 重新配置雷达并启动 DataProcessor
    """
    global radar_ctrl, processor, _session, _gui_rec_timer

    # 若已有旧会话，先关闭
    if _session is not None:
        try:
            _session.close()
        except Exception:
            pass
        _session = None

    # 创建新会话目录 recordings/YYYYMMDD_HHMMSS_xxx
    _session = RecordingSession(base_dir="recordings")

    # raw.bin 录制配置
    _session.start_raw(
        adc_samples=NUM_ADC_SAMPLES,
        chirps=NUM_CHIRPS,
        tx=NUM_TX,
        rx=NUM_RX,
        frame_length_shorts=frame_length,
        config_file=config_file
    )

    # GUI 录屏
    try:
        _session.start_gui(fps=20)
        if _gui_rec_timer is None:
            _gui_rec_timer = QtCore.QTimer()
            _gui_rec_timer.timeout.connect(_capture_gui_frame)
        _gui_rec_timer.start(50)   # ~20 fps
    except Exception as e:
        printlog("GUI 录屏启动失败: " + repr(e), "red")

    # 串口控制雷达
    radar_ctrl = SerialConfig(
        name='ConnectRadar',
        CLIPort=CLIport_name,
        BaudRate=115200
    )
    radar_ctrl.StopRadar()
    radar_ctrl.SendConfig(config_file)

    # DSP 处理线程（把 raw short -> 队列）
    processor = DataProcessor(
        'Processor',
        radar_config,
        BinData,
        RTIData,
        DTIData,
        RDIData,
        RAIData,
        REIData,
        raw_recorder=_session.raw
    )
    processor.start()
    processor.join(timeout=1)

    radar_ctrl.StartRadar()
    update_figure()


def openradar(config, com):
    if not com:
        printlog("No CLI port", 'red')
        return
    try:
        _start_session_and_threads(config)
    except Exception as e:
        printlog("openradar failed: " + repr(e), 'red')
        try:
            if processor:
                processor.stop()
        except Exception:
            pass


# ================== 应用程序入口 ==================
def application():
    global color_, radarparameters, maxVelocitylabel, maxRangelabel
    global dopplerResolutionlabel, rangeResolutionlabel, logtxt
    global Recognizebtn, view_gesture, modelfile
    global img_rdi, img_rai, img_rti, img_rei, img_dti
    global plot_rdi, plot_rai, plot_rti, plot_rei, plot_dti
    global MainWindow

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

    # 右侧控件
    color_ = ui.comboBox
    modelfile = ui.comboBox_2
    radarparameters = ui.comboBox_7
    Cliportbox = ui.comboBox_8
    Recognizebtn = ui.pushButton_15
    sendcfgbtn = ui.pushButton_11
    exitbtn = ui.pushButton_12
    logtxt = ui.textEdit
    rangeResolutionlabel = ui.label_14
    dopplerResolutionlabel = ui.label_35
    maxRangelabel = ui.label_16
    maxVelocitylabel = ui.label_37

    # 顶部：RTI, DTI, RDI
    plot_rti, img_rti = _init_plot_with_axes(
        ui.graphicsView, "Time", "s", "Range", "m"
    )
    plot_dti, img_dti = _init_plot_with_axes(
        ui.graphicsView_2, "Time", "s", "Velocity", "m/s"
    )
    plot_rdi, img_rdi = _init_plot_with_axes(
        ui.graphicsView_6, "Range", "m", "Velocity", "m/s"
    )
    # 底部：REI, RAI
    plot_rei, img_rei = _init_plot_with_axes(
        ui.graphicsView_3, "Range", "m", "Angle", "deg"
    )
    plot_rai, img_rai = _init_plot_with_axes(
        ui.graphicsView_4, "Range", "m", "Angle", "deg"
    )

    # colormap 默认
    lut = pg_get_cmap('customize').getLookupTable(0.0, 1.0, 256)
    for img in (img_rdi, img_rai, img_rti, img_dti, img_rei):
        img.setLookupTable(lut)

    # 相机嵌入 —— Realtime Camera 区域（graphicsView_5）
    global _cam
    view_gesture = ui.graphicsView_5
    view_gesture.setPixmap(QtGui.QPixmap("gesture_icons/7.jpg"))
    if ENABLE_CAMERA:
        _camera_start(view_gesture)

    # 事件绑定
    Cliportbox.arrowClicked.connect(lambda: updatacomstatus(Cliportbox))
    Cliportbox.currentIndexChanged.connect(
        lambda: setserialport(Cliportbox, com='CLI')
    )
    color_.currentIndexChanged.connect(setcolor)
    modelfile.currentIndexChanged.connect(loadmodel)
    radarparameters.currentIndexChanged.connect(getradarparameters)
    sendcfgbtn.clicked.connect(sendconfigfunc)
    Recognizebtn.clicked.connect(setintervaltime)
    exitbtn.clicked.connect(app.instance().exit)

    app.instance().exec_()

    # 退出清理
    try:
        if _cam is not None:
            _cam.release()
    except Exception:
        pass
    try:
        if _gui_rec_timer is not None:
            _gui_rec_timer.stop()
    except Exception:
        pass
    try:
        if _session is not None:
            _session.close()
    except Exception:
        pass
    try:
        if radar_ctrl and radar_ctrl.CLIPort and radar_ctrl.CLIPort.isOpen():
            radar_ctrl.StopRadar()
    except Exception:
        pass


if __name__ == '__main__':
    gl._init()
    gl.set_value('usr_gesture', False)

    # 上层数据队列
    BinData = Queue()
    RTIData = Queue()
    DTIData = Queue()
    RDIData = Queue()
    RAIData = Queue()
    REIData = Queue()

    # 与处理链保持一致（用于范围计算）
    NUM_TX = 3
    NUM_RX = 4
    NUM_CHIRPS = 64
    NUM_ADC_SAMPLES = 64
    radar_config = [NUM_ADC_SAMPLES, NUM_CHIRPS, NUM_TX, NUM_RX]
    frame_length = NUM_ADC_SAMPLES * NUM_CHIRPS * NUM_TX * NUM_RX * 2

    # DCA1000
    dca1000_cfg = DCA1000Config(
        'DCA1000Config',
        config_address=('192.168.33.30', 4096),
        FPGA_address_cfg=('192.168.33.180', 4096)
    )

    # 采集线程
    collector = UdpListener('Listener', BinData, frame_length)
    collector.start()

    # 设备端口占位
    CLIport_name = ""
    Dataport_name = ""

    # 启动 GUI
    application()

    try:
        dca1000_cfg.DCA1000_close()
    except Exception:
        pass

    collector.join(timeout=1)
    print("Program close")
    sys.exit()
