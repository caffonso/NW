"""
Neurowood - interface de calibração/inspeção com PySide6.

Executar a partir da raiz do projeto:
    python gui.py

Dependências adicionais:
    pip install PySide6 PyYAML

Para câmera Basler / emulador Basler, o projeto também precisa de pypylon.

Esta interface NÃO substitui main.py. Ela é apenas uma segunda entrada para o
mesmo core.pipeline.NeuroWoodPipeline.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover - mensagem amigável em runtime
    raise SystemExit("PyYAML não instalado. Execute: pip install PyYAML") from exc

try:
    from PySide6.QtCore import QObject, QThread, Qt, Signal
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PySide6 não instalado. Execute: pip install PySide6") from exc

from core.pipeline import NeuroWoodPipeline
from hardware.production_camera import ProductionSimulationCamera
from storage.production_report import ProductionReport, timestamped_report_path
from vision.defects import detect_texture_defects
from vision.reconstruction import reconstruct_board


APP_TITLE = "Neurowood - Calibration Tool"
DEFAULT_CONFIG_NAMES = ("config.yaml", "config_vision_example.yaml")
EMU_TARGET_SIZE = (1280, 1024)


# ---------------------------------------------------------------------------
# Câmera Basler: mesma estratégia validada no notebook
# ---------------------------------------------------------------------------


class BaslerTriggeredSource:
    """Adapter mínimo com a interface capture() esperada pelo pipeline."""

    def __init__(self, camera: Any, pylon_module: Any, timeout_ms: int = 3000):
        self.camera = camera
        self.pylon = pylon_module
        self.timeout_ms = timeout_ms

    def start(self) -> None:
        if self.camera.IsGrabbing():
            return

        try:
            self.camera.TriggerSelector.Value = "FrameStart"
        except Exception:
            pass

        self.camera.TriggerMode.Value = "On"
        self.camera.TriggerSource.Value = "Software"
        self.camera.StartGrabbing(
            self.pylon.GrabStrategy_OneByOne,
            self.pylon.GrabLoop_ProvidedByUser,
        )

    def capture(self) -> np.ndarray:
        self.start()

        ready = self.camera.WaitForFrameTriggerReady(
            self.timeout_ms,
            self.pylon.TimeoutHandling_ThrowException,
        )
        if not ready:
            raise TimeoutError("Câmera não ficou pronta para o trigger.")

        self.camera.ExecuteSoftwareTrigger()
        result = self.camera.RetrieveResult(
            self.timeout_ms,
            self.pylon.TimeoutHandling_ThrowException,
        )

        try:
            if not result.GrabSucceeded():
                raise RuntimeError(
                    f"Grab failed: {result.ErrorCode} - {result.ErrorDescription}"
                )
            return result.Array.copy()
        finally:
            result.Release()

    def stop(self) -> None:
        try:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
        finally:
            try:
                if self.camera.IsOpen():
                    self.camera.Close()
            except Exception:
                pass


def _device_text(dev: Any) -> str:
    values = []
    for getter in ("GetModelName", "GetFriendlyName", "GetDeviceClass"):
        try:
            values.append(getattr(dev, getter)())
        except Exception:
            pass
    return " ".join(map(str, values)).lower()


def _prepare_emulator_image(input_path: str) -> Path:
    """Central-crop + resize para 1280x1024, equivalente ao preparo do notebook."""
    src = Path(input_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Não foi possível abrir a imagem: {src}")

    target_w, target_h = EMU_TARGET_SIZE
    h, w = image.shape[:2]

    src_ratio = w / h
    target_ratio = target_w / target_h

    if src_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x0 = max(0, (w - new_w) // 2)
        image = image[:, x0 : x0 + new_w]
    else:
        new_h = int(w / target_ratio)
        y0 = max(0, (h - new_h) // 2)
        image = image[y0 : y0 + new_h, :]

    prepared = cv2.resize(image, EMU_TARGET_SIZE, interpolation=cv2.INTER_AREA)

    out_dir = Path.cwd() / ".neurowood_gui"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "basler_emulator_input.png"

    if not cv2.imwrite(str(out_path), prepared):
        raise RuntimeError(f"Não foi possível salvar a imagem preparada em {out_path}")

    return out_path.resolve()


def open_basler_source(mode: str, emulator_image: str | None = None) -> BaslerTriggeredSource:
    """Abre uma Basler física ou o emulador Pylon usando software trigger."""
    use_emulator = mode == "Basler Emulator"

    if use_emulator:
        if not emulator_image:
            raise ValueError("Selecione uma imagem para o emulador Basler.")
        # Deve ser definido antes da primeira enumeração/importação efetiva do Pylon.
        os.environ["PYLON_CAMEMU"] = "1"

    try:
        from pypylon import pylon
    except ImportError as exc:
        raise RuntimeError(
            "pypylon não está instalado. Instale/configure o Basler pylon + pypylon."
        ) from exc

    factory = pylon.TlFactory.GetInstance()
    devices = list(factory.EnumerateDevices())
    if not devices:
        raise RuntimeError("Nenhuma câmera Basler encontrada pelo pylon.")

    emu_devices = [d for d in devices if "emu" in _device_text(d) or "emulation" in _device_text(d)]

    if use_emulator:
        if not emu_devices:
            raise RuntimeError(
                "O pylon não retornou uma câmera emulada. Verifique Camera Emulation e reinicie o gui.py."
            )
        selected = emu_devices[0]
    else:
        physical = [d for d in devices if d not in emu_devices]
        selected = physical[0] if physical else devices[0]

    camera = pylon.InstantCamera(factory.CreateDevice(selected))
    camera.Open()

    # Preferência do notebook: BGR8. Se indisponível, Mono8.
    pixel_format = None
    for value in ("BGR8", "Mono8"):
        try:
            if camera.PixelFormat.TrySetValue(value):
                pixel_format = value
                break
        except Exception:
            try:
                camera.PixelFormat.Value = value
                pixel_format = value
                break
            except Exception:
                pass

    if use_emulator:
        prepared_path = _prepare_emulator_image(emulator_image)

        if camera.IsGrabbing():
            camera.StopGrabbing()

        try:
            camera.TestImageSelector.Value = "Off"
        except Exception:
            pass

        camera.ImageFileMode.Value = "On"
        camera.ImageFilename.Value = str(prepared_path)

        try:
            camera.Width.Value = EMU_TARGET_SIZE[0]
            camera.Height.Value = EMU_TARGET_SIZE[1]
        except Exception:
            pass

    source = BaslerTriggeredSource(camera, pylon)
    source.start()
    logging.getLogger(__name__).info(
        "Câmera aberta: %s | PixelFormat=%s",
        camera.GetDeviceInfo().GetFriendlyName(),
        pixel_format,
    )
    return source


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


class PipelineWorker(QObject):
    finished = Signal(object, object)
    failed = Signal(str)

    def __init__(self, camera: Any, config: dict[str, Any]):
        super().__init__()
        self.camera = camera
        self.config = config

    def run(self) -> None:
        try:
            # GUI de calibração não envia resultado ao PLC.
            pipeline = NeuroWoodPipeline(self.camera, None, self.config)
            result = pipeline.run_once()
            self.finished.emit(pipeline, result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class ReprocessWorker(QObject):
    finished = Signal(object, object, object)
    failed = Signal(str)

    def __init__(self, piece_frames: list[dict[str, Any]], config: dict[str, Any]):
        super().__init__()
        self.piece_frames = piece_frames
        self.config = config

    def run(self) -> None:
        try:
            vision = self.config["vision"]
            results, summary = detect_texture_defects(
                self.piece_frames,
                min_area_frac=vision["min_area_frac"],
                min_width_frac=vision["min_width_frac"],
                rigor=vision["rigor"],
            )
            board_marked = reconstruct_board(results, use_marked=True)
            self.finished.emit(results, summary, board_marked)
        except Exception:
            self.failed.emit(traceback.format_exc())


class ProductionWorker(QObject):
    board_finished = Signal(object, object, object)
    finished = Signal(object, str)
    failed = Signal(str)

    def __init__(
        self,
        camera: ProductionSimulationCamera,
        config: dict[str, Any],
        n_boards: int,
        report_path: str,
    ):
        super().__init__()
        self.camera = camera
        self.config = config
        self.n_boards = int(n_boards)
        self.report_path = report_path
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            report = ProductionReport(self.report_path)

            for sequence in range(1, self.n_boards + 1):
                if self._stop_event.is_set():
                    break

                self.camera.begin_board()
                pipeline = NeuroWoodPipeline(self.camera, None, self.config)
                result = pipeline.run_once()

                row = report.add_board(
                    sequence=sequence,
                    pipeline=pipeline,
                    result=result,
                    camera=self.camera,
                )
                self.board_finished.emit(pipeline, result, row)

            self.finished.emit(report.summary(), str(report.path))
        except Exception:
            self.failed.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Logging -> GUI
# ---------------------------------------------------------------------------


class LogEmitter(QObject):
    message = Signal(str)


class QtLogHandler(logging.Handler):
    def __init__(self, emitter: LogEmitter):
        super().__init__()
        self.emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.emitter.message.emit(self.format(record))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Utilidades de imagem
# ---------------------------------------------------------------------------


def ndarray_to_pixmap(image: np.ndarray) -> QPixmap:
    if image is None:
        return QPixmap()

    arr = np.ascontiguousarray(image)

    if arr.ndim == 2:
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, arr.strides[0], QImage.Format_Grayscale8).copy()
        return QPixmap.fromImage(qimg)

    if arr.ndim == 3 and arr.shape[2] == 3:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        qimg = QImage(
            rgb.data,
            w,
            h,
            rgb.strides[0],
            QImage.Format_RGB888,
        ).copy()
        return QPixmap.fromImage(qimg)

    if arr.ndim == 3 and arr.shape[2] == 4:
        rgba = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
        rgba = np.ascontiguousarray(rgba)
        h, w, _ = rgba.shape
        qimg = QImage(
            rgba.data,
            w,
            h,
            rgba.strides[0],
            QImage.Format_RGBA8888,
        ).copy()
        return QPixmap.fromImage(qimg)

    raise ValueError(f"Formato de imagem não suportado: {arr.shape}")


class ImageLabel(QLabel):
    def __init__(self, text: str = ""):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(220, 130)
        self.setStyleSheet("QLabel { background: #161616; color: #aaa; border: 1px solid #444; }")
        self._original_pixmap: QPixmap | None = None

    def set_image(self, image: np.ndarray, max_height: int | None = None) -> None:
        pixmap = ndarray_to_pixmap(image)
        if max_height and not pixmap.isNull():
            pixmap = pixmap.scaledToHeight(max_height, Qt.SmoothTransformation)
        self._original_pixmap = pixmap
        self.setPixmap(pixmap)
        self.adjustSize()


# ---------------------------------------------------------------------------
# Janela principal
# ---------------------------------------------------------------------------


class NeurowoodWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1500, 900)

        self.camera_source: BaslerTriggeredSource | None = None
        self.last_pipeline: NeuroWoodPipeline | None = None
        self.last_piece_frames: list[dict[str, Any]] | None = None
        self.last_detection_results: list[dict[str, Any]] | None = None
        self.last_summary: dict[str, Any] | None = None

        self._thread: QThread | None = None
        self._worker: QObject | None = None

        self._build_ui()
        self._install_log_handler()
        self._load_default_config_if_present()
        self._update_camera_controls()
        self._update_buttons()

    # ----------------------------- UI ---------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(360)
        controls_scroll.setMaximumWidth(470)

        controls = QWidget()
        self.controls_layout = QVBoxLayout(controls)
        controls_scroll.setWidget(controls)
        splitter.addWidget(controls_scroll)

        self._build_logo()
        self._build_camera_group()
        self._build_production_group()
        self._build_capture_group()
        self._build_piece_group()
        self._build_defect_group()
        self._build_actions_group()
        self._build_result_group()
        self.controls_layout.addStretch(1)

        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)

        # Reconstrução
        board_tab = QWidget()
        board_layout = QVBoxLayout(board_tab)
        self.board_scroll = QScrollArea()
        self.board_scroll.setWidgetResizable(False)
        self.board_label = ImageLabel("A peça reconstruída aparecerá aqui.")
        self.board_label.setMinimumSize(850,160)#(700, 320)
        self.board_scroll.setWidget(self.board_label)
        board_layout.addWidget(self.board_scroll)
        self.tabs.addTab(board_tab, "Peça reconstruída")

        # Patches
        patches_tab = QWidget()
        patches_layout = QVBoxLayout(patches_tab)
        self.patches_scroll = QScrollArea()
        self.patches_scroll.setWidgetResizable(True)
        self.patches_container = QWidget()
        self.patches_grid = QGridLayout(self.patches_container)
        self.patches_scroll.setWidget(self.patches_container)
        patches_layout.addWidget(self.patches_scroll)
        self.tabs.addTab(patches_tab, "Patches")

        # Defeitos
        defects_tab = QWidget()
        defects_layout = QVBoxLayout(defects_tab)
        self.defect_table = QTableWidget(0, 5)
        self.defect_table.setHorizontalHeaderLabels(
            ["Patch", "Defeito", "Área (px²)", "Área (%)", "BBox (x,y,w,h)"]
        )
        header = self.defect_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        defects_layout.addWidget(self.defect_table)
        self.tabs.addTab(defects_tab, "Defeitos")

        # Logs
        logs_tab = QWidget()
        logs_layout = QVBoxLayout(logs_tab)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        logs_layout.addWidget(self.log_text)
        self.tabs.addTab(logs_tab, "Logs")

    def _build_logo(self) -> None:
        """Exibe o logo Neurowood no topo da barra lateral."""
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setMinimumHeight(145)
        self.logo_label.setMaximumHeight(175)

        base_dir = Path(__file__).resolve().parent
        logo_candidates = [
            base_dir / "assets" / "neurowood_logo.png",
            base_dir / "neurowood_logo.png",
        ]

        logo_path = next(
            (path for path in logo_candidates if path.exists()),
            None,
        )

        if logo_path is not None:
            pixmap = QPixmap(str(logo_path))
            if not pixmap.isNull():
                pixmap = pixmap.scaled(
                    250,
                    160,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.logo_label.setPixmap(pixmap)
            else:
                self.logo_label.setText("NEUROWOOD")
        else:
            self.logo_label.setText("NEUROWOOD")

        self.controls_layout.addWidget(self.logo_label)

    def _build_camera_group(self) -> None:
        group = QGroupBox("Câmera")
        layout = QFormLayout(group)

        self.camera_mode = QComboBox()
        self.camera_mode.addItems(["Basler Emulator", "Basler Física", "Produção simulada"])
        self.camera_mode.currentTextChanged.connect(self._update_camera_controls)
        layout.addRow("Modo", self.camera_mode)

        image_row = QWidget()
        image_layout = QHBoxLayout(image_row)
        image_layout.setContentsMargins(0, 0, 0, 0)
        self.emulator_image = QLineEdit()
        self.emulator_image.setPlaceholderText("Imagem real de madeira para emulação")
        self.browse_image_btn = QPushButton("...")
        self.browse_image_btn.setMaximumWidth(42)
        self.browse_image_btn.clicked.connect(self._browse_emulator_image)
        image_layout.addWidget(self.emulator_image)
        image_layout.addWidget(self.browse_image_btn)
        layout.addRow("Imagem", image_row)

        self.init_camera_btn = QPushButton("Inicializar câmera")
        self.init_camera_btn.clicked.connect(self._initialize_camera)
        layout.addRow(self.init_camera_btn)

        self.camera_status = QLabel("Não inicializada")
        self.camera_status.setWordWrap(True)
        layout.addRow("Status", self.camera_status)

        self.controls_layout.addWidget(group)


    def _build_production_group(self) -> None:
        group = QGroupBox("Produção simulada")
        layout = QFormLayout(group)

        dir_row = QWidget()
        dir_layout = QHBoxLayout(dir_row)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        self.production_dir = QLineEdit(str(Path.cwd() / ".neurowood_gui"))
        self.production_dir_btn = QPushButton("...")
        self.production_dir_btn.setMaximumWidth(42)
        self.production_dir_btn.clicked.connect(self._browse_production_dir)
        dir_layout.addWidget(self.production_dir)
        dir_layout.addWidget(self.production_dir_btn)
        layout.addRow("Patches", dir_row)

        self.production_boards = QSpinBox()
        self.production_boards.setRange(1, 100000)
        self.production_boards.setValue(20)
        layout.addRow("Nº de tábuas", self.production_boards)

        self.production_width_min = self._double_spin(20, 500, 90, 5, 1)
        self.production_width_max = self._double_spin(20, 500, 220, 5, 1)
        layout.addRow("Largura mín. (mm)", self.production_width_min)
        layout.addRow("Largura máx. (mm)", self.production_width_max)

        self.production_seed = QSpinBox()
        self.production_seed.setRange(-1, 2147483647)
        self.production_seed.setValue(-1)
        self.production_seed.setSpecialValueText("Aleatório")
        layout.addRow("Seed", self.production_seed)

        self.production_report = QLineEdit()
        self.production_report.setPlaceholderText(
            "Automático: reports/production_report_*.csv"
        )
        layout.addRow("Relatório CSV", self.production_report)

        self.production_summary_label = QLabel("Nenhuma simulação executada.")
        self.production_summary_label.setWordWrap(True)
        layout.addRow(self.production_summary_label)

        self.production_group = group
        self.controls_layout.addWidget(group)

    def _build_capture_group(self) -> None:
        group = QGroupBox("Captura / Esteira")
        layout = QFormLayout(group)

        self.belt_speed = self._double_spin(1, 10000, 600, 10, 1)
        self.frame_spacing = self._double_spin(1, 5000, 225, 5, 1)
        self.speed_factor = self._double_spin(0.1, 3.0, 1.0, 0.01, 3)
        self.max_frames = QSpinBox()
        self.max_frames.setRange(1, 100)
        self.max_frames.setValue(13)

        layout.addRow("Velocidade (mm/s)", self.belt_speed)
        layout.addRow("Passo captura (mm)", self.frame_spacing)
        layout.addRow("Correção velocidade", self.speed_factor)
        layout.addRow("Nº de frames", self.max_frames)

        self.controls_layout.addWidget(group)

    def _build_piece_group(self) -> None:
        group = QGroupBox("Detecção da peça")
        layout = QFormLayout(group)

        self.piece_contrast = self._double_spin(0, 255, 25, 1, 1)
        self.piece_height = self._double_spin(0.01, 1.0, 0.10, 0.01, 3)
        self.sample_step = QSpinBox()
        self.sample_step.setRange(1, 128)
        self.sample_step.setValue(8)

        layout.addRow("Contraste mínimo", self.piece_contrast)
        layout.addRow("Altura mínima (fração)", self.piece_height)
        layout.addRow("Sample step", self.sample_step)

        self.controls_layout.addWidget(group)

    def _build_defect_group(self) -> None:
        group = QGroupBox("Detecção de defeitos")
        layout = QFormLayout(group)

        self.rigor = self._double_spin(0.10, 10.0, 1.50, 0.05, 2)
        self.min_area_frac = self._double_spin(0.00001, 1.0, 0.003, 0.0005, 5)
        self.min_width_frac = self._double_spin(0.001, 1.0, 0.08, 0.005, 4)

        layout.addRow("Rigor", self.rigor)
        layout.addRow("Área mínima (fração)", self.min_area_frac)
        layout.addRow("Largura mínima (fração)", self.min_width_frac)

        hint = QLabel("Rigor menor = mais sensível; rigor maior = mais conservador.")
        hint.setWordWrap(True)
        layout.addRow(hint)

        self.controls_layout.addWidget(group)

    def _build_actions_group(self) -> None:
        group = QGroupBox("Execução")
        layout = QVBoxLayout(group)

        self.run_btn = QPushButton("PROCESSAR UMA PEÇA")
        self.run_btn.setMinimumHeight(45)
        self.run_btn.clicked.connect(self._run_pipeline)
        layout.addWidget(self.run_btn)

        self.production_run_btn = QPushButton("SIMULAR PRODUÇÃO")
        self.production_run_btn.setMinimumHeight(40)
        self.production_run_btn.clicked.connect(self._start_production_simulation)
        layout.addWidget(self.production_run_btn)

        self.production_stop_btn = QPushButton("Parar simulação")
        self.production_stop_btn.clicked.connect(self._stop_production_simulation)
        self.production_stop_btn.setEnabled(False)
        layout.addWidget(self.production_stop_btn)

        self.reprocess_btn = QPushButton("Reprocessar captura atual")
        self.reprocess_btn.clicked.connect(self._reprocess_current_capture)
        layout.addWidget(self.reprocess_btn)

        config_row = QHBoxLayout()
        self.load_config_btn = QPushButton("Carregar YAML")
        self.save_config_btn = QPushButton("Salvar YAML")
        self.load_config_btn.clicked.connect(self._load_config_dialog)
        self.save_config_btn.clicked.connect(self._save_config_dialog)
        config_row.addWidget(self.load_config_btn)
        config_row.addWidget(self.save_config_btn)
        layout.addLayout(config_row)

        self.close_camera_btn = QPushButton("Fechar câmera")
        self.close_camera_btn.clicked.connect(self._close_camera)
        layout.addWidget(self.close_camera_btn)

        self.controls_layout.addWidget(group)

    def _build_result_group(self) -> None:
        group = QGroupBox("Resultado")
        layout = QFormLayout(group)

        self.classification_label = QLabel("—")
        self.classification_label.setStyleSheet("font-size: 28px; font-weight: bold;")
        self.board_id_label = QLabel("—")
        self.width_label = QLabel("—")
        self.defects_label = QLabel("0")
        self.processing_label = QLabel("—")
        self.summary_label = QLabel("Sem processamento.")
        self.summary_label.setWordWrap(True)

        layout.addRow("Classe", self.classification_label)
        layout.addRow("Board ID", self.board_id_label)
        layout.addRow("Largura máxima", self.width_label)
        layout.addRow("Defeitos", self.defects_label)
        layout.addRow("Tempo", self.processing_label)
        layout.addRow(self.summary_label)

        self.controls_layout.addWidget(group)

    @staticmethod
    def _double_spin(minimum: float, maximum: float, value: float, step: float, decimals: int) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSingleStep(step)
        widget.setValue(value)
        return widget

    # -------------------------- Config --------------------------------

    def current_config(self) -> dict[str, Any]:
        return {
            "conveyor": {
                "nominal_speed_mm_s": float(self.belt_speed.value()),
                "speed_factor": float(self.speed_factor.value()),
            },
            "capture": {
                "frame_spacing_mm": float(self.frame_spacing.value()),
                "max_frames": int(self.max_frames.value()),
            },
            "vision": {
                "piece_min_contrast": float(self.piece_contrast.value()),
                "piece_min_height_frac": float(self.piece_height.value()),
                "piece_sample_step": int(self.sample_step.value()),
                "rigor": float(self.rigor.value()),
                "min_area_frac": float(self.min_area_frac.value()),
                "min_width_frac": float(self.min_width_frac.value()),
            },
        }

    def apply_config(self, config: dict[str, Any]) -> None:
        conveyor = config.get("conveyor", {})
        capture = config.get("capture", {})
        vision = config.get("vision", {})

        self.belt_speed.setValue(float(conveyor.get("nominal_speed_mm_s", 600)))
        self.speed_factor.setValue(float(conveyor.get("speed_factor", conveyor.get("speed_correction", 1.0))))
        self.frame_spacing.setValue(float(capture.get("frame_spacing_mm", capture.get("capture_step_mm", 225))))
        self.max_frames.setValue(int(capture.get("max_frames", capture.get("n_frames", 13))))

        self.piece_contrast.setValue(float(vision.get("piece_min_contrast", 25)))
        self.piece_height.setValue(float(vision.get("piece_min_height_frac", 0.10)))
        self.sample_step.setValue(int(vision.get("piece_sample_step", 8)))
        self.rigor.setValue(float(vision.get("rigor", 1.5)))
        self.min_area_frac.setValue(float(vision.get("min_area_frac", 0.003)))
        self.min_width_frac.setValue(float(vision.get("min_width_frac", 0.08)))

    def _load_default_config_if_present(self) -> None:
        for name in DEFAULT_CONFIG_NAMES:
            path = Path(name)
            if path.exists():
                try:
                    self._load_config_file(path)
                    self._append_log(f"Configuração carregada: {path.resolve()}")
                    return
                except Exception as exc:
                    self._append_log(f"Falha ao carregar {path}: {exc}")

    def _load_config_file(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        self.apply_config(config)

    def _load_config_dialog(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Carregar configuração",
            str(Path.cwd()),
            "YAML (*.yaml *.yml);;Todos os arquivos (*)",
        )
        if not filename:
            return
        try:
            self._load_config_file(Path(filename))
            self._append_log(f"Configuração carregada: {filename}")
        except Exception as exc:
            QMessageBox.critical(self, "Erro", str(exc))

    def _save_config_dialog(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar configuração",
            str(Path.cwd() / "config_vision_gui.yaml"),
            "YAML (*.yaml *.yml)",
        )
        if not filename:
            return
        try:
            with open(filename, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.current_config(), f, sort_keys=False, allow_unicode=True)
            self._append_log(f"Configuração salva: {filename}")
        except Exception as exc:
            QMessageBox.critical(self, "Erro", str(exc))

    # -------------------------- Câmera --------------------------------

    def _update_camera_controls(self) -> None:
        mode = self.camera_mode.currentText()
        emulator = mode == "Basler Emulator"
        production = mode == "Produção simulada"

        self.emulator_image.setEnabled(emulator)
        self.browse_image_btn.setEnabled(emulator)

        if hasattr(self, "production_group"):
            self.production_group.setEnabled(production)

        self._update_buttons()

    def _browse_production_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Selecionar diretório de patches",
            self.production_dir.text().strip() or str(Path.cwd()),
        )
        if directory:
            self.production_dir.setText(directory)

    def _browse_emulator_image(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar imagem da madeira",
            str(Path.cwd()),
            "Imagens (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;Todos os arquivos (*)",
        )
        if filename:
            self.emulator_image.setText(filename)

    def _initialize_camera(self) -> None:
        if self._thread is not None:
            return

        self._close_camera()
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            mode = self.camera_mode.currentText()

            if mode == "Produção simulada":
                seed = (
                    None
                    if self.production_seed.value() < 0
                    else int(self.production_seed.value())
                )
                self.camera_source = ProductionSimulationCamera(
                    image_dir=self.production_dir.text().strip(),
                    frames_per_board=int(self.max_frames.value()),
                    frame_width=1280,
                    frame_height=1024,
                    pixels_per_mm=2.0,
                    width_min_mm=float(self.production_width_min.value()),
                    width_max_mm=float(self.production_width_max.value()),
                    seed=seed,
                )
                self.camera_source.open()
                self.camera_status.setText(
                    f"OK — produção simulada\n"
                    f"{len(self.camera_source.image_files)} imagens disponíveis"
                )
                self._append_log(
                    "Câmera de produção simulada inicializada."
                )
            else:
                self.camera_source = open_basler_source(
                    mode,
                    self.emulator_image.text().strip() or None,
                )
                info = self.camera_source.camera.GetDeviceInfo()
                self.camera_status.setText(
                    f"OK — {info.GetFriendlyName()}\n{info.GetModelName()}"
                )
                self._append_log("Câmera inicializada.")

        except Exception as exc:
            self.camera_source = None
            self.camera_status.setText("Falha ao inicializar")
            QMessageBox.critical(self, "Erro ao abrir câmera", str(exc))
            self._append_log(traceback.format_exc())
        finally:
            QApplication.restoreOverrideCursor()
            self._update_buttons()

    def _close_camera(self) -> None:
        if self._thread is not None:
            QMessageBox.information(
                self,
                "Processamento em andamento",
                "A câmera será mantida aberta até o processamento atual terminar.",
            )
            return

        if self.camera_source is not None:
            try:
                self.camera_source.stop()
            except Exception:
                self._append_log(traceback.format_exc())
            finally:
                self.camera_source = None
        self.camera_status.setText("Não inicializada")
        self._update_buttons()

    # ------------------------- Execução -------------------------------

    def _run_pipeline(self) -> None:
        if self.camera_source is None:
            self._initialize_camera()
            if self.camera_source is None:
                return

        if self._thread is not None:
            return

        if isinstance(self.camera_source, ProductionSimulationCamera):
            self.camera_source.begin_board()

        self._set_busy(True)
        self._append_log("Iniciando processamento de uma peça...")

        self._thread = QThread(self)
        self._worker = PipelineWorker(self.camera_source, self.current_config())
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._pipeline_finished)
        self._worker.failed.connect(self._worker_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread_finished)

        self._thread.start()

    def _start_production_simulation(self) -> None:
        if self._thread is not None:
            return

        if self.camera_mode.currentText() != "Produção simulada":
            self.camera_mode.setCurrentText("Produção simulada")

        # Reabre para aplicar diretório, seed e faixa de largura atuais.
        self._close_camera()
        self._initialize_camera()
        if not isinstance(self.camera_source, ProductionSimulationCamera):
            return

        width_min = float(self.production_width_min.value())
        width_max = float(self.production_width_max.value())
        if width_max < width_min:
            QMessageBox.warning(
                self,
                "Faixa inválida",
                "A largura máxima deve ser maior ou igual à largura mínima.",
            )
            return

        report_text = self.production_report.text().strip()
        report_path = (
            Path(report_text)
            if report_text
            else timestamped_report_path("reports")
        )

        self.production_report.setText(str(report_path))
        self.production_summary_label.setText("Simulação em andamento...")

        self._set_busy(True)
        self.production_stop_btn.setEnabled(True)

        self._thread = QThread(self)
        self._worker = ProductionWorker(
            camera=self.camera_source,
            config=self.current_config(),
            n_boards=int(self.production_boards.value()),
            report_path=str(report_path),
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.board_finished.connect(self._production_board_finished)
        self._worker.finished.connect(self._production_finished)
        self._worker.failed.connect(self._worker_failed)

        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread_finished)

        self._append_log(
            f"Simulação de produção iniciada: "
            f"{self.production_boards.value()} tábuas."
        )
        self._thread.start()

    def _stop_production_simulation(self) -> None:
        if isinstance(self._worker, ProductionWorker):
            self._worker.request_stop()
            self.production_stop_btn.setEnabled(False)
            self.production_summary_label.setText(
                "Parada solicitada; finalizando a peça atual."
            )
            self._append_log("Parada da simulação solicitada.")

    def _production_board_finished(
        self,
        pipeline: NeuroWoodPipeline,
        result: Any,
        row: dict[str, Any],
    ) -> None:
        self.last_pipeline = pipeline
        self.last_piece_frames = pipeline.last_piece_frames
        self.last_detection_results = pipeline.last_detection_results
        self.last_summary = pipeline.last_summary

        if result is not None:
            self._show_pipeline_result(pipeline, result)

        self.production_summary_label.setText(
            f"Peça {row.get('sequence')} | "
            f"{row.get('classification')} | "
            f"largura={row.get('width_mm') if row.get('width_mm') is not None else '—'} mm | "
            f"tempo={row.get('processing_time_ms') if row.get('processing_time_ms') is not None else '—'} ms"
        )

        self._append_log(
            f"Produção peça {row.get('sequence')}: "
            f"{row.get('classification')} | "
            f"largura={row.get('width_mm')} mm | "
            f"tempo={row.get('processing_time_ms')} ms | "
            f"defeitos={row.get('n_defects')}"
        )

    def _production_finished(
        self,
        summary: dict[str, Any],
        report_path: str,
    ) -> None:
        avg_time = summary.get("avg_processing_time_ms")
        avg_width = summary.get("avg_width_mm")

        self.production_summary_label.setText(
            f"Processadas: {summary.get('boards_processed', 0)} | "
            f"Rejeitadas: {summary.get('rejected', 0)} "
            f"({summary.get('reject_rate_pct', 0):.1f}%) | "
            f"Tempo médio: {avg_time:.1f} ms | "
            f"Largura média: {avg_width:.1f} mm"
            if avg_time is not None and avg_width is not None
            else (
                f"Processadas: {summary.get('boards_processed', 0)} | "
                f"Rejeitadas: {summary.get('rejected', 0)} | "
                f"Relatório: {report_path}"
            )
        )
        self.production_report.setText(report_path)
        self.production_stop_btn.setEnabled(False)

        self._append_log(
            f"Simulação concluída. "
            f"processadas={summary.get('boards_processed', 0)} | "
            f"rejeitadas={summary.get('rejected', 0)} | "
            f"taxa={summary.get('reject_rate_pct', 0):.1f}% | "
            f"relatório={report_path}"
        )

    def _pipeline_finished(self, pipeline: NeuroWoodPipeline, result: Any) -> None:
        self.last_pipeline = pipeline
        self.last_piece_frames = pipeline.last_piece_frames
        self.last_detection_results = pipeline.last_detection_results
        self.last_summary = pipeline.last_summary

        if result is None:
            self.classification_label.setText("NO PIECE")
            self.board_id_label.setText("—")
            self.width_label.setText("—")
            self.defects_label.setText("0")
            self.processing_label.setText("—")
            info = pipeline.last_piece_info or {}
            self.summary_label.setText(
                "Nenhuma peça detectada. "
                f"Contraste medido: {info.get('contrast', 0):.2f}"
            )
            self._append_log("Nenhuma peça detectada no frame inicial.")
            return

        self._show_pipeline_result(pipeline, result)
        self._append_log(
            f"Processamento concluído: {result.classification} | "
            f"defeitos={len(result.defects)} | {result.processing_time_ms:.1f} ms"
        )

    def _show_pipeline_result(self, pipeline: NeuroWoodPipeline, result: Any) -> None:
        classification = str(result.classification)
        self.classification_label.setText(classification)
        if classification == "REJECT":
            self.classification_label.setStyleSheet(
                "font-size: 28px; font-weight: bold; color: #d9534f;"
            )
        else:
            self.classification_label.setStyleSheet(
                "font-size: 28px; font-weight: bold; color: #5cb85c;"
            )

        self.board_id_label.setText(str(result.board_id))

        piece_info = pipeline.last_piece_info or {}
        width_mm = piece_info.get("width_mm")
        self.width_label.setText(
            f"{float(width_mm):.1f} mm"
            if width_mm is not None
            else "—"
        )

        self.defects_label.setText(str(len(result.defects)))
        self.processing_label.setText(f"{result.processing_time_ms:.1f} ms")

        summary = pipeline.last_summary or {}
        self.summary_label.setText(
            f"Frames: {summary.get('frames_analyzed', 0)} | "
            f"Frames c/ defeito: {summary.get('frames_with_defect', 0)} | "
            f"Área total marcada: {summary.get('defect_area_pct', 0):.3f}%"
        )

        if pipeline.last_board_marked is not None:
            self._display_board(pipeline.last_board_marked)
        self._display_patches(pipeline.last_detection_results or [])
        self._fill_defect_table(summary)

    def _reprocess_current_capture(self) -> None:
        if not self.last_piece_frames:
            QMessageBox.information(
                self,
                "Sem captura",
                "Primeiro processe uma peça para capturar os patches.",
            )
            return
        if self._thread is not None:
            return

        self._set_busy(True)
        self._append_log(
            "Reprocessando os mesmos patches com os parâmetros atuais de defeitos..."
        )

        self._thread = QThread(self)
        self._worker = ReprocessWorker(self.last_piece_frames, self.current_config())
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._reprocess_finished)
        self._worker.failed.connect(self._worker_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread_finished)
        self._thread.start()

    def _reprocess_finished(self, results: list[dict[str, Any]], summary: dict[str, Any], board_marked: np.ndarray) -> None:
        self.last_detection_results = results
        self.last_summary = summary

        classification = "REJECT" if summary.get("total_defects", 0) else "ACCEPT"
        self.classification_label.setText(classification)
        self.classification_label.setStyleSheet(
            "font-size: 28px; font-weight: bold; color: "
            + ("#d9534f;" if classification == "REJECT" else "#5cb85c;")
        )
        self.board_id_label.setText("REPROCESSADO")
        # A largura permanece a obtida no primeiro processamento da captura.
        self.defects_label.setText(str(summary.get("total_defects", 0)))
        self.processing_label.setText("—")
        self.summary_label.setText(
            f"Mesma captura | rigor={summary.get('rigor', 0):.2f} | "
            f"frames c/ defeito={summary.get('frames_with_defect', 0)} | "
            f"área marcada={summary.get('defect_area_pct', 0):.3f}%"
        )

        self._display_board(board_marked)
        self._display_patches(results)
        self._fill_defect_table(summary)
        self._append_log(
            f"Reprocessamento concluído: {classification} | "
            f"defeitos={summary.get('total_defects', 0)}"
        )

    def _worker_failed(self, traceback_text: str) -> None:
        self._append_log(traceback_text)
        QMessageBox.critical(
            self,
            "Erro no processamento",
            traceback_text.splitlines()[-1] if traceback_text else "Erro desconhecido",
        )

    def _thread_finished(self) -> None:
        if self._worker is not None:
            try:
                self._worker.deleteLater()
            except Exception:
                pass
        if self._thread is not None:
            try:
                self._thread.deleteLater()
            except Exception:
                pass
        self._worker = None
        self._thread = None
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        production_mode = self.camera_mode.currentText() == "Produção simulada"

        self.run_btn.setEnabled(not busy and self.camera_source is not None)
        self.production_run_btn.setEnabled(not busy and production_mode)
        self.production_stop_btn.setEnabled(
            busy and isinstance(self._worker, ProductionWorker)
        )
        self.reprocess_btn.setEnabled(not busy and bool(self.last_piece_frames))
        self.init_camera_btn.setEnabled(not busy)
        self.close_camera_btn.setEnabled(not busy and self.camera_source is not None)
        self.load_config_btn.setEnabled(not busy)
        self.save_config_btn.setEnabled(not busy)

    def _update_buttons(self) -> None:
        busy = self._thread is not None
        production_mode = self.camera_mode.currentText() == "Produção simulada"

        self.run_btn.setEnabled(not busy and self.camera_source is not None)
        self.production_run_btn.setEnabled(not busy and production_mode)
        self.production_stop_btn.setEnabled(
            busy and isinstance(self._worker, ProductionWorker)
        )
        self.reprocess_btn.setEnabled(not busy and bool(self.last_piece_frames))
        self.close_camera_btn.setEnabled(not busy and self.camera_source is not None)

    # ------------------------- Visualização ---------------------------

    def _display_board(self, image: np.ndarray) -> None:
        self.board_label.set_image(image, max_height=28)
        self.tabs.setCurrentIndex(0)

    def _clear_patches(self) -> None:
        while self.patches_grid.count():
            item = self.patches_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _display_patches(self, results: list[dict[str, Any]]) -> None:
        self._clear_patches()

        columns = 3
        for idx, result in enumerate(results):
            card = QGroupBox(
                f"Patch {result.get('frame_id')} — defeitos: {result.get('n_defects', 0)}"
            )
            card_layout = QVBoxLayout(card)
            label = ImageLabel()
            label.set_image(result["marked_image"], max_height=190)
            card_layout.addWidget(label)

            details = QLabel(
                f"Área marcada: {result.get('defect_area_pct', 0):.3f}% | "
                f"threshold: {result.get('threshold', 0):.2f}"
            )
            card_layout.addWidget(details)
            self.patches_grid.addWidget(card, idx // columns, idx % columns)

    def _fill_defect_table(self, summary: dict[str, Any]) -> None:
        rows: list[tuple[Any, ...]] = []
        for patch in summary.get("patches", []):
            for i, defect in enumerate(patch.get("defects", []), start=1):
                rows.append(
                    (
                        patch.get("frame_id"),
                        i,
                        defect.get("area_pixels"),
                        defect.get("area_pct"),
                        defect.get("bbox"),
                    )
                )

        self.defect_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            values = [
                str(row[0]),
                str(row[1]),
                str(row[2]),
                f"{float(row[3]):.3f}",
                str(tuple(row[4])) if row[4] is not None else "—",
            ]
            for col_idx, value in enumerate(values):
                self.defect_table.setItem(row_idx, col_idx, QTableWidgetItem(value))

    # --------------------------- Logging ------------------------------

    def _install_log_handler(self) -> None:
        self.log_emitter = LogEmitter()
        self.log_emitter.message.connect(self._append_log)
        self.log_handler = QtLogHandler(self.log_emitter)
        self.log_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )

        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(self.log_handler)

    def _append_log(self, text: str) -> None:
        self.log_text.append(str(text).rstrip())

    # --------------------------- Close --------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self._thread is not None:
            QMessageBox.warning(
                self,
                "Processamento em andamento",
                "Feche a janela depois que o processamento terminar.",
            )
            event.ignore()
            return

        if self.camera_source is not None:
            try:
                self.camera_source.stop()
            except Exception:
                pass

        try:
            logging.getLogger().removeHandler(self.log_handler)
        except Exception:
            pass

        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = NeurowoodWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
