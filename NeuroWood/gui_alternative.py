"""
Neurowood - GUI alternativa com foco na visualizacao da tabua reconstruida.

Uso:
    1. Coloque este arquivo na raiz do projeto, ao lado de gui.py.
    2. Execute:
           python gui_alternative.py

A logica de camera, pipeline, reprocessamento, configuracao e deteccao continua
sendo herdada do gui.py atual. Este arquivo altera somente a composicao visual
e a forma de exibir a tabua reconstruida.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui import ImageLabel, NeurowoodWindow, ndarray_to_pixmap


APP_TITLE_ALT = "Neurowood - Inspection Console"


class NeurowoodWideWindow(NeurowoodWindow):
    """Layout alternativo que prioriza a tabua reconstruida."""

    def __init__(self) -> None:
        self._last_board_image: np.ndarray | None = None
        super().__init__()
        self.setWindowTitle(APP_TITLE_ALT)
        self.resize(1600, 950)
        self.setMinimumSize(1180, 760)
        self._apply_alt_style()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # --------------------------------------------------------------
        # 1) Barra superior: resultado sempre visivel
        # --------------------------------------------------------------
        status_bar = QFrame()
        status_bar.setObjectName("statusBar")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(14, 8, 14, 8)
        status_layout.setSpacing(18)

        title_box = QVBoxLayout()
        title = QLabel("NEUROWOOD")
        title.setObjectName("appTitle")
        subtitle = QLabel("Inspecao e calibracao da classificacao de madeira")
        subtitle.setObjectName("appSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        status_layout.addLayout(title_box, 2)

        status_layout.addWidget(self._separator())

        self.classification_label = QLabel("—")
        self.classification_label.setObjectName("classification")
        self.classification_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self._metric_box("CLASSIFICACAO", self.classification_label), 1)

        self.board_id_label = QLabel("—")
        self.board_id_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        status_layout.addWidget(self._metric_box("BOARD ID", self.board_id_label), 1)

        self.defects_label = QLabel("0")
        status_layout.addWidget(self._metric_box("DEFEITOS", self.defects_label), 1)

        self.processing_label = QLabel("—")
        status_layout.addWidget(self._metric_box("TEMPO", self.processing_label), 1)

        status_layout.addWidget(self._separator())

        self.summary_label = QLabel("Sem processamento.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setMinimumWidth(260)
        status_layout.addWidget(self.summary_label, 2)

        root.addWidget(status_bar)

        # --------------------------------------------------------------
        # 2) Area principal: tabua ocupa a maior parte da janela
        # --------------------------------------------------------------
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.setChildrenCollapsible(False)
        root.addWidget(main_splitter, 1)

        board_group = QGroupBox("Tabua reconstruida")
        board_layout = QVBoxLayout(board_group)
        board_layout.setContentsMargins(8, 8, 8, 8)
        board_layout.setSpacing(6)

        board_toolbar = QHBoxLayout()
        board_hint = QLabel(
            "A imagem reconstruida permanece visivel enquanto voce ajusta parametros e consulta resultados."
        )
        board_hint.setObjectName("mutedText")
        board_toolbar.addWidget(board_hint, 1)

        board_toolbar.addWidget(QLabel("Visualizacao:"))
        self.board_view_mode = QComboBox()
        self.board_view_mode.addItems(
            ["Ajustar a area", "Altura 420 px", "Tamanho original"]
        )
        self.board_view_mode.setMinimumWidth(160)
        self.board_view_mode.currentTextChanged.connect(self._render_board_view)
        board_toolbar.addWidget(self.board_view_mode)
        board_layout.addLayout(board_toolbar)

        self.board_scroll = QScrollArea()
        self.board_scroll.setWidgetResizable(False)
        self.board_scroll.setAlignment(Qt.AlignCenter)

        self.board_label = ImageLabel("A tabua reconstruida aparecera aqui.")
        self.board_label.setMinimumSize(1000, 330)
        self.board_scroll.setWidget(self.board_label)
        board_layout.addWidget(self.board_scroll, 1)

        main_splitter.addWidget(board_group)

        # --------------------------------------------------------------
        # 3) Faixa inferior: controles e diagnostico em abas
        # --------------------------------------------------------------
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        main_splitter.addWidget(self.tabs)

        self._build_controls_tab()
        self._build_patches_tab()
        self._build_defects_tab()
        self._build_logs_tab()

        # Aproximadamente 70% da altura para a tabua na abertura.
        main_splitter.setStretchFactor(0, 7)
        main_splitter.setStretchFactor(1, 3)
        main_splitter.setSizes([650, 270])

    def _build_controls_tab(self) -> None:
        tab = QWidget()
        outer = QHBoxLayout(tab)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        # Cada coluna tem seu proprio scroll para funcionar bem em telas menores.
        left_widget, left_layout, left_scroll = self._control_column()
        middle_widget, middle_layout, middle_scroll = self._control_column()
        right_widget, right_layout, right_scroll = self._control_column()

        # Os builders originais adicionam seus grupos em self.controls_layout.
        # Redirecionamos esse atributo somente durante a montagem de cada coluna.
        self.controls_layout = left_layout
        self._build_camera_group()
        self._build_actions_group()
        left_layout.addStretch(1)

        self.controls_layout = middle_layout
        self._build_capture_group()
        self._build_piece_group()
        middle_layout.addStretch(1)

        self.controls_layout = right_layout
        self._build_defect_group()
        right_layout.addStretch(1)

        # Mantem uma referencia valida para compatibilidade com o gui.py base.
        self.controls_layout = left_layout

        outer.addWidget(left_scroll, 1)
        outer.addWidget(middle_scroll, 1)
        outer.addWidget(right_scroll, 1)
        self.tabs.addTab(tab, "Controles")

    @staticmethod
    def _control_column() -> tuple[QWidget, QVBoxLayout, QScrollArea]:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(container)
        return container, layout, scroll

    def _build_patches_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)

        self.patches_scroll = QScrollArea()
        self.patches_scroll.setWidgetResizable(True)
        self.patches_container = QWidget()
        self.patches_grid = QGridLayout(self.patches_container)
        self.patches_grid.setContentsMargins(6, 6, 6, 6)
        self.patches_grid.setSpacing(8)
        self.patches_scroll.setWidget(self.patches_container)

        layout.addWidget(self.patches_scroll)
        self.tabs.addTab(tab, "Patches")

    def _build_defects_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)

        self.defect_table = QTableWidget(0, 5)
        self.defect_table.setHorizontalHeaderLabels(
            ["Patch", "Defeito", "Area (px2)", "Area (%)", "BBox (x,y,w,h)"]
        )
        header = self.defect_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        self.defect_table.setAlternatingRowColors(True)

        layout.addWidget(self.defect_table)
        self.tabs.addTab(tab, "Defeitos")

    def _build_logs_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)
        layout.addWidget(self.log_text)

        self.tabs.addTab(tab, "Logs")

    # ------------------------------------------------------------------
    # Visualizacao da tabua
    # ------------------------------------------------------------------

    def _display_board(self, image: np.ndarray) -> None:
        """Mantem a tabua sempre visivel; nao troca a aba inferior."""
        self._last_board_image = np.ascontiguousarray(image)
        self._render_board_view()

    def _render_board_view(self, *_args: Any) -> None:
        if self._last_board_image is None:
            return

        pixmap = ndarray_to_pixmap(self._last_board_image)
        if pixmap.isNull():
            return

        mode = self.board_view_mode.currentText()

        if mode == "Ajustar a area":
            viewport = self.board_scroll.viewport().size()
            target_w = max(200, viewport.width() - 18)
            target_h = max(140, viewport.height() - 18)
            shown = pixmap.scaled(
                target_w,
                target_h,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        elif mode == "Altura 420 px":
            shown = pixmap.scaledToHeight(420, Qt.SmoothTransformation)
        else:
            shown = pixmap

        self.board_label._original_pixmap = shown
        self.board_label.setPixmap(shown)
        self.board_label.resize(
            max(self.board_label.minimumWidth(), shown.width()),
            max(self.board_label.minimumHeight(), shown.height()),
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - API Qt
        super().resizeEvent(event)
        if (
            hasattr(self, "board_view_mode")
            and self.board_view_mode.currentText() == "Ajustar a area"
            and self._last_board_image is not None
        ):
            self._render_board_view()

    # ------------------------------------------------------------------
    # Pequenos helpers de apresentacao
    # ------------------------------------------------------------------

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

    @staticmethod
    def _metric_box(title: str, value_widget: QLabel) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(2)

        caption = QLabel(title)
        caption.setObjectName("metricCaption")
        caption.setAlignment(Qt.AlignCenter)
        value_widget.setObjectName(value_widget.objectName() or "metricValue")
        value_widget.setAlignment(Qt.AlignCenter)

        layout.addWidget(caption)
        layout.addWidget(value_widget)
        return box

    def _apply_alt_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f3f5f7;
            }
            QFrame#statusBar {
                background: white;
                border: 1px solid #d8dde3;
                border-radius: 7px;
            }
            QLabel#appTitle {
                font-size: 22px;
                font-weight: 700;
                color: #1f2933;
            }
            QLabel#appSubtitle, QLabel#mutedText {
                color: #6b7280;
            }
            QLabel#metricCaption {
                font-size: 10px;
                font-weight: 600;
                color: #6b7280;
            }
            QLabel#metricValue {
                font-size: 16px;
                font-weight: 600;
                color: #1f2933;
            }
            QLabel#classification {
                font-size: 24px;
                font-weight: 800;
            }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #d5d9df;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 9px;
                padding: 0 4px;
            }
            QTabWidget::pane {
                border: 1px solid #d5d9df;
                background: white;
            }
            QTabBar::tab {
                padding: 7px 16px;
            }
            QPushButton {
                min-height: 28px;
                padding: 3px 8px;
            }
            QTableWidget, QTextEdit, QScrollArea {
                background: white;
            }
            """
        )


def main() -> int:
    app = QApplication(sys.argv)
    window = NeurowoodWideWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
