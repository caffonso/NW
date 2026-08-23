from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from hardware.camera import Camera


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ProductionSimulationCamera(Camera):
    """
    Câmera virtual para simulação de produção em série.

    Por padrão, a tábua simulada tem 400 mm, pois os patches de referência
    usados nesta etapa correspondem a tábuas de 400 mm.

    A faixa continua configurável: se width_min_mm != width_max_mm, a largura
    é sorteada dentro da faixa para testes posteriores.
    """

    def __init__(
        self,
        image_dir,
        frames_per_board=13,
        frame_width=1280,
        frame_height=1024,
        pixels_per_mm=2.0,
        width_min_mm=400.0,
        width_max_mm=400.0,
        background_intensity=35,
        seed=None,
    ):
        self.image_dir = Path(image_dir)
        self.frames_per_board = int(frames_per_board)
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.pixels_per_mm = float(pixels_per_mm)
        self.width_min_mm = float(width_min_mm)
        self.width_max_mm = float(width_max_mm)
        self.background_intensity = int(background_intensity)
        self.rng = random.Random(seed)

        self.connected = False
        self.image_files = []

        self.current_board_index = 0
        self.current_width_mm = None
        self.current_width_px = None
        self.current_source_files = []
        self._current_frames = []
        self._capture_counter = 0

    def open(self):
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Diretório de patches não encontrado: {self.image_dir}"
            )

        self.image_files = sorted(
            p
            for p in self.image_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
            and not p.name.lower().startswith("basler_emulator_input")
        )

        if not self.image_files:
            raise FileNotFoundError(
                f"Nenhuma imagem válida encontrada em {self.image_dir}"
            )

        if self.frames_per_board <= 0:
            raise ValueError("frames_per_board deve ser > 0.")

        if self.pixels_per_mm <= 0:
            raise ValueError("pixels_per_mm deve ser > 0.")

        if self.width_min_mm <= 0 or self.width_max_mm < self.width_min_mm:
            raise ValueError("Faixa de largura simulada inválida.")

        max_supported_mm = (self.frame_height - 80) / self.pixels_per_mm
        if self.width_max_mm > max_supported_mm:
            raise ValueError(
                f"Largura simulada {self.width_max_mm:.1f} mm excede o máximo "
                f"de {max_supported_mm:.1f} mm para frame_height={self.frame_height} "
                f"e pixels_per_mm={self.pixels_per_mm}."
            )

        self.connected = True

    def close(self):
        self.connected = False
        self._current_frames = []
        self.current_source_files = []

    def is_connected(self):
        return self.connected

    def stop(self):
        self.close()

    def begin_board(self):
        """Prepara uma nova tábua e seus N frames."""
        if not self.connected:
            raise RuntimeError("ProductionSimulationCamera não está aberta.")

        self.current_board_index += 1
        self._capture_counter = 0

        if self.width_min_mm == self.width_max_mm:
            self.current_width_mm = float(self.width_min_mm)
        else:
            self.current_width_mm = self.rng.uniform(
                self.width_min_mm,
                self.width_max_mm,
            )

        self.current_width_px = int(
            round(self.current_width_mm * self.pixels_per_mm)
        )
        self.current_width_mm = self.current_width_px / self.pixels_per_mm

        if len(self.image_files) >= self.frames_per_board:
            selected = self.rng.sample(
                self.image_files,
                self.frames_per_board,
            )
        else:
            selected = [
                self.rng.choice(self.image_files)
                for _ in range(self.frames_per_board)
            ]

        self.current_source_files = [str(p) for p in selected]
        self._current_frames = [
            self._patch_to_frame(p, self.current_width_px)
            for p in selected
        ]

        return {
            "simulation_board_index": self.current_board_index,
            "simulated_width_mm": float(self.current_width_mm),
            "simulated_width_px": int(self.current_width_px),
            "source_files": list(self.current_source_files),
        }

    def capture(self):
        if not self.connected:
            raise RuntimeError("ProductionSimulationCamera não está aberta.")

        if not self._current_frames:
            self.begin_board()

        if self._capture_counter == 0:
            frame = self._current_frames[0]
        else:
            idx = self._capture_counter - 1
            if idx >= len(self._current_frames):
                raise RuntimeError(
                    "Foram solicitados mais frames que o preparado para a tábua. "
                    "Chame begin_board() antes de processar a próxima peça."
                )
            frame = self._current_frames[idx]

        self._capture_counter += 1
        return frame.copy()

    def _patch_to_frame(self, path: Path, target_board_height: int):
        patch = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if patch is None:
            raise ValueError(f"Não foi possível abrir patch: {path}")

        patch = self._ensure_wood_brightness(patch)

        patch = cv2.resize(
            patch,
            (self.frame_width, target_board_height),
            interpolation=cv2.INTER_AREA
            if patch.shape[0] > target_board_height
            else cv2.INTER_LINEAR,
        )

        frame = np.full(
            (self.frame_height, self.frame_width, 3),
            self.background_intensity,
            dtype=np.uint8,
        )

        top = (self.frame_height - target_board_height) // 2
        bottom = top + target_board_height
        frame[top:bottom, :] = patch

        return frame

    def _ensure_wood_brightness(self, patch):
        arr = patch.astype(np.float32)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        median = float(np.median(gray))

        target_median = max(self.background_intensity + 70, 105)
        if median < target_median:
            arr += target_median - median

        return np.clip(arr, 0, 255).astype(np.uint8)
