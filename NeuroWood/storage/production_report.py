from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from statistics import mean


REPORT_FIELDS = [
    "sequence",
    "timestamp",
    "board_id",
    "classification",
    "rejected",
    "processing_time_ms",
    "width_mm",
    "width_px",
    "width_min_mm",
    "width_max_mm",
    "width_std_mm",
    "simulated_width_mm",
    "width_error_mm",
    "valid_width_samples",
    "n_defects",
    "frames_analyzed",
    "frames_with_defect",
    "defect_area_pct",
    "rigor",
    "source_files",
]


class ProductionReport:
    """Relatório incremental de produção simulada em CSV."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []

    def add_board(self, sequence, pipeline, result, camera=None):
        piece_info = pipeline.last_piece_info or {}
        summary = pipeline.last_summary or {}

        width_mm = piece_info.get("width_mm")
        width_px = piece_info.get("width_px")
        width_min_mm = piece_info.get("width_min_mm")
        width_max_mm = piece_info.get("width_max_mm")
        width_std_mm = piece_info.get("width_std_mm")

        simulated_width_mm = getattr(camera, "current_width_mm", None)
        source_files = getattr(camera, "current_source_files", []) or []

        if width_mm is not None and simulated_width_mm is not None:
            width_error_mm = float(width_mm) - float(simulated_width_mm)
        else:
            width_error_mm = None

        if result is None:
            board_id = ""
            classification = "NO_PIECE"
            processing_time_ms = None
            n_defects = 0
            timestamp = datetime.now().isoformat(timespec="seconds")
        else:
            board_id = str(result.board_id)
            classification = str(result.classification)
            processing_time_ms = float(result.processing_time_ms)
            n_defects = len(result.defects)
            ts = getattr(result, "timestamp", None)
            timestamp = (
                ts.isoformat(timespec="seconds")
                if hasattr(ts, "isoformat")
                else datetime.now().isoformat(timespec="seconds")
            )

        row = {
            "sequence": int(sequence),
            "timestamp": timestamp,
            "board_id": board_id,
            "classification": classification,
            "rejected": int(classification == "REJECT"),
            "processing_time_ms": _round_or_none(processing_time_ms, 2),
            "width_mm": _round_or_none(width_mm, 2),
            "width_px": _round_or_none(width_px, 1),
            "width_min_mm": _round_or_none(width_min_mm, 2),
            "width_max_mm": _round_or_none(width_max_mm, 2),
            "width_std_mm": _round_or_none(width_std_mm, 3),
            "simulated_width_mm": _round_or_none(simulated_width_mm, 2),
            "width_error_mm": _round_or_none(width_error_mm, 2),
            "valid_width_samples": len(piece_info.get("width_samples_mm", []) or []),
            "n_defects": int(n_defects),
            "frames_analyzed": int(summary.get("frames_analyzed", 0) or 0),
            "frames_with_defect": int(summary.get("frames_with_defect", 0) or 0),
            "defect_area_pct": _round_or_none(
                summary.get("defect_area_pct", 0.0),
                4,
            ),
            "rigor": _round_or_none(summary.get("rigor"), 3),
            "source_files": "|".join(source_files),
        }

        self.rows.append(row)
        self.save()
        return row

    def save(self):
        with self.path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=REPORT_FIELDS,
                delimiter=";",
            )
            writer.writeheader()
            writer.writerows(self.rows)

    def summary(self):
        processed = [
            r for r in self.rows
            if r["classification"] in {"ACCEPT", "REJECT"}
        ]
        rejected = [r for r in processed if r["classification"] == "REJECT"]

        times = [
            float(r["processing_time_ms"])
            for r in processed
            if r["processing_time_ms"] is not None
        ]
        widths = [
            float(r["width_mm"])
            for r in processed
            if r["width_mm"] is not None
        ]
        width_errors = [
            abs(float(r["width_error_mm"]))
            for r in processed
            if r["width_error_mm"] is not None
        ]
        width_stds = [
            float(r["width_std_mm"])
            for r in processed
            if r["width_std_mm"] is not None
        ]

        count = len(processed)

        return {
            "boards_total": len(self.rows),
            "boards_processed": count,
            "accepted": count - len(rejected),
            "rejected": len(rejected),
            "reject_rate_pct": (
                100.0 * len(rejected) / count
                if count else 0.0
            ),
            "avg_processing_time_ms": mean(times) if times else None,
            "min_processing_time_ms": min(times) if times else None,
            "max_processing_time_ms": max(times) if times else None,
            "avg_width_mm": mean(widths) if widths else None,
            "min_width_mm": min(widths) if widths else None,
            "max_width_mm": max(widths) if widths else None,
            "avg_width_std_mm": mean(width_stds) if width_stds else None,
            "avg_abs_width_error_mm": mean(width_errors) if width_errors else None,
            "report_path": str(self.path),
        }


def timestamped_report_path(directory="reports"):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(directory) / f"production_report_{stamp}.csv"


def _round_or_none(value, decimals):
    if value is None:
        return None
    return round(float(value), decimals)
