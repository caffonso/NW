from __future__ import annotations

import argparse
from pathlib import Path

from core.pipeline import NeuroWoodPipeline
from hardware.file_camera import FileCamera
from hardware.production_camera import ProductionSimulationCamera
from hardware.simulated_plc import SimulatedPLC
from storage.config import load_config
from storage.logging_config import setup_logging
from storage.production_report import ProductionReport, timestamped_report_path
from vision.visualization import show_pipeline


def run_single(config):
    """Mantém o comportamento anterior: processa uma única imagem."""
    logger = setup_logging(config["logging"])
    logger.info("APP_STARTED")

    camera = FileCamera(config["camera"])
    plc = SimulatedPLC(config["plc"])

    camera.open()
    plc.connect()

    try:
        pipeline = NeuroWoodPipeline(camera, plc, config)
        result = pipeline.run_once()

        if pipeline.last_piece_info:
            width_mm = pipeline.last_piece_info.get("width_mm")
            if width_mm is not None:
                print(f"Largura máxima: {width_mm:.1f} mm")

        if result is not None:
            print(result)
            show_pipeline(
                first=pipeline.last_first_frame,
                piece_frames=pipeline.last_piece_frames,
                board=pipeline.last_board_marked,
                detection_results=pipeline.last_detection_results,
                summary=pipeline.last_summary,
                result=result,
            )
    finally:
        camera.close()
        plc.disconnect()
        logger.info("APP_STOPPED")


def run_production(config, args):
    """
    Simula produção em série usando imagens/paches do diretório informado.

    Exemplo:
        python main.py --production --boards 100
    """
    logger = setup_logging(config["logging"])
    logger.info("PRODUCTION_SIMULATION_STARTED")

    patch_dir = Path(args.patch_dir)
    report_path = (
        Path(args.report)
        if args.report
        else timestamped_report_path("reports")
    )

    frames_per_board = int(
        config.get("capture", {}).get("max_frames", 13)
    )

    camera = ProductionSimulationCamera(
        image_dir=patch_dir,
        frames_per_board=frames_per_board,
        frame_width=1280,
        frame_height=1024,
        pixels_per_mm=2.0,
        width_min_mm=args.width_min,
        width_max_mm=args.width_max,
        seed=args.seed,
    )
    plc = SimulatedPLC(config.get("plc", {}))
    report = ProductionReport(report_path)

    camera.open()
    plc.connect()

    try:
        for sequence in range(1, args.boards + 1):
            camera.begin_board()

            pipeline = NeuroWoodPipeline(camera, plc, config)
            result = pipeline.run_once()

            row = report.add_board(
                sequence=sequence,
                pipeline=pipeline,
                result=result,
                camera=camera,
            )

            print(
                f"[{sequence:04d}/{args.boards:04d}] "
                f"{row['classification']:<8} | "
                f"largura={_fmt(row['width_mm'], 'mm')} | "
                f"tempo={_fmt(row['processing_time_ms'], 'ms')} | "
                f"defeitos={row['n_defects']} | "
                f"área={_fmt(row['defect_area_pct'], '%')}"
            )

        summary = report.summary()

        print("\nRESUMO DA PRODUÇÃO")
        print(f"Peças processadas: {summary['boards_processed']}")
        print(
            f"Rejeitadas: {summary['rejected']} "
            f"({summary['reject_rate_pct']:.1f}%)"
        )
        print(f"Aceitas: {summary['accepted']}")
        print(
            "Tempo médio/peça: "
            f"{_fmt(summary['avg_processing_time_ms'], 'ms')}"
        )
        print(
            "Largura média: "
            f"{_fmt(summary['avg_width_mm'], 'mm')}"
        )
        print(
            "Erro médio absoluto da largura: "
            f"{_fmt(summary['avg_abs_width_error_mm'], 'mm')}"
        )
        print(f"Relatório: {report.path.resolve()}")

    finally:
        camera.close()
        plc.disconnect()
        logger.info("PRODUCTION_SIMULATION_STOPPED")


def _fmt(value, unit):
    if value is None:
        return "—"
    return f"{float(value):.2f} {unit}"


def build_parser():
    parser = argparse.ArgumentParser(description="NeuroWood")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Executa simulação de produção em série.",
    )
    parser.add_argument(
        "--boards",
        type=int,
        default=20,
        help="Número de tábuas na simulação.",
    )
    parser.add_argument(
        "--patch-dir",
        default=str(Path(".neurowood_gui")),
        help="Diretório contendo os patches/imagens.",
    )
    parser.add_argument(
        "--width-min",
        type=float,
        default=90.0,
        help="Largura mínima simulada em mm.",
    )
    parser.add_argument(
        "--width-max",
        type=float,
        default=220.0,
        help="Largura máxima simulada em mm.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed opcional para repetir exatamente a simulação.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Caminho do CSV. Se omitido, cria reports/production_report_*.csv.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    config = load_config("config.yaml")

    if args.production:
        run_production(config, args)
    else:
        run_single(config)


if __name__ == "__main__":
    main()
