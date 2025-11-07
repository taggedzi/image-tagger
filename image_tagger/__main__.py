"""Command line entry point for the Image Tagger project."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import ImageAnalyzer, OutputMode, SettingsStore
from .models.registry import ModelRegistry


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Image Tagger")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        help="Image file or directory to process in headless mode.",
    )
    parser.add_argument(
        "--model",
        help="Override the configured model identifier.",
    )
    parser.add_argument(
        "--output-mode",
        choices=[mode.value for mode in OutputMode],
        help="Override how metadata is persisted.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available models and exit.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without launching the GUI.",
    )

    args = parser.parse_args(argv)

    if args.list_models:
        infos = ModelRegistry.list_model_infos()
        payload = []
        for info in infos:
            data = asdict(info)
            data["capabilities"] = [cap.value for cap in info.capabilities]
            payload.append(data)
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    if not args.headless and args.input is None:
        from .gui import run_app

        run_app()
        return

    if args.input is None:
        parser.error("--input is required when running in headless mode.")

    store = SettingsStore()
    config = store.load()

    if args.model:
        config.model_name = args.model
    if args.output_mode:
        config.output_mode = OutputMode(args.output_mode)

    analyzer = ImageAnalyzer(config)
    results = analyzer.analyze_target(args.input)

    output = [
        {
            "path": str(result.image_path),
            "caption": result.caption,
            "tags": result.tags,
            "embedded": result.embedded,
            "sidecar_path": str(result.sidecar_path) if result.sidecar_path else None,
            "error": result.error_message,
        }
        for result in results
    ]

    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":  # pragma: no cover
    main()
