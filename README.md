# Image Tagger

Image Tagger is a cross-platform Python library and desktop application that captions and tags images using pluggable AI models. The default build focuses on Windows, but the PySide6-based GUI and the processing pipeline work on macOS and Linux as well.

## Highlights

- Drag-and-drop or dialog-based selection of individual images or entire folders
- Batch processing with threaded execution and live progress feedback
- Two output modes: embed metadata inside supported formats or emit YAML sidecars
- Validated settings dialog with persistent storage in the user's configuration directory
- Extensible model registry with a lightweight fallback plus optional OpenCLIP and multiple BLIP captioning checkpoints
- Clean separation between the core analysis pipeline, metadata IO, and the GUI

## Getting started

### Prerequisites

- Python 3.10 or newer
- Virtual environment (recommended)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\\Scripts\\activate
pip install -U pip
pip install -e .
```

> **Note:** Optional extras can be installed with e.g. `pip install -e .[clip]` for OpenCLIP tagging, `pip install -e .[blip]` for BLIP captioning, or `pip install -e .[full]` for everything. The first time you run a BLIP or CLIP model, Hugging Face will download the weights (up to ~1 GB); this happens in the background once you start processing images.

### Launch the GUI

```bash
python -m image_tagger
```

Drag files onto the drop area or use the buttons to choose files/folders. Open the **Settings** dialog to switch models, tweak tagging behaviour, or change the output mode. Settings are saved to:

- Windows: `%APPDATA%\image_tagger\settings.yaml`
- Linux/macOS: `$XDG_CONFIG_HOME/image_tagger/settings.yaml` (defaults to `~/.config/image_tagger/settings.yaml`)

### Headless usage

You can also process images without the GUI:

```bash
python -m image_tagger --headless --input /path/to/images --model builtin.simple --output-mode sidecar
```

Headless runs emit a JSON summary that lists captions, tags, and where the data was written.

### Listing installed models

```bash
python -m image_tagger --list-models
```

## Architecture overview

```
image_tagger/
├── config.py          # Pydantic-based application settings
├── settings_store.py  # Cross-platform config persistence helper
├── utils/             # Path discovery utilities
├── models/            # Model interfaces, registry, heuristic/CLIP/BLIP implementations
├── services/          # High-level pipeline coordinating models and IO
├── io/                # Metadata writers (EXIF/PNG) and YAML sidecars
└── gui/               # PySide6 application with drag-and-drop and settings dialog
```

`ImageAnalyzer` orchestrates model execution, metadata embedding, and sidecar generation. Models plug in via the `ModelRegistry`; each model adheres to the `TaggingModel` protocol defined in `models/base.py`.

## Extending with new models

1. Create a new module in `image_tagger/models/` that implements `TaggingModel`.
2. Register the factory with `ModelRegistry.register("your.id", YourModel)`.
3. Add any extra dependencies to the relevant optional dependency group in `pyproject.toml`.

The GUI automatically discovers new models via the registry.

## Next steps

- Add additional ML models (e.g. Stable Diffusion captioners, ONNX pipelines)
- Improve metadata embedding coverage for additional formats (WebP, TIFF, etc.)
- Provide thumbnail previews inside the GUI result list
- Bundle automated tests around the heuristic model and pipeline
