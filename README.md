# Image Tagger

Image Tagger is a cross-platform Python library and desktop application that captions and tags images using pluggable AI models. The default build focuses on Windows, but the PySide6-based GUI and the processing pipeline work on macOS and Linux as well.

## Highlights

- Drag-and-drop or dialog-based selection of individual images or entire folders
- Batch processing with threaded execution and live progress feedback
- Two output modes: embed metadata inside supported formats or emit YAML sidecars
- Validated settings dialog with persistent storage in the user's configuration directory
- Extensible model registry with local BLIP captioners (base and large checkpoints) plus Ollama-powered multimodal models
- Connect to local Ollama servers to run multimodal models such as Qwen2.5-VL, LLaVA, MiniCPM-V, Gemma 3, or PaliGemma 2
- Automatic GPU detection with graceful CPU fallback for BLIP; PyTorch selects Metal/CUDA when available
- Preserves existing embedded captions and tags unless you enable the overwrite toggle in settings
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

> **Note:** Install the BLIP captioners with `pip install -e .[blip]`. The first time you run a BLIP model, Hugging Face will download the weights (up to ~1 GB); this happens automatically once you start processing images.

### Development workflow

Install development dependencies (ruff, pycodestyle, pytest, etc.) plus the package in editable mode:

```bash
make install-dev
```

Available quality commands (all defined in the `Makefile`):

- `make fmt` – format the codebase in-place with `ruff format`.
- `make lint` – static analysis via `ruff check`.
- `make style` – run the legacy `pycodestyle` (PEP 8) checks.
- `make test` – execute the pytest suite.
- `make check` – run linting, style, and tests in one go (CI uses this target).
- `make coverage` – collect coverage data with `coverage run` + `coverage report`.

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
python -m image_tagger --headless --input /path/to/images --model caption.blip-base --output-mode sidecar
```

Headless runs emit a JSON summary that lists captions, tags, and where the data was written.

### Using Ollama vision models

1. Install and launch Ollama with a multimodal model, e.g. `ollama run llava` (or `qwen2.5-vl`, `minicpm-v`, `gemma3`, `paligemma-2`).
2. Open the Image Tagger settings dialog and choose **Ollama Vision** from the model list.
3. Adjust the *Remote base URL*, *Remote model id*, *Remote temperature*, *Remote max tokens*, and optional *Remote API key* fields as needed. Use **Refresh list** to pull the currently available vision-capable models from Ollama. All remote-specific settings are persisted alongside the rest of the application configuration.
4. Run analyses through the GUI or via headless mode, e.g.

```bash
python -m image_tagger --headless --model remote.ollama --input ./images
```

> **Tip:** The same remote settings are shared between the GUI and CLI modes. Configuration files now accept the following additional keys: `remote_base_url`, `remote_model`, `remote_temperature`, `remote_max_tokens`, `remote_timeout`, `remote_api_key`, and `overwrite_embedded_metadata`.
> **Warm-up:** The first request may take a while as Ollama loads the model; Image Tagger retries once with an extended timeout, but you can also raise the *Remote timeout* value in settings (default 90 s).
> **Accessibility:** Vision models prompted through Ollama now generate 2–3 sentence captions optimised for alt text, highlighting subjects, relationships, and colours while staying under ~420 characters.

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
├── models/            # Model interfaces plus BLIP and Ollama implementations
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

- Add additional ML models (e.g. OpenAI / Gemini multimodal endpoints)
- Improve metadata embedding coverage for additional formats (WebP, TIFF, etc.)
- Provide thumbnail previews inside the GUI result list
- Bundle automated tests around the BLIP pipelines and remote integrations
