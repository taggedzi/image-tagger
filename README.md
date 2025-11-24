# Image Tagger

Image Tagger is a cross-platform Python app and library that captions and tags photos. It can run a local BLIP captioning model or connect to an Ollama server so you can choose the workflow that best fits your computer and bandwidth.

## What you can do
- Drag-and-drop photos or whole folders and process them in batches with live progress
- Create clean, alt-text-friendly captions plus keyword tags
- Write results straight into the image metadata or into YAML sidecar files
- Switch between local BLIP models and remote Ollama vision models without touching the code
- Surface suggested filenames in the results grid (and optionally rename files) so you can accept or ignore them.
- Highlight the drag-and-drop area with a dashed frame that always shows the drop zone and brightens when you hover files over it.
- Suggested filenames now also appear in sidecar metadata (null when disabled) so downstream tools can consistently read that value.
- Ask the model to suggest safe, descriptive filenames and optionally auto-rename your images

## How tagging works

### Local BLIP captioners (default)
- Runs directly on your CPU or GPU using PyTorch and Hugging Face Transformers.
- First launch downloads ~1 GB of weights and caches them in your Hugging Face cache directory (usually `~/.cache/huggingface`).
- Works fully offline once the weights are cached.

### Ollama vision models (optional)
- Requires the free [Ollama](https://ollama.com/download) desktop/server app.
- You download the multimodal model you want (`ollama run llava`, `qwen2.5-vl`, `minicpm-v`, etc.).
- Image Tagger talks to the local Ollama HTTP endpoint and streams the captions/tags back into the app.

## What you need
- Windows, macOS, or Linux with Python 3.10+ already installed.
- ~3 GB of free disk space (Python env + BLIP weights download).
- Optional: an NVIDIA/AMD GPU or Apple Silicon for faster BLIP inference. CPU-only still works.
- Internet access the first time you install dependencies, download BLIP weights, or pull an Ollama model.

## Install without touching system Python
All commands below run inside the project folder. Replace the activation command with the Windows version if needed.

1. **Get the code**
   ```bash
   git clone https://github.com/taggedzi/image-tagger.git
   cd image-tagger
   ```
2. **Create a virtual environment so you do not pollute your global Python**
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```
3. **Install the app**
   ```bash
   pip install -e .
   ```
4. **Install BLIP support (required for local captioning)**
   ```bash
   pip install -e .[blip]
   ```
   The first BLIP run triggers an automatic Hugging Face download. Keep the terminal open until it finishes.
5. **(Optional) list the available models**
   ```bash
   python -m image_tagger --list-models
   ```

## Prepare Ollama (only if you want remote models)
1. Download and install Ollama from [ollama.com/download](https://ollama.com/download). Launch it so the background service starts.
2. Pull a multimodal model once. Example:
   ```bash
   ollama run llava
   ```
   The first call both downloads and tests the model.
3. Keep Ollama running. Image Tagger will connect to `http://127.0.0.1:11434` by default. If you host Ollama elsewhere, copy the base URL and API key (if any) for later.

## Run the application

### Launch the desktop app
```bash
python -m image_tagger
```
- Drop images or folders into the window, or use the buttons to pick them.
- Open **Settings** → **Model** to pick `caption.blip-base`, `caption.blip-large`, or `remote.ollama`.
- When `remote.ollama` is selected, fill in **Remote base URL**, **Remote model id**, and other fields (temperature, token limit, timeout, API key). Click **Refresh list** to see which Ollama models are currently running.
- Under **Metadata Output**, enable **Filename suggestions** to include a proposed slug in results, and turn on **Auto-rename** if you want Image Tagger to rename files on disk using those suggestions (collisions get `-1`, `-2`, etc.).
- Choose **Output mode**: *Embedded metadata* edits supported image formats in place, while *YAML sidecars* keeps the original files untouched and writes `.yaml` files next to them.
- Changes are saved automatically to `%APPDATA%\image_tagger\settings.yaml` on Windows or `~/.config/image_tagger/settings.yaml` on Linux/macOS.

### Command-line (headless) mode
```bash
python -m image_tagger \
  --headless \
  --input /path/to/images \
  --model caption.blip-base \
  --output-mode sidecar \
  --suggest-filenames \
  --auto-rename-files
```
Headless runs print a JSON summary with every caption, tag list, and destination path. This mode is handy for automation or server use.

## Tips and troubleshooting
- **Slow first run?** BLIP weight downloads or Ollama model loads can take several minutes. They are cached, so future runs are fast.
- **CPU vs GPU:** BLIP auto-detects CUDA/Metal. If you only have a CPU, expect longer processing times but the results are identical.
- **Need to start fresh?** Delete the settings file listed above; Image Tagger will recreate it with safe defaults.
- **Remote timeouts:** Ollama may take longer than 90 s to warm up. Increase **Remote timeout** under Settings if you see timeout errors.

## Developer guide
1. Install tooling plus the package in editable mode:
   ```bash
   make install-dev          # Equivalent to pip install -e .[blip] plus test/lint deps
   ```
2. Helpful commands (defined in the `Makefile`):
   - `make fmt` – format with `ruff format`.
   - `make lint` – static analysis via `ruff check`.
   - `make style` – legacy `pycodestyle`.
   - `make test` – run the pytest suite.
   - `make check` – lint + style + tests (CI default).
   - `make coverage` – run coverage and print a summary.
3. Project layout:
   ```
   image_tagger/
   ├── models/            # BLIP + Ollama implementations registered via ModelRegistry
   ├── services/          # High-level pipeline and metadata orchestration
   ├── gui/               # PySide6 desktop UI
   ├── io/                # Metadata writers and YAML sidecars
   ├── config.py          # Pydantic settings shared by GUI and CLI
   └── settings_store.py  # Cross-platform persistence helper
   ```
4. Adding new models:
   - Implement `TaggingModel` in `image_tagger/models/your_model.py`.
   - Register it with `ModelRegistry.register("your.id", YourModel)`.
   - List extra dependencies inside `pyproject.toml` under `[project.optional-dependencies]`.

With these steps, someone with basic command-line skills can install Image Tagger in an isolated Python environment, choose either local BLIP or Ollama-powered captions, and start tagging images in minutes.

## Licensing and acknowledgements
- Image Tagger itself is released under the MIT License (see `LICENSE`). Third-party libraries and models that ship with or are installed alongside the app are documented in `THIRD_PARTY_NOTICES.md` plus the companion files inside the `licenses/` directory.
- The desktop UI relies on PySide6/Qt for Python, which is licensed under the GNU LGPL v3.0. If you redistribute a packaged build, you must preserve Qt's license text and keep the Qt libraries relinkable (typically by shipping shared libraries).
- Local captioning uses Salesforce Research's BLIP checkpoints via Hugging Face Transformers. Cite Salesforce BLIP if you redistribute the model weights and keep their MIT license in your distributions.
- Remote captioning can talk to any Ollama-served multimodal model (LLaVA, Qwen-VL, MiniCPM-V, etc.). Each model and Ollama itself has its own license/usage policy—be sure your usage and redistribution comply with those upstream terms.

## Release workflow
1. **Set the version.** Edit `pyproject.toml` and bump `[project].version` to the number you intend to release. Commit this change (and anything else required) before building artifacts; release files should be created from clean, tagged commits.
2. **Run checks.** Execute `make check` (or `make test`, `make lint`, etc.) to verify the codebase is ready. Commit any fixes.
3. **Tag the release.** Create an annotated git tag such as `git tag -a v1.1.0 -m "Image Tagger 1.1.0"` and push it with `git push origin --tags`.
4. **Build distributables.** Run `make build` (it invokes `python -m build`) to generate both the source distribution and wheel in `dist/`. If you prefer not to use the Makefile you can directly run `python -m build`. The `build` package is included in the `dev` extra, so `make install-dev` installs everything required.
5. **Publish.** Attach the wheel (`dist/*.whl`) and source tarball (`dist/*.tar.gz`) to a GitHub release for the corresponding tag, or upload them to PyPI with `twine` if desired. These generated files are artifacts—do not commit them to git.
6. **Document the release.** In the GitHub Release notes (or CHANGELOG), summarize notable changes, requirements, and any manual steps (e.g., BLIP model downloads).
7. **Release notes tip.** For `v1.1.0`, highlight the new Ollama filename suggestions/auto-rename option, the CLI/GUI sinks for that feature, and remind users to enable the new metadata options if they want file renaming behavior.
