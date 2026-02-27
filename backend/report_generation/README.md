# DetectifAI Report Generation Module

Automatically generates professional forensic incident reports from detected surveillance events using a local LLM.

## 📋 Overview

This module takes already-processed event data (detections, timestamps, captions, keyframes) and uses a local instruction-tuned LLM to generate structured, professional reports exportable as PDF or HTML.

### Key Features

- **Offline Operation**: Uses local LLM (Qwen2.5-3B-Instruct or Phi-3-mini)
- **Deterministic Output**: No hallucinations - only uses provided data
- **Professional Reports**: Structured Markdown converted to PDF/HTML
- **Evidence Integration**: Embeds keyframes and face crops
- **Zero Cloud Dependencies**: Everything runs locally

## 🛠️ Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | 8+ cores |
| Disk Space | 5 GB | 10 GB |
| GPU (optional) | None | NVIDIA with CUDA |

### Software Requirements

1. **Python 3.9+** (already installed for DetectifAI)
2. **GTK3 Runtime** (for WeasyPrint PDF export on Windows)

## 📦 Installation

### Step 1: Install Python Dependencies

```bash
# Navigate to backend directory
cd backend

# Install report generation dependencies
pip install llama-cpp-python huggingface_hub jinja2 markdown weasyprint reportlab Pillow
```

**Note for GPU acceleration (optional):**
```bash
# For NVIDIA CUDA support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# For Windows with CUDA
set CMAKE_ARGS=-DLLAMA_CUBLAS=on
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Step 2: Install GTK3 (Required for PDF Export on Windows)

WeasyPrint requires GTK3 runtime. Choose one method:

**Option A: Using MSYS2 (Recommended)**
1. Download MSYS2 from: https://www.msys2.org/
2. Install and open MSYS2 terminal
3. Run: `pacman -S mingw-w64-x86_64-gtk3`
4. Add to PATH: `C:\msys64\mingw64\bin`

**Option B: Standalone GTK3**
1. Download from: https://github.com/nicothin/MSYS2-GTK-Windows
2. Extract to `C:\GTK3`
3. Add `C:\GTK3\bin` to system PATH

**Option C: Skip PDF (Use HTML only)**
- If GTK3 is problematic, use HTML export or the `SimplePDFExporter` (reportlab-based)

### Step 3: Download the LLM Model

The module will auto-download on first use, but you can pre-download:

```bash
# Run the download script
python -c "from report_generation.llm_engine import LLMEngine; e = LLMEngine(); e.download_model()"
```

**Or manually download:**

| Model | Size | License | Download |
|-------|------|---------|----------|
| **Qwen2.5-3B-Instruct** (Primary) | ~2 GB | Apache 2.0 | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) |
| **Phi-3-mini-4k-instruct** (Alt) | ~2.3 GB | MIT | [HuggingFace](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) |

Download the `q4_k_m.gguf` quantized version and place in:
```
backend/report_generation/models/qwen2.5-3b-instruct-q4_k_m.gguf
```

## 🚀 Usage

### Basic Usage

```python
from report_generation import ReportGenerator

# Initialize generator
generator = ReportGenerator()

# Generate report for a video
report = generator.generate_report(
    video_id="video_20240101_120000_abc123"
)

# Export as HTML
html_path = generator.export_html(report)
print(f"HTML report: {html_path}")

# Export as PDF
pdf_path = generator.export_pdf(report)
print(f"PDF report: {pdf_path}")
```

### With Time Range Filter

```python
from datetime import datetime

report = generator.generate_report(
    video_id="video_20240101_120000_abc123",
    time_range=(
        datetime(2024, 1, 1, 12, 0, 0),
        datetime(2024, 1, 1, 13, 0, 0)
    )
)
```

### Selective Sections

```python
# Generate only specific sections
report = generator.generate_report(
    video_id="video_123",
    include_sections=['header', 'executive_summary', 'timeline', 'conclusion']
)
```

### Without LLM (Fallback Mode)

If the LLM fails to load, the module automatically uses fallback templates:

```python
from report_generation.config import ReportConfig

# Explicitly disable LLM
config = ReportConfig()
config.llm.n_gpu_layers = 0  # CPU only
config.llm.n_threads = 2     # Reduce for slow systems

generator = ReportGenerator(config)
```

## 📁 Module Structure

```
report_generation/
├── __init__.py              # Package initialization
├── config.py                # Configuration (LLM, paths, settings)
├── llm_engine.py            # LLM loading and inference
├── prompt_templates.py      # Prompt engineering templates
├── data_collector.py        # MongoDB data collection
├── report_builder.py        # Main orchestration
├── html_renderer.py         # Jinja2 HTML generation
├── pdf_exporter.py          # WeasyPrint/reportlab PDF export
├── templates/               # HTML/CSS templates
│   ├── report_base.html
│   └── report_styles.css
├── models/                  # LLM model files (.gitignored)
│   └── .gitkeep
└── README.md
```

## 📊 Report Sections

| Section | Description | LLM Generated |
|---------|-------------|---------------|
| **Header** | Report ID, metadata, video info | ❌ |
| **Executive Summary** | Overview of findings | ✅ |
| **Incident Timeline** | Chronological event list | ✅ |
| **Evidence Catalog** | Keyframes and face crops | ✅ |
| **Observations** | Pattern analysis | ✅ |
| **Conclusion** | Summary and recommendations | ✅ |

## ⚙️ Configuration

Edit `config.py` or pass custom config:

```python
from report_generation.config import ReportConfig, LLMConfig

# Custom LLM settings
llm_config = LLMConfig(
    n_threads=8,           # More CPU threads
    n_gpu_layers=35,       # Offload to GPU
    temperature=0.1,       # Low for determinism
    max_tokens=2048        # Max output length
)

config = ReportConfig(llm=llm_config)
config.organization_name = "My Security Company"
config.report_classification = "INTERNAL"

generator = ReportGenerator(config)
```

## 🔧 Troubleshooting

### LLM Won't Load

```
Error: Model not found
```
**Solution:** Download the model manually or check path in `config.py`

### PDF Export Fails on Windows

```
OSError: cannot load library 'gobject-2.0-0'
```
**Solution:** Install GTK3 runtime and add to PATH (see Step 2)

### Out of Memory

```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Set `n_gpu_layers=0` for CPU-only
- Use smaller context: `n_ctx=2048`
- Close other applications

### Slow Generation

**Solutions:**
- Increase `n_threads` (up to CPU core count)
- Enable GPU with `n_gpu_layers > 0`
- Use smaller model (Phi-3 instead of Qwen)

## 📝 API Reference

### ReportGenerator

```python
class ReportGenerator:
    def __init__(config: ReportConfig = None)
    def initialize() -> bool
    def generate_report(
        video_id: str,
        time_range: Tuple[datetime, datetime] = None,
        include_sections: List[str] = None
    ) -> GeneratedReport
    def export_html(report: GeneratedReport, output_path: str = None) -> str
    def export_pdf(report: GeneratedReport, output_path: str = None) -> str
```

### GeneratedReport

```python
@dataclass
class GeneratedReport:
    report_id: str
    video_id: str
    title: str
    generated_at: datetime
    time_range: Tuple[datetime, datetime]
    sections: List[ReportSection]
    metadata: Dict[str, Any]
    statistics: Dict[str, Any]
```

## 🔒 Security Notes

1. **Local Processing**: All data stays on your machine
2. **No Cloud Calls**: LLM runs entirely offline
3. **Fact-Based**: Reports only contain provided data
4. **Confidential Marking**: Reports are marked CONFIDENTIAL by default

## 📄 License

This module is part of DetectifAI and follows the project license.

The recommended LLM models have the following licenses:
- **Qwen2.5**: Apache 2.0
- **Phi-3**: MIT

Both are free for commercial use.
