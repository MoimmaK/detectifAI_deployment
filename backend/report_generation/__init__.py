"""
DetectifAI Report Generation Module

Automatically generates professional forensic incident reports from 
detected surveillance events using a local LLM.

Features:
- Offline operation with local LLM (Qwen2.5-3B-Instruct)
- Structured report generation (Markdown/JSON)
- PDF and HTML export
- Evidence image embedding
- Deterministic, fact-based output (no hallucinations)

Usage:
    from report_generation import ReportGenerator
    
    generator = ReportGenerator()
    report = generator.generate_report(
        video_id="video_20240101_120000_abc123",
        time_range=("2024-01-01 12:00:00", "2024-01-01 13:00:00")
    )
    
    # Export as PDF
    generator.export_pdf(report, "incident_report.pdf")
    
    # Export as HTML
    generator.export_html(report, "incident_report.html")
"""

from .report_builder import ReportGenerator
from .config import ReportConfig

__all__ = ['ReportGenerator', 'ReportConfig']
__version__ = '1.0.0'
