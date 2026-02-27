"""
HTML Renderer for Report Generation

Renders GeneratedReport objects to HTML using Jinja2 templates.
Handles image embedding, Markdown conversion, and styling.
"""

import os
import base64
import logging
import markdown
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from .config import ReportConfig

logger = logging.getLogger(__name__)


class HTMLRenderer:
    """
    Renders reports to HTML format using Jinja2 templates.
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize the HTML renderer.
        
        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()
        self._setup_jinja()
        self._setup_markdown()
    
    def _setup_jinja(self):
        """Setup Jinja2 environment."""
        try:
            from jinja2 import Environment, FileSystemLoader, select_autoescape
            
            # Check if templates directory exists, create default template if not
            templates_dir = self.config.templates_dir
            if not os.path.exists(os.path.join(templates_dir, 'report_base.html')):
                self._create_default_template()
            
            self.jinja_env = Environment(
                loader=FileSystemLoader(templates_dir),
                autoescape=select_autoescape(['html', 'xml'])
            )
            
            # Add custom filters
            self.jinja_env.filters['markdown'] = self._markdown_filter
            self.jinja_env.filters['format_datetime'] = self._format_datetime
            
        except ImportError:
            logger.error("Jinja2 not installed. Install with: pip install Jinja2")
            raise
    
    def _setup_markdown(self):
        """Setup Markdown processor."""
        self.md = markdown.Markdown(
            extensions=['tables', 'fenced_code', 'nl2br', 'toc'],
            output_format='html5'
        )
    
    def _markdown_filter(self, text: str) -> str:
        """Jinja2 filter to convert Markdown to HTML."""
        if not text:
            return ''
        self.md.reset()
        return self.md.convert(text)
    
    def _format_datetime(self, dt, format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
        """Jinja2 filter to format datetime objects."""
        if isinstance(dt, datetime):
            return dt.strftime(format_str)
        return str(dt)
    
    def _create_default_template(self):
        """Create default HTML template if not exists."""
        template_path = os.path.join(self.config.templates_dir, 'report_base.html')
        css_path = os.path.join(self.config.templates_dir, 'report_styles.css')
        
        os.makedirs(self.config.templates_dir, exist_ok=True)
        
        # Default HTML template with improved structure
        html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report.title }}</title>
    <style>
        {% include 'report_styles.css' %}
    </style>
</head>
<body>
    <div class="report-container">
        <!-- Report Header -->
        <header class="report-header">
            <div class="logo">
                <h1>🛡️ DetectifAI</h1>
                <p class="subtitle">AI-Powered Surveillance Analysis Report</p>
            </div>
            <div class="report-meta">
                <span class="classification {{ report.metadata.classification|default('CONFIDENTIAL')|lower }}">
                    {{ report.metadata.classification|default('CONFIDENTIAL') }}
                </span>
            </div>
        </header>

        <!-- Report Content -->
        <main class="report-content">
            {% for section in report.sections|sort(attribute='order') %}
            <section class="report-section section-{{ section.name }}" id="section-{{ section.name }}">
                <div class="section-content">
                    {{ section.content|markdown|safe }}
                </div>
                
                {% if section.images %}
                <div class="evidence-gallery">
                    <h3 class="gallery-title">Evidence Images</h3>
                    <div class="gallery-grid">
                        {% for img in section.images[:max_images] %}
                        <figure class="evidence-item">
                            {% if img.embedded_data %}
                            <img src="data:image/jpeg;base64,{{ img.embedded_data }}" 
                                 alt="{{ img.caption|default('Evidence image') }}"
                                 class="evidence-image">
                            {% elif img.path %}
                            <img src="{{ img.path }}" 
                                 alt="{{ img.caption|default('Evidence image') }}"
                                 class="evidence-image">
                            {% elif img.url %}
                            <img src="{{ img.url }}" 
                                 alt="{{ img.caption|default('Evidence image') }}"
                                 class="evidence-image">
                            {% else %}
                            <div class="image-placeholder">
                                <span>📷 Image: {{ img.id }}</span>
                            </div>
                            {% endif %}
                            <figcaption>{{ img.caption|default('Evidence ' + loop.index|string) }}</figcaption>
                        </figure>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
            </section>
            <div class="section-divider"></div>
            {% endfor %}
        </main>

        <!-- Report Footer -->
        <footer class="report-footer">
            <div class="footer-content">
                <div class="footer-info">
                    <p><strong>Report ID:</strong> {{ report.report_id }}</p>
                    <p><strong>Generated:</strong> {{ report.generated_at|format_datetime }}</p>
                </div>
                <p class="disclaimer">
                    ⚠️ This report was automatically generated by DetectifAI using AI analysis. 
                    All findings should be verified by qualified security personnel before taking action.
                </p>
            </div>
        </footer>
    </div>
</body>
</html>'''

        # Default CSS styles with improved readability
        css_styles = '''/* DetectifAI Report Styles - Enhanced Readability */
:root {
    --primary-color: #1a365d;
    --secondary-color: #2d3748;
    --accent-color: #3182ce;
    --danger-color: #e53e3e;
    --warning-color: #dd6b20;
    --success-color: #38a169;
    --bg-color: #ffffff;
    --text-color: #2d3748;
    --border-color: #e2e8f0;
    --section-bg: #f8fafc;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 12pt;
    line-height: 1.8;
    color: var(--text-color);
    background-color: #f7fafc;
}

.report-container {
    max-width: 210mm;
    margin: 20px auto;
    background: var(--bg-color);
    box-shadow: 0 4px 30px rgba(0,0,0,0.15);
}

/* Header Styles */
.report-header {
    background: linear-gradient(135deg, var(--primary-color) 0%, #2c5282 100%);
    color: white;
    padding: 40px 50px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 4px solid var(--accent-color);
}

.report-header .logo h1 {
    font-size: 32pt;
    margin-bottom: 8px;
    font-weight: 700;
}

.report-header .subtitle {
    font-size: 12pt;
    opacity: 0.95;
    font-weight: 300;
}

.classification {
    padding: 10px 20px;
    border-radius: 6px;
    font-weight: bold;
    text-transform: uppercase;
    font-size: 10pt;
    letter-spacing: 1px;
}

.classification.confidential {
    background: var(--danger-color);
}

.classification.internal {
    background: var(--warning-color);
}

.classification.public {
    background: var(--success-color);
}

/* Content Styles */
.report-content {
    padding: 50px;
}

.report-section {
    margin-bottom: 50px;
    page-break-inside: avoid;
}

.section-content {
    background: var(--section-bg);
    padding: 30px;
    border-radius: 8px;
    border-left: 4px solid var(--accent-color);
}

.section-divider {
    height: 2px;
    background: linear-gradient(to right, transparent, var(--border-color), transparent);
    margin: 40px 0;
}

h1, h2, h3, h4 {
    color: var(--primary-color);
    margin-top: 30px;
    margin-bottom: 20px;
    font-weight: 600;
}

h1 { 
    font-size: 28pt; 
    border-bottom: 3px solid var(--accent-color); 
    padding-bottom: 15px;
    margin-top: 0;
}

h2 { 
    font-size: 20pt; 
    border-bottom: 2px solid var(--border-color); 
    padding-bottom: 12px;
    margin-top: 0;
}

h3 { 
    font-size: 16pt;
    color: var(--secondary-color);
}

h4 { 
    font-size: 13pt;
    color: var(--secondary-color);
}

p {
    margin-bottom: 16px;
    text-align: justify;
}

strong {
    color: var(--primary-color);
    font-weight: 600;
}

/* Table Styles - Enhanced for Professional Reports */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 25px 0;
    font-size: 11pt;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border-radius: 8px;
    overflow: hidden;
    background: white;
}

thead {
    background: var(--primary-color);
    color: white;
}

th, td {
    border: 1px solid var(--border-color);
    padding: 14px 16px;
    text-align: left;
    vertical-align: top;
}

th {
    background: var(--primary-color);
    color: white;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 10pt;
    letter-spacing: 0.5px;
    position: sticky;
    top: 0;
    z-index: 10;
}

tbody tr:nth-child(odd) {
    background: white;
}

tbody tr:nth-child(even) {
    background: #f8fafc;
}

tbody tr:hover {
    background: #edf2f7;
    transition: background 0.2s ease;
}

td:first-child {
    font-weight: 600;
    color: var(--secondary-color);
}

/* Table caption */
table caption {
    caption-side: top;
    padding: 10px;
    font-weight: 600;
    color: var(--primary-color);
    text-align: left;
    font-size: 12pt;
}

/* Evidence Gallery */
.evidence-gallery {
    margin: 30px 0;
    padding: 25px;
    background: white;
    border-radius: 8px;
    border: 1px solid var(--border-color);
}

.gallery-title {
    font-size: 14pt;
    color: var(--primary-color);
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--accent-color);
}

.gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 25px;
}

.evidence-item {
    border: 2px solid var(--border-color);
    border-radius: 10px;
    overflow: hidden;
    background: #f8fafc;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.evidence-item:hover {
    transform: translateY(-4px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
}

.evidence-image {
    width: 100%;
    height: 180px;
    object-fit: cover;
    border-bottom: 2px solid var(--border-color);
}

.image-placeholder {
    width: 100%;
    height: 180px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%);
    color: #718096;
    font-size: 11pt;
    border-bottom: 2px solid var(--border-color);
}

.evidence-item figcaption {
    padding: 12px;
    font-size: 10pt;
    color: #4a5568;
    background: white;
    font-weight: 500;
    text-align: center;
    line-height: 1.4;
    border-top: 1px solid var(--border-color);
}

.evidence-item figcaption::before {
    content: "📷 ";
    color: var(--accent-color);
}

/* Image counter for evidence */
.evidence-gallery {
    counter-reset: evidence-counter;
}

.evidence-item {
    counter-increment: evidence-counter;
}

.evidence-item figcaption::before {
    content: "Evidence #" counter(evidence-counter) ": ";
    font-weight: 600;
    color: var(--primary-color);
    display: block;
    margin-bottom: 4px;
}

/* Lists - Enhanced Formatting */
ul, ol {
    margin: 16px 0;
    padding-left: 30px;
}

ul {
    list-style-type: disc;
}

ul ul {
    list-style-type: circle;
    margin-top: 8px;
}

ol {
    list-style-type: decimal;
}

ol ol {
    list-style-type: lower-alpha;
    margin-top: 8px;
}

li {
    margin-bottom: 10px;
    line-height: 1.6;
    padding-left: 8px;
}

li::marker {
    color: var(--accent-color);
    font-weight: 600;
}

/* Blockquotes for important notes */
blockquote {
    margin: 20px 0;
    padding: 20px 25px;
    border-left: 4px solid var(--accent-color);
    background: #f8fafc;
    border-radius: 0 8px 8px 0;
    font-style: italic;
    color: var(--secondary-color);
}

blockquote p {
    margin-bottom: 0;
}

blockquote strong {
    font-style: normal;
    color: var(--primary-color);
}

/* Code blocks */
code {
    background: #edf2f7;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 10pt;
    font-family: 'Courier New', monospace;
    color: #c53030;
}

pre {
    background: #2d3748;
    color: #e2e8f0;
    padding: 20px;
    border-radius: 8px;
    overflow-x: auto;
    font-size: 10pt;
    margin: 20px 0;
    line-height: 1.4;
}

pre code {
    background: transparent;
    padding: 0;
    color: inherit;
}

/* Definition Lists for Metadata */
dl {
    margin: 20px 0;
    display: grid;
    grid-template-columns: max-content auto;
    gap: 12px 20px;
}

dt {
    font-weight: 600;
    color: var(--primary-color);
    text-align: right;
}

dt::after {
    content: ":";
}

dd {
    margin: 0;
    color: var(--text-color);
}

/* Horizontal spacing improvements */
.section-content > *:first-child {
    margin-top: 0;
}

.section-content > *:last-child {
    margin-bottom: 0;
}

/* Footer Styles */
.report-footer {
    background: var(--section-bg);
    padding: 30px 50px;
    border-top: 3px solid var(--accent-color);
    font-size: 10pt;
    color: #718096;
}

.footer-content {
    max-width: 100%;
}

.footer-info {
    margin-bottom: 20px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border-color);
}

.footer-info p {
    margin-bottom: 8px;
    text-align: left;
}

.disclaimer {
    margin-top: 20px;
    font-style: italic;
    padding: 15px;
    background: #fff3cd;
    border-left: 4px solid var(--warning-color);
    border-radius: 4px;
    color: #856404;
    text-align: left;
}

/* Print/PDF Styles - Professional Multi-Page Layout */
@media print, (min-width: 0) {
    body {
        background: white;
        font-size: 11pt;
    }
    
    .report-container {
        box-shadow: none;
        max-width: 100%;
        margin: 0;
    }
    
    .report-header {
        page-break-after: avoid;
    }
    
    .report-section {
        page-break-inside: avoid;
        orphans: 3;
        widows: 3;
    }
    
    h1, h2, h3, h4, h5, h6 {
        page-break-after: avoid;
        page-break-inside: avoid;
    }
    
    .section-divider {
        page-break-before: avoid;
        page-break-after: avoid;
    }
    
    table {
        page-break-inside: avoid;
    }
    
    thead {
        display: table-header-group;
    }
    
    tfoot {
        display: table-footer-group;
    }
    
    .evidence-gallery {
        page-break-inside: avoid;
    }
    
    .evidence-item {
        page-break-inside: avoid;
    }
    
    /* Force proper page breaks for long sections */
    .section-metadata,
    .section-details,
    .section-summary {
        page-break-inside: avoid;
    }
    
    /* Ensure images don't break awkwardly */
    img {
        max-width: 100%;
        page-break-inside: avoid;
    }
    
    a {
        text-decoration: none;
        color: var(--primary-color);
    }
    
    /* Print URLs for important links */
    a[href^="http"]::after {
        content: " (" attr(href) ")";
        font-size: 8pt;
        color: #718096;
    }
}

/* Threat Level Indicators */
.threat-critical { 
    color: #c53030; 
    font-weight: bold;
    background: #fff5f5;
    padding: 2px 6px;
    border-radius: 3px;
}

.threat-high { 
    color: #dd6b20; 
    font-weight: bold;
    background: #fffaf0;
    padding: 2px 6px;
    border-radius: 3px;
}

.threat-medium { 
    color: #d69e2e;
    background: #fffff0;
    padding: 2px 6px;
    border-radius: 3px;
}

.threat-low { 
    color: #38a169;
    background: #f0fff4;
    padding: 2px 6px;
    border-radius: 3px;
}

/* Horizontal rules */
hr {
    border: none;
    border-top: 2px solid var(--border-color);
    margin: 30px 0;
}'''

        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write(css_styles)
        
        logger.info(f"Created default templates at {self.config.templates_dir}")
    
    def _embed_images(self, report) -> None:
        """
        Embed images as base64 in the report.
        
        Args:
            report: GeneratedReport object (modified in place)
        """
        for section in report.sections:
            for img in section.images:
                if img.get('path') and os.path.exists(img['path']):
                    try:
                        with open(img['path'], 'rb') as f:
                            img_data = f.read()
                        img['embedded_data'] = base64.b64encode(img_data).decode('utf-8')
                    except Exception as e:
                        logger.warning(f"Failed to embed image {img['path']}: {e}")
    
    def _process_image_placeholders(self, content: str, data_collector) -> str:
        """
        Replace image placeholders with actual image references.
        
        Args:
            content: Report content with placeholders
            data_collector: DataCollector instance
            
        Returns:
            Processed content
        """
        import re
        
        # Replace [[IMAGE:id]] placeholders
        def replace_image(match):
            img_id = match.group(1)
            path = data_collector.get_image_path(img_id, 'keyframe') if data_collector else None
            # Check for URL or local path
            if path and (path.startswith('http') or os.path.exists(path)):
                return f'![Keyframe {img_id}]({path})'
            return f'*[Image {img_id} not available]*'
        
        # Replace [[FACE:id]] placeholders
        def replace_face(match):
            face_id = match.group(1)
            path = data_collector.get_image_path(face_id, 'face') if data_collector else None
            if path and (path.startswith('http') or os.path.exists(path)):
                return f'![Face {face_id}]({path})'
            return f'*[Face {face_id} not available]*'
        
        content = re.sub(r'\[\[IMAGE:([^\]]+)\]\]', replace_image, content)
        content = re.sub(r'\[\[FACE:([^\]]+)\]\]', replace_face, content)
        
        return content
    
    def _cleanup_remaining_placeholders(self, report) -> None:
        """
        Remove any remaining [[IMAGE:...]] and [[FACE:...]] placeholders
        from all section content. These are replaced with italic 'not available'
        messages so no raw placeholder text appears in the final report.
        """
        import re
        for section in report.sections:
            if not section.content:
                continue
            section.content = re.sub(
                r'\[\[IMAGE:[^\]]+\]\]',
                '*[Image not available]*',
                section.content
            )
            section.content = re.sub(
                r'\[\[FACE:[^\]]+\]\]',
                '*[Face image not available]*',
                section.content
            )
    
    def render(
        self,
        report,
        output_path: Optional[str] = None,
        embed_images: bool = True
    ) -> str:
        """
        Render report to HTML.
        
        Args:
            report: GeneratedReport object
            output_path: Output file path (auto-generated if None)
            embed_images: Whether to embed images as base64
            
        Returns:
            Path to generated HTML file
        """
        logger.info(f"Rendering HTML report: {report.report_id}")
        
        # Embed images if requested
        if embed_images and self.config.include_evidence_images:
            self._embed_images(report)
        
        # Clean up any remaining [[IMAGE:...]] and [[FACE:...]] placeholders in section content
        self._cleanup_remaining_placeholders(report)
        
        # Generate output path if not provided
        if not output_path:
            output_path = os.path.join(
                self.config.output_dir,
                f"{report.report_id}.html"
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Render template
        try:
            template = self.jinja_env.get_template('report_base.html')
            
            html_content = template.render(
                report=report,
                config=self.config,
                max_images=self.config.max_images_per_event * 10,
                generated_at=datetime.utcnow()
            )
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"✅ HTML report saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error rendering HTML: {e}")
            raise
