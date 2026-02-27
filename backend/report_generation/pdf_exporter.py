"""
PDF Exporter for Report Generation

Exports HTML reports to PDF using WeasyPrint.
Handles page setup, headers/footers, and print styling.
"""

import os
import logging
from datetime import datetime
from typing import Optional

from .config import ReportConfig
from .html_renderer import HTMLRenderer

logger = logging.getLogger(__name__)


class PDFExporter:
    """
    Exports reports to PDF format using WeasyPrint.
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize the PDF exporter.
        
        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()
        self.html_renderer = HTMLRenderer(config)
        self._weasyprint_available = self._check_weasyprint()
    
    def _check_weasyprint(self) -> bool:
        """Check if WeasyPrint is available."""
        try:
            import weasyprint
            return True
        except ImportError:
            logger.warning(
                "WeasyPrint not installed. PDF export will not be available.\n"
                "Install with: pip install weasyprint\n"
                "Note: WeasyPrint requires GTK libraries. On Windows, install GTK3:\n"
                "https://github.com/nicothin/MSYS2-GTK-Windows"
            )
            return False
        except OSError as e:
            logger.warning(
                f"WeasyPrint dependencies not found: {e}\n"
                "On Windows, GTK3 runtime is required. Install from:\n"
                "https://github.com/nicothin/MSYS2-GTK-Windows"
            )
            return False
    
    def export(
        self,
        report,
        output_path: Optional[str] = None,
        embed_images: bool = True
    ) -> str:
        """
        Export report to PDF.
        
        Args:
            report: GeneratedReport object
            output_path: Output file path (auto-generated if None)
            embed_images: Whether to embed images
            
        Returns:
            Path to generated PDF file
        """
        if not self._weasyprint_available:
            raise RuntimeError(
                "WeasyPrint is not available. Cannot export to PDF.\n"
                "Install with: pip install weasyprint"
            )
        
        logger.info(f"Exporting PDF report: {report.report_id}")
        
        # Generate output path if not provided
        if not output_path:
            output_path = os.path.join(
                self.config.output_dir,
                f"{report.report_id}.pdf"
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # First render to HTML
        html_path = output_path.replace('.pdf', '.html')
        self.html_renderer.render(report, html_path, embed_images)
        
        # Convert to PDF using WeasyPrint
        try:
            from weasyprint import HTML, CSS
            from weasyprint.text.fonts import FontConfiguration
            
            font_config = FontConfiguration()
            
            # Additional PDF-specific CSS for professional formatting
            pdf_css = CSS(string='''
                @page {
                    size: A4;
                    margin: 20mm 20mm 25mm 20mm;
                    
                    @top-center {
                        content: "DETECTIFAI INCIDENT REPORT | CONFIDENTIAL";
                        font-size: 8pt;
                        font-weight: bold;
                        color: #1a365d;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                        border-bottom: 1px solid #e2e8f0;
                        padding-bottom: 3mm;
                    }
                    
                    @bottom-left {
                        content: "Report ID: ''' + report.report_id + '''";
                        font-size: 8pt;
                        color: #718096;
                        font-family: monospace;
                    }
                    
                    @bottom-center {
                        content: "Generated: ''' + datetime.now().strftime('%Y-%m-%d %H:%M UTC') + '''";
                        font-size: 7pt;
                        color: #a0aec0;
                    }
                    
                    @bottom-right {
                        content: "Page " counter(page) " of " counter(pages);
                        font-size: 8pt;
                        color: #718096;
                        font-weight: bold;
                    }
                }
                
                @page :first {
                    @top-center { content: none; }
                    margin-top: 15mm;
                }
                
                /* Better page break control */
                h1, h2, h3 {
                    page-break-after: avoid;
                    page-break-inside: avoid;
                }
                
                table {
                    page-break-inside: avoid;
                }
                
                .evidence-gallery {
                    page-break-inside: avoid;
                }
                
                .report-section {
                    orphans: 3;
                    widows: 3;
                }
                
                /* Ensure good typography */
                body {
                    font-size: 10pt;
                    line-height: 1.6;
                }
                
                p, li {
                    text-align: justify;
                    hyphens: auto;
                }
            ''', font_config=font_config)
            
            # Generate PDF
            html = HTML(filename=html_path)
            html.write_pdf(
                output_path,
                stylesheets=[pdf_css],
                font_config=font_config
            )
            
            logger.info(f"✅ PDF report saved to: {output_path}")
            
            # Optionally clean up intermediate HTML
            # os.remove(html_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting PDF: {e}")
            raise
    
    def export_from_html(
        self,
        html_path: str,
        output_path: Optional[str] = None,
        report_id: str = "UNKNOWN"
    ) -> str:
        """
        Export an existing HTML file to PDF.
        
        Args:
            html_path: Path to HTML file
            output_path: Output PDF path
            report_id: Report ID for footer
            
        Returns:
            Path to generated PDF
        """
        if not self._weasyprint_available:
            raise RuntimeError("WeasyPrint is not available")
        
        if not os.path.exists(html_path):
            raise FileNotFoundError(f"HTML file not found: {html_path}")
        
        if not output_path:
            output_path = html_path.replace('.html', '.pdf')
        
        try:
            from weasyprint import HTML, CSS
            from weasyprint.text.fonts import FontConfiguration
            
            font_config = FontConfiguration()
            
            pdf_css = CSS(string=f'''
                @page {{
                    size: A4;
                    margin: 20mm;
                    
                    @bottom-left {{
                        content: "Report ID: {report_id}";
                        font-size: 8pt;
                        color: #718096;
                    }}
                    
                    @bottom-right {{
                        content: "Page " counter(page);
                        font-size: 8pt;
                        color: #718096;
                    }}
                }}
            ''', font_config=font_config)
            
            html = HTML(filename=html_path)
            html.write_pdf(output_path, stylesheets=[pdf_css], font_config=font_config)
            
            logger.info(f"✅ PDF exported to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting PDF from HTML: {e}")
            raise


class SimplePDFExporter:
    """
    Fallback PDF exporter using reportlab (simpler, fewer dependencies).
    Use this if WeasyPrint installation is problematic.
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize simple PDF exporter."""
        self.config = config or ReportConfig()
        self._check_reportlab()
    
    def _check_reportlab(self) -> bool:
        """Check if reportlab is available."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate
            return True
        except ImportError:
            logger.warning(
                "reportlab not installed. Install with: pip install reportlab"
            )
            return False
    
    def _convert_inline_markdown(self, text: str) -> str:
        """Convert inline markdown (bold, italic, links) to ReportLab XML tags."""
        import re
        # Remove image markdown references (they are handled separately)
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[\1]', text)
        # Convert markdown links to just text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # Convert **bold** to <b>bold</b>
        text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
        # Convert *italic* to <i>italic</i>
        text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)
        # Escape XML special characters (but not our tags)
        text = text.replace('&', '&amp;')
        text = text.replace('<b>', '\x00b\x00').replace('</b>', '\x00/b\x00')
        text = text.replace('<i>', '\x00i\x00').replace('</i>', '\x00/i\x00')
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        text = text.replace('\x00b\x00', '<b>').replace('\x00/b\x00', '</b>')
        text = text.replace('\x00i\x00', '<i>').replace('\x00/i\x00', '</i>')
        # Strip remaining placeholder markers
        text = re.sub(r'\[\[IMAGE:[^\]]*\]\]', '[Image not available]', text)
        text = re.sub(r'\[\[FACE:[^\]]*\]\]', '[Face image not available]', text)
        return text
    
    def _parse_markdown_table(self, lines):
        """Parse markdown table lines into a list of rows (list of cells)."""
        rows = []
        for line in lines:
            line = line.strip()
            if line.startswith('|') and line.endswith('|'):
                cells = [c.strip() for c in line[1:-1].split('|')]
                # Skip separator rows (e.g., | --- | --- |)
                if all(set(c.strip()) <= {'-', ':', ' '} for c in cells):
                    continue
                rows.append(cells)
            elif '|' in line:
                cells = [c.strip() for c in line.split('|')]
                cells = [c for c in cells if c]
                if all(set(c.strip()) <= {'-', ':', ' '} for c in cells):
                    continue
                if cells:
                    rows.append(cells)
        return rows
    
    def _download_image(self, url: str, max_width: float = 400, max_height: float = 250):
        """Download an image from URL and return a ReportLab Image element."""
        try:
            import urllib.request
            import tempfile
            from reportlab.platypus import Image
            from reportlab.lib.units import mm
            
            # Download to temp file
            tmp_fd, tmp_path = tempfile.mkstemp(suffix='.jpg')
            os.close(tmp_fd)
            
            req = urllib.request.Request(url, headers={'User-Agent': 'DetectifAI-Report/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                with open(tmp_path, 'wb') as f:
                    f.write(response.read())
            
            # Create Image element with proper sizing
            img = Image(tmp_path)
            # Scale to fit within max dimensions while maintaining aspect ratio
            iw, ih = img.drawWidth, img.drawHeight
            if iw > 0 and ih > 0:
                ratio = min(max_width / iw, max_height / ih, 1.0)
                img.drawWidth = iw * ratio
                img.drawHeight = ih * ratio
            
            return img
        except Exception as e:
            logger.debug(f"Could not download image from {url}: {e}")
            return None
    
    def _parse_section_content(self, content: str, styles: dict):
        """Parse markdown content into a list of ReportLab flowable elements."""
        import re
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        
        elements = []
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].rstrip()
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                i += 1
                continue
            
            # Horizontal rule
            if stripped in ('---', '***', '___'):
                elements.append(Spacer(1, 6))
                elements.append(HRFlowable(
                    width="100%", thickness=1.5,
                    color=colors.HexColor('#e2e8f0'),
                    spaceBefore=6, spaceAfter=6
                ))
                i += 1
                continue
            
            # Headings
            if stripped.startswith('# ') and not stripped.startswith('## '):
                text = self._convert_inline_markdown(stripped[2:].strip())
                elements.append(Spacer(1, 12))
                elements.append(Paragraph(text, styles['ReportTitle']))
                elements.append(Spacer(1, 8))
                i += 1
                continue
            
            if stripped.startswith('## '):
                text = self._convert_inline_markdown(stripped[3:].strip())
                elements.append(Spacer(1, 10))
                elements.append(Paragraph(text, styles['SectionTitle']))
                elements.append(Spacer(1, 6))
                i += 1
                continue
            
            if stripped.startswith('### '):
                text = self._convert_inline_markdown(stripped[4:].strip())
                elements.append(Spacer(1, 8))
                elements.append(Paragraph(text, styles['SubsectionTitle']))
                elements.append(Spacer(1, 4))
                i += 1
                continue
            
            # Image references: ![alt](url)
            img_match = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)\s*$', stripped)
            if img_match:
                alt_text = img_match.group(1)
                img_url = img_match.group(2)
                img_element = self._download_image(img_url)
                if img_element:
                    elements.append(Spacer(1, 4))
                    elements.append(img_element)
                    if alt_text:
                        elements.append(Paragraph(
                            f"<i>{self._convert_inline_markdown(alt_text)}</i>",
                            styles['ImageCaption']
                        ))
                    elements.append(Spacer(1, 6))
                else:
                    elements.append(Paragraph(
                        f"<i>[{alt_text or 'Image'} — could not be loaded]</i>",
                        styles['ImageCaption']
                    ))
                i += 1
                continue
            
            # Table (collect consecutive lines with |)
            if '|' in stripped and (stripped.startswith('|') or stripped.count('|') >= 2):
                table_lines = []
                while i < len(lines) and ('|' in lines[i].strip()):
                    table_lines.append(lines[i])
                    i += 1
                
                rows = self._parse_markdown_table(table_lines)
                if rows and len(rows) >= 1:
                    # Convert cells to Paragraphs for text wrapping
                    col_count = max(len(r) for r in rows)
                    # Normalize all rows to same column count
                    for r_idx in range(len(rows)):
                        while len(rows[r_idx]) < col_count:
                            rows[r_idx].append('')
                    
                    table_data = []
                    for r_idx, row in enumerate(rows):
                        table_row = []
                        for cell in row:
                            cell_text = self._convert_inline_markdown(cell)
                            style_name = 'TableHeader' if r_idx == 0 else 'TableCell'
                            table_row.append(Paragraph(cell_text, styles[style_name]))
                        table_data.append(table_row)
                    
                    # Calculate column widths (distribute evenly)
                    available_width = 155 * mm  # A4 width minus margins
                    col_widths = [available_width / col_count] * col_count
                    
                    table = Table(table_data, colWidths=col_widths, repeatRows=1)
                    table.setStyle(TableStyle([
                        # Header row
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a365d')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        # Body rows
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, -1), 9),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
                        # Grid
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                        ('TOPPADDING', (0, 0), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                        ('LEFTPADDING', (0, 0), (-1, -1), 8),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ]))
                    elements.append(Spacer(1, 6))
                    elements.append(table)
                    elements.append(Spacer(1, 8))
                continue
            
            # Bullet list items
            if stripped.startswith('- ') or stripped.startswith('* '):
                bullet_text = self._convert_inline_markdown(stripped[2:].strip())
                elements.append(Paragraph(
                    f"\u2022  {bullet_text}",
                    styles['BulletItem']
                ))
                elements.append(Spacer(1, 3))
                i += 1
                continue
            
            # Numbered list items
            num_match = re.match(r'^(\d+)\.\s+(.*)', stripped)
            if num_match:
                num = num_match.group(1)
                item_text = self._convert_inline_markdown(num_match.group(2).strip())
                elements.append(Paragraph(
                    f"{num}.  {item_text}",
                    styles['BulletItem']
                ))
                elements.append(Spacer(1, 3))
                i += 1
                continue
            
            # Regular paragraph — collect consecutive non-special lines
            para_lines = [stripped]
            i += 1
            while i < len(lines):
                next_stripped = lines[i].strip()
                if not next_stripped:
                    i += 1
                    break
                # Stop if next line is a special element
                if (next_stripped.startswith('#') or next_stripped.startswith('- ') or
                    next_stripped.startswith('* ') or next_stripped.startswith('|') or
                    next_stripped in ('---', '***', '___') or
                    re.match(r'^\d+\.\s+', next_stripped) or
                    re.match(r'^!\[', next_stripped)):
                    break
                para_lines.append(next_stripped)
                i += 1
            
            para_text = ' '.join(para_lines)
            para_text = self._convert_inline_markdown(para_text)
            if para_text.strip():
                elements.append(Paragraph(para_text, styles['BodyText']))
                elements.append(Spacer(1, 6))
        
        return elements
    
    def export(self, report, output_path: Optional[str] = None) -> str:
        """
        Export report to PDF using reportlab with proper formatting.
        
        Handles markdown headings, tables, bold text, bullet lists,
        images from URLs, and proper section structure.
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import mm
            from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                PageBreak, HRFlowable
            )
            from reportlab.lib import colors
            
        except ImportError:
            raise RuntimeError("reportlab is not installed")
        
        if not output_path:
            output_path = os.path.join(
                self.config.output_dir,
                f"{report.report_id}.pdf"
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create document with page numbering
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=25*mm,
            bottomMargin=25*mm,
            title=f"DetectifAI Incident Report - {report.video_id}",
            author="DetectifAI Security System"
        )
        
        # Define custom styles
        styles = getSampleStyleSheet()
        
        styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=styles['Heading1'],
            fontSize=22,
            spaceAfter=16,
            spaceBefore=0,
            textColor=colors.HexColor('#1a365d'),
            leading=28
        ))
        styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=styles['Heading2'],
            fontSize=15,
            spaceBefore=16,
            spaceAfter=8,
            textColor=colors.HexColor('#1a365d'),
            leading=20,
            borderWidth=0,
            borderColor=colors.HexColor('#3182ce'),
            borderPadding=(0, 0, 4, 0)
        ))
        styles.add(ParagraphStyle(
            name='SubsectionTitle',
            parent=styles['Heading3'],
            fontSize=12,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor('#2d3748'),
            leading=16
        ))
        styles.add(ParagraphStyle(
            name='BodyText',
            parent=styles['Normal'],
            fontSize=10,
            leading=15,
            alignment=TA_JUSTIFY,
            spaceBefore=2,
            spaceAfter=4,
            textColor=colors.HexColor('#2d3748')
        ))
        styles.add(ParagraphStyle(
            name='BulletItem',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            leftIndent=20,
            spaceBefore=2,
            spaceAfter=2,
            textColor=colors.HexColor('#2d3748')
        ))
        styles.add(ParagraphStyle(
            name='TableHeader',
            parent=styles['Normal'],
            fontSize=9,
            leading=12,
            textColor=colors.white,
            fontName='Helvetica-Bold'
        ))
        styles.add(ParagraphStyle(
            name='TableCell',
            parent=styles['Normal'],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#2d3748')
        ))
        styles.add(ParagraphStyle(
            name='ImageCaption',
            parent=styles['Normal'],
            fontSize=8,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#718096'),
            spaceBefore=2,
            spaceAfter=6
        ))
        styles.add(ParagraphStyle(
            name='FooterText',
            parent=styles['Normal'],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor('#718096'),
            alignment=TA_CENTER
        ))
        
        # Build story (list of flowable elements)
        story = []
        
        # --- Title Banner ---
        story.append(Paragraph("DetectifAI", styles['ReportTitle']))
        story.append(Paragraph(
            "<i>AI-Powered Surveillance Analysis Report</i>",
            styles['BodyText']
        ))
        story.append(Spacer(1, 6))
        story.append(HRFlowable(
            width="100%", thickness=2,
            color=colors.HexColor('#3182ce'),
            spaceBefore=4, spaceAfter=12
        ))
        
        # --- Report Metadata Table ---
        meta_data = [
            ['Report ID:', report.report_id],
            ['Video ID:', report.video_id],
            ['Classification:', report.metadata.get('classification', 'CONFIDENTIAL')],
            ['Generated:', report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')],
        ]
        meta_table = Table(meta_data, colWidths=[35*mm, 120*mm])
        meta_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1a365d')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#2d3748')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 12))
        story.append(HRFlowable(
            width="100%", thickness=1,
            color=colors.HexColor('#e2e8f0'),
            spaceBefore=6, spaceAfter=12
        ))
        
        # --- Sections ---
        for section in sorted(report.sections, key=lambda x: x.order):
            content = section.content or ''
            
            # Parse section content markdown into ReportLab elements
            section_elements = self._parse_section_content(content, styles)
            story.extend(section_elements)
            
            # Add evidence gallery images if any
            if section.images:
                for img_data in section.images:
                    url = img_data.get('url')
                    caption = img_data.get('caption', '')
                    if url:
                        img_el = self._download_image(url)
                        if img_el:
                            story.append(Spacer(1, 4))
                            story.append(img_el)
                            if caption:
                                story.append(Paragraph(
                                    f"<i>{self._convert_inline_markdown(caption)}</i>",
                                    styles['ImageCaption']
                                ))
                            story.append(Spacer(1, 6))
            
            # Section separator
            story.append(Spacer(1, 8))
            story.append(HRFlowable(
                width="80%", thickness=0.5,
                color=colors.HexColor('#e2e8f0'),
                spaceBefore=4, spaceAfter=8
            ))
        
        # --- Footer ---
        story.append(Spacer(1, 20))
        story.append(HRFlowable(
            width="100%", thickness=1.5,
            color=colors.HexColor('#3182ce'),
            spaceBefore=6, spaceAfter=8
        ))
        story.append(Paragraph(
            f"Report ID: {report.report_id} | Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            styles['FooterText']
        ))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "<i>This report was automatically generated by DetectifAI. "
            "All findings are based on AI analysis and should be verified by qualified personnel.</i>",
            styles['FooterText']
        ))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"✅ Simple PDF report saved to: {output_path}")
        return output_path
