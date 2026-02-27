"""
Report Builder - Main Orchestrator

Coordinates all components to generate complete incident reports:
1. Collects data from database
2. Generates content using LLM
3. Assembles the report structure
4. Exports to PDF/HTML
"""

import os
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

from .config import ReportConfig
from .llm_engine import LLMEngine, get_llm_engine
from .data_collector import DataCollector
from .prompt_templates import (
    SYSTEM_PROMPT_REPORT,
    SYSTEM_PROMPT_SUMMARY,
    SYSTEM_PROMPT_TIMELINE,
    SYSTEM_PROMPT_OBSERVATIONS,
    format_executive_summary_prompt,
    format_timeline_prompt,
    format_observations_prompt,
    format_evidence_section_prompt,
    format_conclusion_prompt
)

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """A section of the generated report."""
    name: str
    title: str
    content: str
    images: List[Dict[str, Any]] = field(default_factory=list)
    order: int = 0


@dataclass
class GeneratedReport:
    """Complete generated report with all sections."""
    report_id: str
    video_id: str
    title: str
    generated_at: datetime
    time_range: Optional[Tuple[datetime, datetime]]
    sections: List[ReportSection]
    metadata: Dict[str, Any]
    statistics: Dict[str, Any]
    raw_data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'report_id': self.report_id,
            'video_id': self.video_id,
            'title': self.title,
            'generated_at': self.generated_at.isoformat(),
            'time_range': [
                self.time_range[0].isoformat() if self.time_range and self.time_range[0] else None,
                self.time_range[1].isoformat() if self.time_range and self.time_range[1] else None
            ],
            'sections': [
                {
                    'name': s.name,
                    'title': s.title,
                    'content': s.content,
                    'images': s.images,
                    'order': s.order
                }
                for s in sorted(self.sections, key=lambda x: x.order)
            ],
            'metadata': self.metadata,
            'statistics': self.statistics
        }


class ReportGenerator:
    """
    Main report generator class that orchestrates the entire
    report generation pipeline.
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize the report generator.
        
        Args:
            config: Report configuration (uses default if None)
        """
        self.config = config or ReportConfig()
        self.llm_engine: Optional[LLMEngine] = None
        self.data_collector: Optional[DataCollector] = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if successful
        """
        if self._initialized:
            return True
        
        try:
            # Initialize data collector
            self.data_collector = DataCollector(self.config)
            
            # Initialize LLM engine
            self.llm_engine = get_llm_engine(self.config)
            
            # Load the model
            if not self.llm_engine.load_model():
                logger.warning("LLM model not loaded - will generate fallback content")
            
            self._initialized = True
            logger.info("✅ Report generator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize report generator: {e}")
            return False
    
    def generate_report(
        self,
        video_id: str,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        include_sections: Optional[List[str]] = None
    ) -> GeneratedReport:
        """
        Generate a complete incident report for a video.
        
        Args:
            video_id: Video identifier
            time_range: Optional time range to filter events
            include_sections: List of sections to include (None = all)
            
        Returns:
            GeneratedReport object
        """
        if not self._initialized:
            self.initialize()
        
        logger.info(f"Generating report for video: {video_id}")
        
        # Default sections
        if include_sections is None:
            include_sections = ['header', 'executive_summary', 'timeline', 
                              'evidence', 'observations', 'conclusion']
        
        # Collect all data
        report_data = self.data_collector.collect_all_report_data(video_id, time_range)
        
        # Generate report ID
        report_id = f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        
        # Generate each section
        sections = []
        
        # 1. Header section (always included, no LLM needed)
        if 'header' in include_sections:
            sections.append(self._generate_header_section(report_id, report_data))
        
        # 2. Executive Summary
        if 'executive_summary' in include_sections:
            logger.info("📝 Generating executive summary...")
            sections.append(self._generate_executive_summary(report_data))
            logger.info("✅ Executive summary complete")
        
        # 3. Timeline
        if 'timeline' in include_sections:
            logger.info("📝 Generating timeline...")
            sections.append(self._generate_timeline(report_data))
            logger.info("✅ Timeline complete")
        
        # 4. Evidence
        if 'evidence' in include_sections:
            logger.info("📝 Generating evidence section...")
            sections.append(self._generate_evidence_section(report_data))
            logger.info("✅ Evidence section complete")
        
        # 5. Observations
        if 'observations' in include_sections:
            logger.info("📝 Generating observations...")
            sections.append(self._generate_observations(report_data))
            logger.info("✅ Observations complete")
        
        # 6. Conclusion
        if 'conclusion' in include_sections:
            logger.info("📝 Generating conclusion...")
            sections.append(self._generate_conclusion(report_data))
            logger.info("✅ Conclusion complete")
        
        # Create the report object
        report = GeneratedReport(
            report_id=report_id,
            video_id=video_id,
            title=f"Incident Report - {video_id}",
            generated_at=datetime.utcnow(),
            time_range=report_data.get('time_range'),
            sections=sections,
            metadata=report_data.get('metadata', {}),
            statistics=report_data.get('statistics', {}),
            raw_data=report_data
        )
        
        logger.info(f"Report generated: {report_id} with {len(sections)} sections")
        
        return report
    
    @staticmethod
    def _clean_llm_output(content: str, section_title: str) -> str:
        """Strip redundant headings and bold titles from LLM output that duplicate the section heading."""
        import re
        if not content:
            return content
        lines = content.strip().split('\n')
        cleaned_lines = []
        skip_next_blank = False
        title_lower = section_title.lower().replace('_', ' ').strip()
        
        for line in lines:
            stripped = line.strip()
            # Skip markdown heading lines that match the section title
            heading_match = re.match(r'^#{1,3}\s+(.*)', stripped)
            if heading_match:
                heading_text = heading_match.group(1).strip().lower().replace('_', ' ')
                if heading_text == title_lower or title_lower in heading_text or heading_text in title_lower:
                    skip_next_blank = True
                    continue
            # Skip bold-only lines that match the section title
            bold_match = re.match(r'^\*\*([^*]+)\*\*$', stripped)
            if bold_match:
                bold_text = bold_match.group(1).strip().lower().replace('_', ' ')
                if bold_text == title_lower or title_lower in bold_text or bold_text in title_lower:
                    skip_next_blank = True
                    continue
            # Skip blank lines immediately after removed headings
            if skip_next_blank and stripped == '':
                skip_next_blank = False
                continue
            skip_next_blank = False
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _generate_header_section(
        self,
        report_id: str,
        data: Dict[str, Any]
    ) -> ReportSection:
        """Generate the report header section."""
        metadata = data.get('metadata', {})
        stats = data.get('statistics', {})
        time_range = data.get('time_range')
        
        time_range_str = "Not specified"
        if time_range:
            # Convert to datetime if needed
            start_dt = time_range[0]
            if isinstance(start_dt, (int, float)):
                start_dt = datetime.utcfromtimestamp(start_dt)
            end_dt = time_range[1]
            if isinstance(end_dt, (int, float)):
                end_dt = datetime.utcfromtimestamp(end_dt)
            
            start = start_dt.strftime('%Y-%m-%d %H:%M:%S') if start_dt else 'N/A'
            end = end_dt.strftime('%Y-%m-%d %H:%M:%S') if end_dt else 'N/A'
            time_range_str = f"{start} to {end}"
        
        content = f"""# INCIDENT REPORT

**Report ID:** {report_id}  
**Classification:** {self.config.report_classification}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Organization:** {self.config.organization_name}

---

## Report Details

| Field | Value |
|-------|-------|
| Video ID | {data.get('video_id', 'Unknown')} |
| Camera ID | {metadata.get('camera_id', 'Unknown')} |
| Location | {metadata.get('location', 'Not specified')} |
| Analysis Period | {time_range_str} |
| Total Events | {stats.get('total_events', 0)} |
| Total Keyframes | {stats.get('total_keyframes', 0)} |
| Faces Detected | {stats.get('total_faces', 0)} |
"""
        # Add Video Link if available
        if metadata.get('video_url'):
            content += f"\n**[Download/View Video]({metadata.get('video_url')})**\n"
        
        content += "\n---\n"""
        return ReportSection(
            name='header',
            title='Report Header',
            content=content,
            order=0
        )
    
    def _generate_executive_summary(self, data: Dict[str, Any]) -> ReportSection:
        """Generate executive summary using LLM."""
        metadata = data.get('metadata', {})
        stats = data.get('statistics', {})
        time_range = data.get('time_range', (None, None))
        
        # Format time range for prompt
        time_range_formatted = ('Start', 'End')
        if time_range:
            # Convert to datetime if needed
            start_dt = time_range[0]
            if isinstance(start_dt, (int, float)):
                start_dt = datetime.utcfromtimestamp(start_dt)
            end_dt = time_range[1]
            if isinstance(end_dt, (int, float)):
                end_dt = datetime.utcfromtimestamp(end_dt)
            
            time_range_formatted = (
                start_dt.strftime('%Y-%m-%d %H:%M:%S') if start_dt else 'Start',
                end_dt.strftime('%Y-%m-%d %H:%M:%S') if end_dt else 'End'
            )
        
        # Create prompt
        user_prompt = format_executive_summary_prompt(
            video_id=data.get('video_id', 'Unknown'),
            camera_info={
                'camera_id': metadata.get('camera_id', 'Unknown'),
                'location': metadata.get('location', 'Not specified')
            },
            time_range=time_range_formatted,
            event_summary=stats.get('event_types', {}),
            total_events=stats.get('total_events', 0),
            threat_levels=stats.get('threat_levels', {})
        )
        
        # Generate with LLM
        if self.llm_engine and self.llm_engine.is_loaded:
            logger.info("🤖 Calling LLM for executive summary...")
            result = self.llm_engine.generate(
                system_prompt=SYSTEM_PROMPT_SUMMARY,
                user_prompt=user_prompt,
                max_tokens=400  # Shorter for executive summary
            )
            logger.info(f"🤖 LLM response received ({result.get('tokens_used', 0)} tokens)")
            content = self._clean_llm_output(result.get('text', ''), 'Executive Summary')
        else:
            logger.info("⚠️ Using fallback executive summary (LLM not loaded)")
            content = self._fallback_executive_summary(data)
        
        return ReportSection(
            name='executive_summary',
            title='Executive Summary',
            content=f"## Executive Summary\n\n{content}",
            order=1
        )
    
    def _generate_timeline(self, data: Dict[str, Any]) -> ReportSection:
        """Generate incident timeline using LLM."""
        events = data.get('events', [])
        
        if not events:
            content = "*No events detected during the analysis period.*"
        else:
            # Prepare events for prompt
            events_for_prompt = [
                {
                    'timestamp': e.get('timestamp').strftime('%H:%M:%S') if e.get('timestamp') else 'Unknown',
                    'event_type': e.get('event_type', 'Unknown'),
                    'caption': e.get('caption') or e.get('description', 'No description'),
                    'threat_level': e.get('threat_level', 'low'),
                    'keyframe_id': e.get('keyframe_id', 'N/A')
                }
                for e in events[:self.config.max_events_in_report]
            ]
            
            user_prompt = format_timeline_prompt(events_for_prompt)
            
            if self.llm_engine and self.llm_engine.is_loaded:
                logger.info("🤖 Calling LLM for timeline...")
                result = self.llm_engine.generate(
                    system_prompt=SYSTEM_PROMPT_TIMELINE,
                    user_prompt=user_prompt,
                    max_tokens=600  # Longer for timeline
                )
                logger.info(f"🤖 LLM response received ({result.get('tokens_used', 0)} tokens)")
                content = self._clean_llm_output(result.get('text', ''), 'Incident Timeline')
            else:
                logger.info("⚠️ Using fallback timeline (LLM not loaded)")
                content = self._fallback_timeline(events_for_prompt)
        
        # Collect images for this section
        images = [
            {'keyframe_id': e.get('keyframe_id'), 'timestamp': e.get('timestamp')}
            for e in events if e.get('keyframe_id')
        ][:self.config.max_images_per_event * 5]
        
        return ReportSection(
            name='timeline',
            title='Incident Timeline',
            content=f"## Incident Timeline\n\n{content}",
            images=images,
            order=2
        )
    
    def _generate_evidence_section(self, data: Dict[str, Any]) -> ReportSection:
        """Generate evidence catalog section with actual images from MinIO.
        
        Instead of using LLM-generated placeholders, this method:
        - Shows actual keyframe images fetched from MinIO when available
        - Shows actual face crop images from MinIO when available
        - Displays bold 'not found' messages when no data exists
        """
        keyframes = data.get('keyframes', [])
        faces = data.get('faces', [])
        
        content_parts = []
        images = []
        
        # --- Keyframes subsection ---
        if keyframes:
            content_parts.append("### Keyframes\n")
            for i, kf in enumerate(keyframes[:20], 1):
                ts = kf.get('timestamp')
                ts_str = ts.strftime('%H:%M:%S') if hasattr(ts, 'strftime') else str(ts or 'Unknown')
                caption = kf.get('caption') or 'No caption available'
                url = kf.get('image_url')
                
                content_parts.append(f"**Keyframe {i}** — {ts_str}")
                content_parts.append(f"{caption}\n")
                if url:
                    content_parts.append(f"![Keyframe {i}]({url})\n")
                
                # Add to gallery images
                images.append({
                    'type': 'keyframe',
                    'id': kf.get('keyframe_id'),
                    'path': kf.get('image_path'),
                    'url': url,
                    'caption': caption
                })
        else:
            content_parts.append("**No keyframes were captured for this video.**\n")
        
        # --- Face Detections subsection ---
        if faces:
            content_parts.append("\n### Face Detections\n")
            for i, f in enumerate(faces[:10], 1):
                ts = f.get('timestamp')
                ts_str = ts.strftime('%H:%M:%S') if hasattr(ts, 'strftime') else str(ts or 'Unknown')
                conf = f.get('confidence', 0)
                person_id = f.get('person_id') or 'Unidentified'
                url = f.get('crop_url')
                
                content_parts.append(f"**Face {i}** — Detected at {ts_str} (confidence: {conf:.2f}, ID: {person_id})")
                if url:
                    content_parts.append(f"\n![Face {i}]({url})\n")
                else:
                    content_parts.append("")
                
                # Add to gallery images
                if self.config.include_face_crops:
                    images.append({
                        'type': 'face',
                        'id': f.get('face_id'),
                        'path': f.get('crop_path'),
                        'url': url,
                        'caption': f"Face {i} at {ts_str} (conf: {conf:.2f})"
                    })
        else:
            content_parts.append("\n**No faces were detected in this video.**\n")
        
        evidence_content = "\n".join(content_parts)
        
        logger.info(f"📸 Evidence section built: {len(keyframes)} keyframes, {len(faces)} faces")
        
        return ReportSection(
            name='evidence',
            title='Evidence Catalog',
            content=f"## Evidence Catalog\n\n{evidence_content}",
            images=images,
            order=3
        )
    
    def _generate_observations(self, data: Dict[str, Any]) -> ReportSection:
        """Generate observations section using LLM."""
        events = data.get('events', [])
        faces = data.get('faces', [])
        patterns = data.get('patterns', {})
        
        if self.llm_engine and self.llm_engine.is_loaded:
            logger.info("🤖 Calling LLM for observations...")
            # Format patterns for prompt
            time_clusters = patterns.get('time_clusters', [])
            cluster_text = "No significant time clusters identified"
            if time_clusters:
                cluster_text = "\n".join([
                    f"- Cluster: {c.get('start')} to {c.get('end')} ({c.get('event_count')} events)"
                    for c in time_clusters
                ])
            
            escalation_text = patterns.get('escalation', 'No clear escalation pattern')
            
            user_prompt = format_observations_prompt(
                events=events,
                faces_detected=faces,
                patterns={
                    'time_clusters': cluster_text,
                    'escalation': escalation_text
                }
            )
            
            result = self.llm_engine.generate(
                system_prompt=SYSTEM_PROMPT_OBSERVATIONS,
                user_prompt=user_prompt,
                max_tokens=400  # Shorter for observations
            )
            logger.info(f"🤖 LLM response received ({result.get('tokens_used', 0)} tokens)")
            content = self._clean_llm_output(result.get('text', ''), 'Observations')
        else:
            logger.info("⚠️ Using fallback observations (LLM not loaded)")
            content = self._fallback_observations(data)
        
        return ReportSection(
            name='observations',
            title='Observations',
            content=f"## Observations\n\n{content}",
            order=4
        )
    
    def _generate_conclusion(self, data: Dict[str, Any]) -> ReportSection:
        """Generate conclusion section using LLM."""
        stats = data.get('statistics', {})
        threat_levels = stats.get('threat_levels', {})
        
        # Compile key findings
        key_findings = []
        
        if threat_levels.get('critical', 0) > 0:
            key_findings.append(f"{threat_levels['critical']} critical threat event(s) detected")
        
        if threat_levels.get('high', 0) > 0:
            key_findings.append(f"{threat_levels['high']} high threat event(s) detected")
        
        patterns = data.get('patterns', {})
        if patterns.get('repeated_faces'):
            key_findings.append(f"{len(patterns['repeated_faces'])} individual(s) appeared multiple times")
        
        if patterns.get('escalation') == 'increasing':
            key_findings.append("Escalating threat pattern observed")
        
        if not key_findings:
            key_findings.append("No significant security concerns identified")
        
        if self.llm_engine and self.llm_engine.is_loaded:
            logger.info("🤖 Calling LLM for conclusion...")
            user_prompt = format_conclusion_prompt(
                total_events=stats.get('total_events', 0),
                critical_events=threat_levels.get('critical', 0),
                high_events=threat_levels.get('high', 0),
                duration_minutes=stats.get('duration_minutes', 0),
                key_findings=key_findings
            )
            
            result = self.llm_engine.generate(
                system_prompt=SYSTEM_PROMPT_REPORT,
                user_prompt=user_prompt,
                max_tokens=300  # Shorter for conclusion
            )
            logger.info(f"🤖 LLM response received ({result.get('tokens_used', 0)} tokens)")
            content = self._clean_llm_output(result.get('text', ''), 'Conclusion')
        else:
            logger.info("⚠️ Using fallback conclusion (LLM not loaded)")
            content = self._fallback_conclusion(stats, key_findings)
        
        return ReportSection(
            name='conclusion',
            title='Conclusion',
            content=f"## Conclusion\n\n{content}",
            order=5
        )
    
    # =========================================================================
    # FALLBACK METHODS (used when LLM is not available)
    # =========================================================================
    
    def _fallback_executive_summary(self, data: Dict[str, Any]) -> str:
        """Generate basic executive summary without LLM."""
        stats = data.get('statistics', {})
        metadata = data.get('metadata', {})
        
        return f"""This report summarizes the automated security analysis of video footage 
from camera {metadata.get('camera_id', 'Unknown')} located at {metadata.get('location', 'unspecified location')}.

During the analysis period, the system detected a total of {stats.get('total_events', 0)} events, 
including {stats.get('threat_levels', {}).get('critical', 0)} critical and 
{stats.get('threat_levels', {}).get('high', 0)} high-priority incidents.

{stats.get('total_faces', 0)} face detections were recorded for potential identification purposes."""
    
    def _fallback_timeline(self, events: List[Dict[str, Any]]) -> str:
        """Generate basic timeline without LLM."""
        if not events:
            return "*No events detected.*"
        
        lines = ["| Time | Event Type | Description | Threat Level |",
                "| ---- | ---------- | ----------- | ------------ |"]
        
        for e in events:
            lines.append(
                f"| {e.get('timestamp', 'N/A')} | {e.get('event_type', 'Unknown')} | "
                f"{e.get('caption', 'No description')[:50]} | {e.get('threat_level', 'low')} |"
            )
        
        return "\n".join(lines)
    
    def _fallback_evidence_section(
        self,
        keyframes: List[Dict[str, Any]],
        faces: List[Dict[str, Any]]
    ) -> str:
        """Generate basic evidence section without LLM."""
        if not keyframes and not faces:
            return "**No keyframes were captured for this video.**\n\n**No faces were detected in this video.**"
        
        content = ""
        if keyframes:
            content += "### Keyframes\n\n"
            for kf in keyframes:
                content += f"- **{kf.get('keyframe_id')}** ({kf.get('timestamp')}): {kf.get('caption', 'No caption')}\n\n"
        else:
            content += "**No keyframes were captured for this video.**\n\n"
        
        if faces:
            content += "### Face Detections\n\n"
            for f in faces:
                content += f"- **{f.get('face_id')}** at {f.get('timestamp')} (confidence: {f.get('confidence')})\n\n"
        else:
            content += "**No faces were detected in this video.**\n\n"
        
        return content
    
    def _fallback_observations(self, data: Dict[str, Any]) -> str:
        """Generate basic observations without LLM."""
        patterns = data.get('patterns', {})
        
        content = "Based on the analyzed data:\n\n"
        
        if patterns.get('repeated_faces'):
            content += f"- {len(patterns['repeated_faces'])} individual(s) appeared multiple times during the analysis period\n"
        
        if patterns.get('time_clusters'):
            content += f"- {len(patterns['time_clusters'])} time period(s) showed concentrated activity\n"
        
        if patterns.get('escalation'):
            content += f"- Threat level trend: {patterns['escalation']}\n"
        
        if content == "Based on the analyzed data:\n\n":
            content += "- No significant patterns identified in the analyzed footage\n"
        
        return content
    
    def _fallback_conclusion(
        self,
        stats: Dict[str, Any],
        key_findings: List[str]
    ) -> str:
        """Generate basic conclusion without LLM."""
        total = stats.get('total_events', 0)
        critical = stats.get('threat_levels', {}).get('critical', 0)
        high = stats.get('threat_levels', {}).get('high', 0)
        
        content = f"""The automated analysis detected {total} events during the review period. """
        
        if critical > 0 or high > 0:
            content += f"Of these, {critical + high} were classified as high-priority incidents requiring attention. "
        else:
            content += "No high-priority security incidents were detected. "
        
        content += "\n\nKey findings:\n"
        for finding in key_findings:
            content += f"- {finding}\n"
        
        content += "\n*This report was generated automatically by DetectifAI.*"
        
        return content
    
    def export_html(self, report: GeneratedReport, output_path: Optional[str] = None) -> str:
        """
        Export report to HTML format.
        
        Args:
            report: Generated report object
            output_path: Output file path (auto-generated if None)
            
        Returns:
            Path to generated HTML file
        """
        from .html_renderer import HTMLRenderer
        
        renderer = HTMLRenderer(self.config)
        return renderer.render(report, output_path)
    
    def export_pdf(self, report: GeneratedReport, output_path: Optional[str] = None) -> str:
        """
        Export report to PDF format.
        
        Args:
            report: Generated report object
            output_path: Output file path (auto-generated if None)
            
        Returns:
            Path to generated PDF file
        """
        from .pdf_exporter import PDFExporter
        
        exporter = PDFExporter(self.config)
        return exporter.export(report, output_path)
