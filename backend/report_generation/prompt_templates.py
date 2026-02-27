"""
Prompt Templates for Report Generation

Contains all prompt templates used by the LLM to generate
structured report content. Templates are designed for:
- Deterministic, fact-based output
- Professional forensic tone
- Structured Markdown format
- No hallucinations or assumptions
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PromptTemplate:
    """A prompt template with system and user components."""
    name: str
    system_prompt: str
    user_template: str
    description: str


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT_REPORT = """You are a professional forensic report writer for a CCTV surveillance system called DetectifAI.

Your role is to convert raw AI detection data into formal, professional incident reports.

CRITICAL RULES:
1. ONLY use information explicitly provided in the input data
2. NEVER invent, assume, or hallucinate any facts
3. Use neutral, professional language - no emotions or opinions
4. If data is missing, state "Data not available" - do not guess
5. Use precise timestamps and measurements
6. Refer to detected persons as "Individual A", "Individual B", etc.
7. Do not make legal judgments or accusations
8. Write in third person, past tense
9. Use Markdown formatting for structure
10. Do NOT include a section heading (## or **Title**) at the start of your output - it will be added automatically
11. Do NOT use [[IMAGE:...]] or [[FACE:...]] placeholder syntax - images are handled separately
12. If no evidence or faces exist, clearly state that in bold (e.g., **No faces detected**)

OUTPUT FORMAT:
- Use ### for sub-headings within the section if needed
- Use bullet points for lists
- Use tables for structured data where appropriate
- Do NOT start your output with the section title
"""

SYSTEM_PROMPT_SUMMARY = """You are a professional forensic report writer creating executive summaries.

RULES:
1. Summarize ONLY the facts provided - no assumptions
2. Keep summaries concise (2-4 paragraphs)
3. Highlight key events and their timestamps
4. Use neutral, professional tone
5. Do not speculate on intent or future actions
6. Output in Markdown format
7. Do NOT include a heading like "## Executive Summary" or "**Executive Summary**" - it will be added automatically
"""

SYSTEM_PROMPT_TIMELINE = """You are creating a chronological incident timeline from surveillance data.

RULES:
1. List events in strict chronological order
2. Include precise timestamps (HH:MM:SS format)
3. Describe events factually using provided captions
4. Use consistent terminology
5. Output as Markdown table or list
6. Do NOT include a heading like "## Incident Timeline" - it will be added automatically
7. Do NOT use [[IMAGE:...]] placeholder syntax
"""

SYSTEM_PROMPT_OBSERVATIONS = """You are analyzing surveillance detection patterns for a forensic report.

RULES:
1. Identify patterns ONLY from provided data
2. Note repeated appearances of same individuals (by face ID)
3. Note escalation patterns in event severity
4. Note correlations between events (time proximity, location)
5. Do NOT speculate on intent or motivation
6. Use hedging language: "appears to", "data suggests", "observed pattern"
7. Output in Markdown format
8. Do NOT include a heading like "## Observations" - it will be added automatically
"""


# =============================================================================
# USER PROMPT TEMPLATES
# =============================================================================

def format_executive_summary_prompt(
    video_id: str,
    camera_info: Dict[str, Any],
    time_range: tuple,
    event_summary: Dict[str, Any],
    total_events: int,
    threat_levels: Dict[str, int]
) -> str:
    """
    Format prompt for executive summary generation.
    
    Args:
        video_id: Video identifier
        camera_info: Camera metadata (location, ID)
        time_range: (start_time, end_time) tuple
        event_summary: Summary of events by type
        total_events: Total number of events
        threat_levels: Count of events by threat level
        
    Returns:
        Formatted user prompt
    """
    return f"""Generate an Executive Summary for the following surveillance analysis:

VIDEO ANALYSIS DATA:
- Video ID: {video_id}
- Camera: {camera_info.get('camera_id', 'Unknown')}
- Location: {camera_info.get('location', 'Not specified')}
- Analysis Period: {time_range[0]} to {time_range[1]}
- Total Events Detected: {total_events}

EVENT BREAKDOWN:
{_format_event_summary(event_summary)}

THREAT LEVEL DISTRIBUTION:
- Critical: {threat_levels.get('critical', 0)} events
- High: {threat_levels.get('high', 0)} events
- Medium: {threat_levels.get('medium', 0)} events
- Low: {threat_levels.get('low', 0)} events

Write a professional 2-3 paragraph executive summary covering:
1. Overview of the analyzed footage
2. Key findings and notable events
3. Overall security assessment based on the data

Use ONLY the information provided above. Do not invent additional details."""


def format_timeline_prompt(events: List[Dict[str, Any]]) -> str:
    """
    Format prompt for timeline generation.
    
    Args:
        events: List of event dictionaries with timestamp, type, caption
        
    Returns:
        Formatted user prompt
    """
    events_text = "\n".join([
        f"- [{e.get('timestamp', 'Unknown')}] Type: {e.get('event_type', 'Unknown')} | "
        f"Caption: {e.get('caption', 'No caption')} | "
        f"Threat: {e.get('threat_level', 'Unknown')} | "
        f"Keyframe: {e.get('keyframe_id', 'None')}"
        for e in events
    ])
    
    return f"""Create a detailed incident timeline from the following detected events:

DETECTED EVENTS:
{events_text}

Generate a chronological timeline in Markdown format with:
1. Each event on its own line with timestamp
2. Brief factual description based on the caption
3. Threat level indicator
4. Do NOT include a section heading - it will be added automatically

Format as a Markdown table:
| Time | Event Type | Description | Threat Level |
|------|------------|-------------|--------------|
"""


def format_observations_prompt(
    events: List[Dict[str, Any]],
    faces_detected: List[Dict[str, Any]],
    patterns: Dict[str, Any]
) -> str:
    """
    Format prompt for observations section.
    
    Args:
        events: List of events
        faces_detected: List of detected faces with IDs
        patterns: Pre-computed patterns (repeated faces, time clusters)
        
    Returns:
        Formatted user prompt
    """
    # Format face appearances
    face_summary = ""
    if faces_detected:
        face_counts = {}
        for face in faces_detected:
            fid = face.get('face_id', 'unknown')
            face_counts[fid] = face_counts.get(fid, 0) + 1
        
        face_summary = "\n".join([
            f"- Face ID {fid}: appeared {count} time(s)"
            for fid, count in face_counts.items()
        ])
    else:
        face_summary = "No faces detected"
    
    # Format event clusters
    cluster_info = patterns.get('time_clusters', 'No clustering data')
    escalation_info = patterns.get('escalation', 'No escalation data')
    
    return f"""Analyze the following surveillance data and identify observable patterns:

FACE DETECTION SUMMARY:
{face_summary}

EVENT CLUSTERING:
{cluster_info}

ESCALATION PATTERN:
{escalation_info}

TOTAL EVENTS: {len(events)}

Based ONLY on the data above, write an Observations section that:
1. Notes any individuals appearing multiple times
2. Identifies time periods with concentrated activity
3. Notes any escalation in event severity over time
4. Highlights correlations between different event types

Use hedging language ("appears to", "data suggests") and cite specific data points.
Do NOT speculate on intent or make accusations."""


def format_evidence_section_prompt(
    keyframes: List[Dict[str, Any]],
    face_crops: List[Dict[str, Any]]
) -> str:
    """
    Format prompt for evidence section.
    
    Args:
        keyframes: List of keyframe metadata
        face_crops: List of face crop metadata
        
    Returns:
        Formatted user prompt
    """
    keyframe_list = "\n".join([
        f"- Keyframe {kf.get('keyframe_id', 'unknown')}: "
        f"Time {kf.get('timestamp', 'unknown')}, "
        f"Caption: {kf.get('caption', 'No caption')}"
        for kf in keyframes[:20]  # Limit to 20
    ])
    
    face_list = "\n".join([
        f"- Face {fc.get('face_id', 'unknown')}: "
        f"Time {fc.get('timestamp', 'unknown')}, "
        f"Confidence: {fc.get('confidence', 'unknown')}"
        for fc in face_crops[:10]  # Limit to 10
    ])
    
    return f"""Create an Evidence Section cataloging the following visual evidence:

KEYFRAMES:
{keyframe_list}

FACE DETECTIONS:
{face_list}

Generate a Markdown Evidence Section that:
1. Lists each piece of evidence with a brief description
2. Groups related evidence together
3. Notes the timestamp and relevance of each item
4. Do NOT include a heading like "## Evidence" - it will be added automatically
5. Do NOT use [[IMAGE:...]] or [[FACE:...]] placeholders - images are handled separately
6. If no keyframes or faces exist, state that clearly in bold

Format with clear sub-headers and organized presentation."""


def format_conclusion_prompt(
    total_events: int,
    critical_events: int,
    high_events: int,
    duration_minutes: float,
    key_findings: List[str]
) -> str:
    """
    Format prompt for conclusion section.
    
    Args:
        total_events: Total events detected
        critical_events: Number of critical threat events
        high_events: Number of high threat events
        duration_minutes: Duration of analyzed footage
        key_findings: List of key findings strings
        
    Returns:
        Formatted user prompt
    """
    findings_text = "\n".join([f"- {f}" for f in key_findings]) if key_findings else "- No specific findings to highlight"
    
    return f"""Write a factual Conclusion section based on the following analysis summary:

ANALYSIS SUMMARY:
- Total Events Detected: {total_events}
- Critical Threat Events: {critical_events}
- High Threat Events: {high_events}
- Footage Duration: {duration_minutes:.1f} minutes

KEY FINDINGS:
{findings_text}

Write a 2-paragraph conclusion that:
1. Summarizes the overall security status based on the data
2. Notes any areas that may warrant attention
3. Closes with a factual statement about the analysis

RULES:
- Do NOT make legal judgments
- Do NOT recommend specific actions unless critical events exist
- Do NOT speculate on future events
- Keep tone professional and neutral"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _format_event_summary(event_summary: Dict[str, Any]) -> str:
    """Format event summary dictionary as readable text."""
    if not event_summary:
        return "No events detected"
    
    lines = []
    for event_type, count in event_summary.items():
        lines.append(f"- {event_type}: {count} event(s)")
    
    return "\n".join(lines)


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================

PROMPT_TEMPLATES = {
    'executive_summary': PromptTemplate(
        name='executive_summary',
        system_prompt=SYSTEM_PROMPT_SUMMARY,
        user_template='format_executive_summary_prompt',
        description='Generate executive summary for the report'
    ),
    'timeline': PromptTemplate(
        name='timeline',
        system_prompt=SYSTEM_PROMPT_TIMELINE,
        user_template='format_timeline_prompt',
        description='Generate chronological incident timeline'
    ),
    'observations': PromptTemplate(
        name='observations',
        system_prompt=SYSTEM_PROMPT_OBSERVATIONS,
        user_template='format_observations_prompt',
        description='Generate pattern observations section'
    ),
    'evidence': PromptTemplate(
        name='evidence',
        system_prompt=SYSTEM_PROMPT_REPORT,
        user_template='format_evidence_section_prompt',
        description='Generate evidence catalog section'
    ),
    'conclusion': PromptTemplate(
        name='conclusion',
        system_prompt=SYSTEM_PROMPT_REPORT,
        user_template='format_conclusion_prompt',
        description='Generate conclusion section'
    )
}


def get_template(name: str) -> PromptTemplate:
    """Get a prompt template by name."""
    if name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown template: {name}. Available: {list(PROMPT_TEMPLATES.keys())}")
    return PROMPT_TEMPLATES[name]
