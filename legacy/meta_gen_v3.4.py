from datetime import datetime
import os
import asyncio
import json
import logging
import pathlib
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import asyncpg
import httpx
import fitz # PyMuPDF
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

# --- Google GenAI Imports ---
from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from google.genai.errors import APIError
# ----------------------------

# ---------------------- Configuration ----------------------
DB_DSN = os.environ.get('DB_DSN', 'postgres://user:pass@host:port/database')
COURSES_BASE_PATH = Path(os.environ.get('COURSES_BASE_PATH', '/data/courses'))

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# --- External Data Paths and Environment Variable Names ---
KCM_PATH = Path('data/competencies.json')
SGOS_PATH = Path('data/SGOS.json')
SCORM_FLAG_ENV = 'SCORM_FLAG'

MAX_CONCURRENCY = int(os.environ.get('MAX_CONCURRENCY', '5'))
WORKER_BATCH_SIZE = int(os.environ.get('WORKER_BATCH_SIZE', '10'))
MAX_ATTEMPTS = int(os.environ.get('MAX_ATTEMPTS', '3'))

# --- GenAI Specific Configuration (Updated for Vertex AI) ---
# For Vertex AI, ensure GOOGLE_APPLICATION_CREDENTIALS is set in the environment.
GOOGLE_PROJECT_ID = os.environ.get('GOOGLE_PROJECT_ID', 'your-gcp-project-id')
GOOGLE_LOCATION = os.environ.get('GOOGLE_LOCATION', 'us-central1')
GENAI_MODEL_NAME = os.environ.get('GENAI_MODEL_NAME', 'gemini-2.5-pro') # Updated default model to pro
LLM_PROMPT_VERSION = 'v3.4-advanced-extended' # Reverted version tracking
# ----------------------------

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True) 

log_filename = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")


# Set up main logger
# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # You can change to DEBUG for more details
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()  # also prints to console
    ]
)
logger = logging.getLogger('course-regenerator')


# Check if environment variable is set before assigning
if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
    client = genai.Client(
        project=GOOGLE_PROJECT_ID,
        location=GOOGLE_LOCATION,
        vertexai=True
    )
else:
    # Initialize without credentials if not available (will fail later if model is called)
    logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set. LLM calls may fail.")
    client = None

# --- SYSTEM PROMPT TEMPLATE (RESTORED to user's requested text) ---
SYSTEM_PROMPT_TEMPLATE = """
You are an *auditable metadata regeneration engine* for the iGOT Karmayogi Bharat platform. 
Your objective is to process the provided inputs and return a single, valid JSON object. This JSON will represent a complete and standardized metadata record for a learning course, suitable for the iGOT platform.

## Inputs:
You will be provided with the following inputs. You must use only these sources.
- Course metadata: (JSON) The raw, original metadata object for the course.
- transcript_text: (String, optional) The full text transcript of the course audio/video.
- pdf_texts: (Array of Strings, optional) Extracted text content from any associated PDF documents.
- kcm_json: (JSON) The master Karmayogi Competency Model, containing all valid Functional and Behavioural competencies.
- sgos_json: (JSON) The master Sector–Subsector–Theme mapping for domain classification.
- scorm_flag: (Boolean) A flag indicating if the content is a SCORM package, which may lack a transcript or PDFs.

## These are global rules that you must follow at all times.
- Strict Adherence: Follow all rules and constraints without deviation.
- JSON Output Only: Your entire output must be a single, valid JSON object. Do not include any introductory text, explanations, apologies, or markdown formatting (like json) outside of the JSON structure itself.
- Schema Compliance: Ensure all data types are correct, and there are no trailing commas.
- Language Consistency: Only generated text fields (title, summary, description, tags, etc.) must be in the same language as originally provided in the input course metadata.
- Auditability: You must generate a comprehensive explain object within the JSON to document your reasoning, confidence levels, and any issues encountered.

## Field Generation Rules:

### 1. Generate the following standardized metadata fields for a learning course based on provided metadata, transcripts, PDFs, and reference texts.

    **Title:**
    - Generate a clear, professional course title of 6–12 words.
    - Preserve the original meaning but improve clarity, capitalization, and precision.
    - Avoid provider names, organization names, or promotional terms.
    - The title must accurately reflect the course subject and context.

    **Summary:**
    - Write 3–4 concise sentences (80–100 words total) summarizing the course.
    - Capture the main topic, its relevance, and the key benefits for the learner.
    - Use a simple, formal, professional tone.
    - Do not repeat provider or organization names.
    - Output must be a single paragraph in plain text.

    **Description:**
    - Write an expanded description of 150–250 words.
    - Cover the course’s learning objectives, structure, and relevance to governance or public sector context.
    - Include:
        - Context: Why this topic is important for the target audience.
        - Key Learning Areas: What will be covered.
        - Application/Outcomes: How learners can apply the knowledge.
        - Optional closing motivational or outcome-focused line
    - Maintain an informative, neutral, and accessible tone suitable for official training material.
    - Avoid marketing or sales-style language.
    - Output must be plain text without bullet points or markdown formatting.

    **Tags:**
    - Generate an array of 10–15 descriptive keyword strings.
    - Keywords must be derived from the course content (metadata, transcript, PDFs).
    - Do not include organization names, provider names, or course codes.
    - If a transcript is available, derive tags from its top 20 significant keywords.
    - Add relevant cross-domain conceptual tags where appropriate (for example: Digital Literacy, Productivity Tools).
    - Output as an array of clean keyword strings.

### 2. Functional and Behavioural competencies: 
    - AI must contextually determine the primary competency area first (Domain/Functional/Behavioural) purely on content analysis, not the predeclared competency area in course metadata.
    - Primary competency area matching score should be > 0.85 (contextually and semantically)
    - Then, map only those Themes and SubThemes from `kcm_json` that are semantically relevant (cosine ≥ 0.85).
    - Analyse the competencies theme and subtheme definition to understand the competencies description to align more contextually.
    - No cross-category mapping allowed.
        - If Theme/SubTheme is Behavioural, it cannot appear in Functional and vice versa.
    - If AI is uncertain (confidence < 0.70), competency must move to "SuggestiveCompetencies" with reason and confidence.
    - Generate only if primary competency area is identified as Behavioural or Functional.
    - Do not generate Functional and Behavioural competencies if primary competency area is identified as *Domain*

### 3. **Suggestive Competencies**: 
    - Suggestive Competencies must also come from the `kcm_json` framework**, that are contextual analysis of the course content or transcript. 
    - Use semantic similarity (contextualization) ≥ **0.85 cosine** to identify relevant KCM entries. 
    - Do not create new competency labels outside the `kcm_json` list. 
    - Each suggestive competency must include `Confidence` and `Rationale`.

### 4. Domain Competencies:
    - Domain competency is identified as Sector, SubSector, and SubSectorTheme.
    - If primary competency area is identified as Domain then only generate Sector, SubSector, and SubSectorTheme from `sgos_json` only
    
### 5. Sector, SubSector, SubSectorTheme (Contextual + Semantic Logic):
    - The AI must determine whether the course has a Domain-oriented nature based purely on content analysis, not the predeclared competency area in course metadata.
    - Do not generate Sector, SubSector, SubSectorTheme if primary competency area is identified as *Behavioural* or *Functional*
    - Evaluation Approach:
        - Use a dual-layer matching logic —
            1. Semantic Similarity Layer (≥0.85 cosine) → Check alignment of key terms and concepts with SGOS Themes.
            2. Contextual Relevance Layer (≥0.85 contextual score) → Check whether the intent and purpose of the course aligns with the real-world scope of the SGOS theme (role relevance, sector function, ministry activity, or outcome).
    - Mapping Criteria:
        - Both semantic and contextual alignment must be met for final mapping.
        - Contextual understanding includes checking if the course’s objectives, roles, policies, or operational impact are tied to that sector/ministry’s mandate.
        - If only one layer matches (semantic yes, contextual no → or vice versa), then mark "Not Applicable" — do not force a mapping.
        - Record the evidence phrases, logic triggers, and confidence in `sgos_reason`.
        - Make sure Sector and subsector and theme should be from `sgos_json` only  

### 6. PRIOR KNOWLEDGE DECLARED BY AUTHOR/SPEAKER
    - This section represents prerequisites *explicitly stated* in the metadata, course introduction,  speaker narration, or transcript text, presentation text, and documents)
    - If such information is found, set "InstructorDeclared": true, otherwise false.
    - When true, include the following sub-fields:
        - DeclaredLevel: (None / Basic / Moderate / Advanced)
        - DeclaredTopics: list of explicitly mentioned prerequisite subjects, tools, or concepts.
        - DeclaredNotes: short 1–2 sentence contextual note from the transcript or metadata.
        - Required: “Yes” or “No” — specify if this prerequisite is *mandatory* for learning outcomes.
        **Example:**
            "PriorKnowledgeDeclared": {
                "InstructorDeclared": true,
                "DeclaredLevel": "Basic",
                "DeclaredTopics": ["Basics of Telecom", "4G LTE Overview"],
                "DeclaredNotes": "Instructor mentions learners should already understand 4G architecture.",
                "Required": "Yes"
            }
    - If no explicit prerequisites are found, set "instructorDeclared": false and leave other fields empty. and 
      If absent, infer `SuggestivePriorKnowledge` (include `SuggestedTopics`, `Confidence`, `Rationale`, and `SuggestedLearningPath` using progressive course titles from metadata, pdf texts, and transcript).

### 7. SUGGESTIVE PRIOR KNOWLEDGE (AI-INFERRED)
    - Populate this object only if priorKnowledgeDeclared.instructorDeclared is false.
    - If no explicit prerequisites are found, infer prerequisites using content semantics
        (keywords, transcript context, technical terms, domain references).
    - Populate only if confidence ≥ 0.75.
    - Include the following sub-fields:
        - AIRecommended: true or false
        - SuggestedLevel: Basic / Moderate / Advanced (based on conceptual density)
        - SuggestedTopics: inferred prerequisite skills/topics (3–6)
        - Required: “Yes” if essential to understanding content; else “No”
        - SuggestedLearningPath: ordered list of course titles from metadata, pdf texts, and transcript
        that logically precede the current course (use cosine ≥ 0.85 similarity)
        - Confidence: 0–1 numeric
        - Rationale: 1–2 sentence reason for inference (mention phrase/topic cues)
    
        **Example:**            
        "SuggestivePriorKnowledge": {
            "AIRecommended": true,
            "SuggestedLevel": "Moderate",
            "SuggestedTopics": ["4G LTE Basics", "Radio Spectrum Management"],
            "Required": "No",
            "SuggestedLearningPath": [
                "Introduction to Digital Communication",
                "Understanding 4G LTE",
                "5G New Radio - Spectrum Related Aspects"
            ],
            "Confidence": 0.88,
            "Rationale": "Transcript mentions LTE, frequency bands, and bandwidth aggregation concepts."
        }
    
    **Important:**
    • If transcript/PDFs are unavailable, rely only on metadata and prior course titles to infer.
    • If both declared and suggestive knowledge exist, keep both but prioritize Declared in final JSON.
    • Always mark presence/absence clearly (InstructorDeclared true/false, AIRecommended true/false).
    • Do not mix Declared and Suggested topics in the same list.
    • If neither found, return both objects with false flags.

### 8. Learning Outcomes:
    - Generate 5–6 clear and measurable learning outcomes and objectives that implicitly reflect purpose (why), process (how), and application (what).
    - Do not label them with “Why”, “How”, or “What”.
    - Each outcome must be written in professional instructional design language suitable for government and public sector learning.
    - Outcomes should begin with strong action verbs (e.g., Identify, Apply, Analyze, Evaluate, Develop, Implement, Interpret).
    - Ensure the set covers cognitive progression — from foundational understanding to applied or strategic capability.
    - Example:
          1. Understand the key elements of emerging technologies and their role in governance.
          2. Analyze how digital transformation impacts citizen service delivery.
          3. Apply appropriate tools and frameworks to design tech-enabled public policies.
          4. Evaluate the potential benefits and risks of technology adoption in public systems.
          5. Demonstrate awareness of ethical and responsible use of technology in governance.

### 9. Transcript Analysis:
    - Extract `KeywordsExtracted`: top 10–15 keywords/phrases (1–4 words), ranked by importance. Provide `method` used (e.g., "term-frequency", "semantic-importance") and a `confidence` (0–1).
    - Output `LearningTone`: one of { "Instructional", "Awareness", "Persuasive", "Narrative", "Technical" } plus `confidence` (0–1)`.
    - Produce `CognitiveMarkers`: list of detected Bloom-level markers derived from verbs & phrasing. Map verbs to Bloom levels using the standard mapping:
        - Remember/Understand → "Remember/Understand" (level 1–2)
        - Apply → "Apply" (3)
        - Analyze → "Analyze" (4)
        - Evaluate → "Evaluate" (5)
        - Create/Design → "Create" (6)
        For each marker include `verb`, `example_phrase`, `estimated_level` (1–6), and `confidence`.
    - Provide `CompetencySignals`: probable KCM Theme/SubTheme matches if `kcm_json` provided. For each suggested competency include `Theme`, `SubTheme`, `Confidence` (0–1), and `trigger_phrases` (the 2–6 phrases from transcript that caused the match). Only include KCM matches with confidence ≥ 0.70; lower-confidence matches should be placed under `Confidence` with their scores.
    - Each returned `confidence` value must be a float between 0 and 1.

## 10. Target Employee Groups:
    - To populate a JSON block called "TargetEmployeeGroups" that contains:
        - "RoleBands": high-level cadre bands (Group A/B/C/D)
        - "Designations": specific official job titles (e.g., Section Officer, Deputy Director, Assistant Engineer)
    - Each designation should directly relate to the **sector**, **sub-sector**, and **functional/Behavioural competencies** of the course.
    - Use the following role band mapping as example:
    | **Sector**          | **Typical Role Bands** | **Common Designation Clusters**                |
    | ------------------- | ---------------------- | ---------------------------------------------- |
    | Governance / Policy | Group A, B             | Under Secretary, Deputy Secretary, Director    |
    | Technology / IT     | Group A, B, C          | Programmer, Systems Analyst, Technical Officer |
    | Finance             | Group A, B             | Section Officer, Accounts Officer, Controller  |
    | Education           | Group A, B, C          | Assistant Professor, Academic Officer, Trainer |
    | Infrastructure      | Group A, B             | Engineer, Executive Engineer, Project Director |
    | Health              | Group A, B             | Medical Officer, Health Analyst                |
    | Social / Welfare    | Group B, C, D          | Case Worker, Field Officer                     |
    | Administration      | All Groups             | Section Officer, LDC, Clerk, Superintendent    |

    - Assign **at least one RoleBand**.
    - Assign multiple if applicable (e.g., "Group A Officers", "Technical Cadre").
    - Avoid overgeneralization (“All groups”) unless course is broad and foundational (e.g., “Ethics in Public Service”).
    - Infer designations using course semantics, sector keywords, and KCM role patterns.
    Each inferred designation must have:
    - A clear linkage to course content or competencies
    - A confidence score (0–1)
    - Rationale (mention phrases, tags, or sector indicators)
    - Use designations from official Government of India hierarchy (examples below).  
    If sector unknown, infer from course context.
        • Include 1–3 RoleBands and 3–6 Designations.
        • Each designation must have:
    - Output designations and role bands in the same language as the course metadata.
    - If the course is in Hindi, provide Hindi equivalents (e.g., “उप सचिव”, “सहायक निदेशक”).
    - Maintain formal capitalization for English and title case for Hindi/Regional.
         
### 11. Rubric scoring: 
    - Calculate a numeric score (0-100) for each field based on the quality and completeness of the generated metadata.
    - Fields to score: learningObjectives, priorKnowledge, bloomTaxonomy, complexityOfContent, expectedRoleOutcome, extentOfLearning, targetAudienceAlignment.
    - ZERO-BASE SCORING POLICY
        ──────────────────────────────────────────────
        • Every parameter starts at **0**.  
        • Increase scores only when you find real, explicit, or confidently inferred evidence from the data (metadata, transcript, PDF text, KCM, SGOS).  
        • No default or assumed minimums.
        
        Parameters to score (each 0–100):
        1️⃣ Learning Objectives (LO)  
        2️⃣ Prior Knowledge (PK)  
        3️⃣ Bloom Taxonomy (BT)  
        4️⃣ Complexity of Content (CC)  
        5️⃣ Expected Role Outcome (ERO)  
        6️⃣ Extent of Learning (EOL)  
        7️⃣ Target Audience Alignment (TAA)
        
        ──────────────────────────────────────────────
        EVIDENCE → SCORE RULES
        ──────────────────────────────────────────────
        
        1️⃣ **Learning Objectives (LO)
        • +10 per measurable, clearly written learning outcome (max 100).  
        • Must start with a Bloom verb (“Apply”, “Analyze”, “Evaluate”, etc.).  
        • Vague or generic (“Learn about”, “Understand”) = +2 only.  
        • Missing or unclear outcomes = +0.
        
        2️⃣ **Prior Knowledge (PK)
        • +0 to 100 if instructor explicitly lists required prerequisites based on match
        • No evidence = +0.
        
        3️⃣ **Bloom Taxonomy (BT)
        • For each learning outcome verb:  
            Remember/Identify = +20, Understand = +40, Apply = +60, Analyze = +80, Evaluate/Create = +100.  
        • Average all verbs used in the course.  
        • If no outcomes = +0.
        
        4️⃣ **Complexity of Content (CC)
        • +15 if domain terms appear moderately (5–10 per 1000 words).  
        • +30 if high technical density (10–20 per 1000 words).  
        • +50 if advanced analytical content, formulas, or policies.  
        • +70–100 for very complex, technical, or data-heavy content (e.g., engineering, analytics).  
        • Purely awareness-level or short = +0.
        
        5️⃣ **Expected Role Outcome (ERO)
        • +20 if course shows clear job-related tasks.  
        • +30–50 if mapped to 1–3 specific designations.  
        • +60–80 if mapped to 6+ designations with role-level examples or applied use-cases.  
        • +100 if highly specialized with measurable job outcomes.  
        • +0 if no job or role connection.
        
        6️⃣ **Extent of Learning (EOL)
        • Duration-based:  
            <30 min = +10, 30–90 min = +20–40, >90 min = +50–70, multi-module (>3 hrs) = +80–100.
        • Practical components add +10–20 (if exercises or labs or any other activity mentioned).  
        • If only single short video = +0–10.
        
        7️⃣ **Target Audience Alignment (TAA)
        • +10 if general public servant audience clear (Group A/B/C/D).  
        • +20–40 if matched to specific cadre or designation (confidence ≥ 0.8).  
        • +50–70 if alignment between content difficulty & designation level.  
        • +100 if perfect sector + role + competency match.  
        • +0 if unspecified or mismatch.
        
        
        RULES FOR EVIDENCE ACCEPTANCE
        ──────────────────────────────────────────────
        • Only assign points when evidence is clear or AI confidence ≥ 0.75.  
        • If no evidence, leave parameter = 0.  
        • All explanations must mention what phrase, transcript part, or metadata element caused the score.  
        • Do not infer or assume; no base or buffer scores allowed. 
    - Calculate totalScore using these weights: LO=5, PK=5, BT=25, CC=35, ERO=10, EOL=20, TAA=0
    - TotalScore = (LO*0.05)+(PK*0.05)+(BT*0.25)+(CC*0.35)+(ERO*0.10)+(EOL*0.20)+(TAA*0.00)
    - Determine the final classification/LearningLevel based on the totalScore.
        - 0 to 55 → Beginner
        - 56 to 75 → Intermediate
        - 76 to 100 → Advanced
    
### 12. SCORM:
    - If `scorm_flag` is `true` or if transcript_text and pdf_texts are not provided, you must still generate all required fields by inferring from the available metadata.
    - In such cases, explicitly state this limitation in the `explain.missing_info` field and assign lower confidence scores (e.g., ≤ 0.7) to all fields that would have otherwise relied on the missing text.

### 13. Tags must be content-descriptive only. Remove any provider or organization names.

### 14. Return an `explain` object including:
    - Triggers used to map competencies (key phrases or sections)
    - Confidence values
    - Missing or low-confidence fields
    - Rationale for SGOS mapping

### 15. Output a **valid JSON** (UTF-8 encoded, no trailing commas, no markdown) with ISO8601 timestamps.

"""
# --- END SYSTEM PROMPT TEMPLATE ---


# Define the expected output structure for the LLM (v3.4-advanced-extended). 
# Domain Competencies are included here as placeholder, matching the schema's required state 
# for consistency with Rule 6.1, even though Rule 5 instructs the LLM not to populate them.
METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "Do_ID": {"type": "string", "description": "Unique course identifier."},
        "CourseName": {"type": "string", "description": "The concise, regenerated title of the course."},
        "Sector": {"type": "string", "description": "The main sector mapped from SGOS."},
        "SubSector": {"type": "string", "description": "The sub-sector mapped from SGOS."},
        "SubSectorTheme": {"type": "string", "description": "The specific theme mapped from SGOS."},
        "CourseSummary": {"type": "string", "description": "A detailed, engaging summary of the course content (2-3 paragraphs)."},
        "CourseDescription": {"type": "string", "description": "A more extensive description detailing what learners will achieve."},
        "LearningOutcomes": {"type": "array", "items": {"type": "string"}, "description": "5-6 clear and measurable learning objectives."},
        "LearningLevel": {"type": "string", "description": "The appropriate learning level.", "enum": ["Beginner", "Intermediate", "Advanced"]},
        "LearningMode": {"type": "string", "description": "The format of the course (e.g., Self-paced, Blended, Instructor-led)."},
        "Duration": {"type": "string", "description": "Total course duration in a human-readable string (e.g., '2 Hours', '90 Minutes')."},
        "TargetEmployeeGroups": {
            "type": "object",
            "properties": {
                "RoleBands": {"type": "array", "items": {"type": "string"}},
                "Designations": {"type": "array", "items": {"type": "string"}}
            }
        },
        "PriorKnowledgeDeclared": {
            "type": "object",
            "properties": {
                "InstructorDeclared": { "type": "boolean" },
                "DeclaredLevel": { "type": "string", "enum": ["None", "Basic", "Moderate", "Advanced"] },
                "DeclaredTopics": { "type": "array", "items": { "type": "string" } },
                "DeclaredNotes": { "type": "string" },
                "Required": { "type": "string", "enum": ["Yes", "No"] },
                "Confidence": { "type": "number" }
            }
        },
        "SuggestivePriorKnowledge": {
            "type": "object",
            "properties": {
                "AIRecommended": {"type": "boolean"},
                "Required": {
                "type": "string",
                "description": "Indicates if the inferred prerequisites are essential for understanding the content.",
                "enum": ["Yes", "No"]
                },
                "SuggestedLevel": {"type": "string"},
                "SuggestedTopics": {"type": "array", "items": {"type": "string"}},
                "SuggestedLearningPath": {"type": "array", "items": {"type": "string"}},
                "Confidence": {"type": "number"},
                "Rationale": {"type": "string"}
            }
        },
        "PrimaryCompetencyArea": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "reason": {"type": "string"}}
        },
        "FunctionalCompetencies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"Theme": {"type": "string"}, "SubTheme": {"type": "string"}}
            }
        },
        "BehaviouralCompetencies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"Theme": {"type": "string"}, "SubTheme": {"type": "string"}}
            }
        },
        "SuggestiveCompetencies": {
            "type": "object",
            "properties": {
                "Functional": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"Theme": {"type": "string"}, "SubTheme": {"type": "string"}, "Confidence": {"type": "number"}, "Rationale": {"type": "string"}}
                    }
                },
                "Behavioural": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"Theme": {"type": "string"}, "SubTheme": {"type": "string"}, "Confidence": {"type": "number"}, "Rationale": {"type": "string"}}
                    }
                }
            }
        },
        "Tags": {"type": "array", "items": {"type": "string"}},
        "RubricScoring": {
            "type": "object",
            "properties": {
                "LearningObjectives": {"type": "integer"},
                "PriorKnowledge": {"type": "integer"},
                "BloomTaxonomy": {"type": "integer"},
                "ComplexityOfContent": {"type": "integer"},
                "ExpectedRoleOutcome": {"type": "integer"},
                "ExtentOfLearning": {"type": "integer"},
                "TargetAudienceAlignment": {"type": "integer"},
                "TotalScore": {"type": "integer"},
                "Classification": {"type": "string", "enum": ["Beginner", "Intermediate", "Advanced"]}
            }
        },
        "TranscriptAnalysis": {
            "type": "object",
            "properties": {
                "KeywordsExtracted": {
                    "type": "object",
                    "properties": {
                        "Values": { "type": "array", "items": { "type": "string" } },
                        "Confidence": { "type": "number" }
                    }
                },
                "LearningTone": {
                    "type": "object",
                    "properties": {
                        "Value": { "type": "string" },
                        "Confidence": { "type": "number" }
                    }
                },
                "CognitiveMarkers": {
                    "type": "object",
                    "properties": {
                        "Values": { "type": "array", "items": { "type": "string" } },
                        "Confidence": { "type": "number" }
                    }
                },
                "CompetencySignals": {
                    "type": "object",
                    "properties": {
                        "Values": { "type": "array", "items": { "type": "string" } },
                        "Confidence": { "type": "number" }
                    }
                }
            }
        },
        "explain": {
            "type": "object",
            "properties": {
                "sgos_reason": {"type": "string"},
                "competency_mapping_triggers": {"type": "object"},
                "prior_knowledge_confidence": {"type": "number"},
                "suggestive_competency_confidence_threshold_used": {"type": "string"},
                "missing_or_low_confidence_fields": {"type": "string"},
                "version_adherence_check": {"type": "string"},
                "mapping_reasons": {
                    "type": "object", 
                    "description": "Detailed reasons for functional and Behavioural competency mapping."
                }, 
                "designation_inference": {
                    "type": "object", 
                    "description": "Details about how target designations were inferred."
                },
                "learning_outcomes_validation": {
                    "type": "object", 
                    "description": "Validation check for learning outcomes structure and verbs."
                },
                "missing_info": {"type": "string"}
            }
        }
    },
    "required": ["Do_ID", "CourseName", "Sector", "SubSector", "SubSectorTheme", "CourseSummary", "LearningOutcomes", "LearningLevel", "Duration", "FunctionalCompetencies", "BehaviouralCompetencies", "Tags", "RubricScoring"]
}
# ---------------------- Logging ----------------------


def load_json_file(path: Path) -> Dict[str, Any]:
    """Loads a JSON file from the given path."""
    if not path.exists():
        logger.error("Required file not found: %s", path)
        raise FileNotFoundError(f"Missing required file: {path}")
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        logger.error("Failed to load or parse JSON from %s: %s", path, e)
        # Return an empty dict if parsing fails, but raise a critical error
        raise ValueError(f"Could not parse JSON from {path}")


async def extract_vtt_text(vtt_path: Path) -> str:
    """Extracts clean text from a VTT file, running I/O in a separate thread."""
    # Use asyncio.to_thread for synchronous file I/O
    def _read_and_clean():
        text_lines = []
        try:
            raw = vtt_path.read_text(encoding='utf-8')
        except Exception:
            # Fallback to latin-1
            raw = vtt_path.read_text(encoding='latin-1')
            
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.upper().startswith('WEBVTT'):
                continue
            if '-->' in line:
                continue
            # skip numeric cue IDs
            if line.isdigit():
                continue
            text_lines.append(line)
        return '\n'.join(text_lines)

    return await asyncio.to_thread(_read_and_clean)


def extract_pdf_text_sync(pdf_path: Path) -> str:
    """Synchronously extracts text from a PDF using PyMuPDF (fitz)."""
    text_parts = []
    try:
        doc = fitz.open(str(pdf_path))
        for page in doc:
            page_text = page.get_text().strip()
            if page_text:
                text_parts.append(page_text)
        doc.close()
    except Exception as e:
        logger.exception('PDF extraction failed for %s: %s', pdf_path, e)
    return '\n\n'.join(text_parts)


async def extract_pdf_text(pdf_path: Path) -> str:
    """Wraps synchronous PDF extraction in an executor."""
    # This correctly uses the run_in_executor default pool
    return await asyncio.get_running_loop().run_in_executor(None, extract_pdf_text_sync, pdf_path)


def list_course_folders(base: Path) -> List[Path]:
    """Lists all child directories assumed to be course folders."""
    if not base.exists():
        logger.error('COURSES_BASE_PATH does not exist: %s', base)
        return []
    # assume each child directory is a course
    return [p for p in base.iterdir() if p.is_dir()]


def build_prompt(course_id: str, current_metadata: Optional[Dict[str, Any]], transcript: str, pdf_snippets: str, kcm_json: Dict[str, Any], sgos_json: Dict[str, Any], scorm_flag: bool) -> str:
    """
    Constructs the detailed prompt for the LLM using the SYSTEM_PROMPT_TEMPLATE
    and includes all input data in the specified format.
    """
    scorm_flag_str = 'true' if scorm_flag else 'false'
    
    prompt = f"""
{SYSTEM_PROMPT_TEMPLATE}

-----------------
INPUT CONTEXTS
-----------------

[Course Metadata]
{json.dumps(current_metadata, indent=2)}

[Course Transcript Text]
<BeginTranscript>
{transcript}
<EndTranscript>

[PDF Extracts]
<BeginSnippets>
{pdf_snippets}
<EndSnippets>

[kcm_json (Karmayogi Competency Model)]
{json.dumps(kcm_json, indent=2)}

[sgos_json (Sector–Subsector–Theme mapping)] 
{json.dumps(sgos_json, indent=2)}

[SCORM Flag] (boolean): 
{scorm_flag_str}

──────────────────────────────────────────────
TASK
──────────────────────────────────────────────
Using all the above input contexts, generate a single JSON output strictly.
Follow all rules above, ensure explainability, and provide all required fields.
"""
    return prompt


@retry(retry=retry_if_exception_type((Exception, APIError)), stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def call_llm(prompt: str) -> Tuple[str, Dict[str, Any]]:
    """Calls the Google GenAI model (via Vertex AI client) and returns the JSON content string and usage metadata."""
    if not client:
        raise RuntimeError("GenAI client is not initialized. Check GOOGLE_APPLICATION_CREDENTIALS.")
        
    logger.info("Calling GenAI model: %s", GENAI_MODEL_NAME)
    
    start = time.time()
    try:
        # Configuration to force structured JSON output based on the defined schema
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=METADATA_SCHEMA,
            temperature=0.1, # Low temperature for factual extraction and consistency,
        )

        filepath = pathlib.Path('data/kcm_book.pdf')

        # Use explicit types.Content for robust SDK interaction
        contents = [
            types.Content(
                role="user",
                parts=[
                    # types.Part.from_bytes(
                    #         data=filepath.read_bytes(),
                    #         mime_type='application/pdf',
                    # ),
                    types.Part.from_text(text=prompt)
                ]
            )
        ]

        file_name = "prompt.txt"

        # Open the file in write mode and write the text
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(prompt)

        # Use the asynchronous Vertex AI method (client.aio)
        response = await client.aio.models.generate_content(
            model=GENAI_MODEL_NAME,
            contents=contents,
            config=config
        )
        
        elapsed = time.time() - start
        logger.info('LLM call took %.2fs', elapsed)
        
        llm_usage = {}
        if response.usage_metadata:
            # Convert usage metadata object to dict for serialization
            llm_usage = response.usage_metadata.to_json_dict()
            logger.info("Gemini usage metadata: %s", llm_usage)

        if not response.text:
            raise RuntimeError("LLM returned an empty response text.")
        
        # response.text is guaranteed to be a valid JSON string adhering to the schema
        return response.text, llm_usage # Return text and usage metadata

    except APIError as e:
        logger.error('GenAI API Error (Retrying): %s', e)
        # Re-raise to trigger the tenacity retry logic
        raise
    except Exception as e:  
        # This catches the Pydantic ValidationError if it happens during schema parsing
        logger.error('LLM call failed (General error): %s', e)
        raise


# ---------------------- Database helpers ----------------------
# Split the multi-statement commands into single statements to avoid PostgresSyntaxError

CREATE_CHECKPOINT_SQL = """
INSERT INTO course_processing_checkpoint (course_id, source_folder, status, last_updated, attempts)
VALUES($1, $2, 'pending', now(), 0)
ON CONFLICT (course_id) DO UPDATE SET source_folder=EXCLUDED.source_folder
"""

CLAIM_COURSE_SQL = """
UPDATE course_processing_checkpoint
SET status='in_progress', attempts = course_processing_checkpoint.attempts + 1, last_updated = now()
WHERE course_id = $1 AND status IN ('pending','failed')
RETURNING course_id, source_folder
"""

# SPLIT MARK_DONE_SQL into two:
MARK_DONE_METADATA_SQL = """
INSERT INTO course_metadata_regenerated (course_id, regenerated_json, source_folder, llm_model, llm_prompt_version, processing_duration_seconds, llm_usage_json, original_metadata_json, status, regenerated_at)
VALUES ($1, $2::jsonb, $3, $4, $5, $6, $7::jsonb, $8::jsonb, 'success', now())
ON CONFLICT (course_id) DO UPDATE SET regenerated_json=EXCLUDED.regenerated_json, source_folder=EXCLUDED.source_folder, llm_model=EXCLUDED.llm_model, llm_prompt_version=EXCLUDED.llm_prompt_version, processing_duration_seconds=EXCLUDED.processing_duration_seconds, llm_usage_json=EXCLUDED.llm_usage_json, original_metadata_json=EXCLUDED.original_metadata_json, status='success', regenerated_at=now();
"""

MARK_DONE_CHECKPOINT_SQL = """
UPDATE course_processing_checkpoint SET status='done', last_updated=now() WHERE course_id=$1;
"""

# SPLIT MARK_FAILED_SQL into two:
MARK_FAILED_CHECKPOINT_SQL = """
UPDATE course_processing_checkpoint SET status='failed', last_updated=now(), last_error=$2 WHERE course_id=$1;
"""

MARK_FAILED_ERROR_LOG_SQL = """
INSERT INTO course_processing_errors (course_id, error_message, created_at) VALUES ($1, $2, now());
"""

# DDL for all required tables
DDL_SCRIPT = """
CREATE TABLE IF NOT EXISTS course_processing_checkpoint (
  course_id TEXT PRIMARY KEY,
  source_folder TEXT,
  status TEXT,
  last_updated TIMESTAMP WITH TIME ZONE,
  attempts INTEGER DEFAULT 0,
  last_error TEXT
);

CREATE TABLE IF NOT EXISTS course_metadata_regenerated (
  course_id TEXT PRIMARY KEY,
  regenerated_json JSONB NOT NULL,
  source_folder TEXT,
  llm_model TEXT,
  llm_prompt_version TEXT,
  processing_duration_seconds NUMERIC,
  llm_usage_json JSONB,
  original_metadata_json JSONB,
  status TEXT,
  regenerated_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS course_processing_errors (
  id BIGSERIAL PRIMARY KEY,
  course_id TEXT,
  error_message TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
"""

async def setup_database(pool: asyncpg.Pool):
    """Executes the DDL to ensure all required tables exist."""
    logger.info("Ensuring database tables exist...")
    async with pool.acquire() as conn:
        # Split DDL script by semicolon and execute each command separately
        for command in DDL_SCRIPT.split(';'):
            clean_command = command.strip()
            if clean_command:
                await conn.execute(clean_command)
    logger.info("Database setup complete.")

# ---------------------- Processing logic ----------------------

async def process_single_course(pool: asyncpg.Pool, course_folder: Path, kcm_json: Dict[str, Any], sgos_json: Dict[str, Any], scorm_flag: bool):
    """
    Main logic to extract data, call the LLM, and persist the result.
    Accepts external reference data (kcm_json, sgos_json) and configuration (scorm_flag).
    """
    course_id = course_folder.name
    logger.info('Processing course %s', course_id)
    start_time = time.time()
    transcript = 'N/A'
    pdf_snippets: List[str] = []
    pdf_snippets_str = 'N/A'
    current_metadata = None
    original_meta_str = 'null' # Initialize for saving the original metadata text
    
    try:
        # 1. READ INPUT DATA
        meta_path = course_folder / 'metadata.json'
        if meta_path.exists():
            try:
                # Capture the original JSON string for storage
                original_meta_str = meta_path.read_text(encoding='utf-8')
                current_metadata = json.loads(original_meta_str)
                del current_metadata['competencies_v6']
            except Exception:
                current_metadata = None
                original_meta_str = 'null'

        for candidate in ['english_subtitles.vtt']:
            p = course_folder / candidate
            if p.exists():
                # Use the asynchronous version of the VTT/TXT extraction
                if p.suffix.lower() == '.vtt':
                    transcript = await extract_vtt_text(p)
                else:
                    # For .txt, offload synchronous read_text to thread pool
                    transcript = await asyncio.to_thread(p.read_text, encoding='utf-8')
                break

        pdfs = list(course_folder.glob('*.pdf'))
        if pdfs:
            # Run PDF extractions concurrently
            pdf_extraction_tasks = [extract_pdf_text(pdf) for pdf in pdfs]
            raw_texts = await asyncio.gather(*pdf_extraction_tasks)
            
            for text in raw_texts:
                if not text:
                    continue
                pdf_snippets.append(text)
            pdf_snippets_str = '\n\n---\n\n'.join([s for s in pdf_snippets if s])

        # 2. CALL LLM
        prompt = build_prompt(course_id, current_metadata, transcript, pdf_snippets_str, kcm_json, sgos_json, scorm_flag)

        # Unpack the response text and the usage metadata
        llm_response_json_str, llm_usage = await call_llm(prompt) 
        
        # 3. PARSE RESPONSE
        # Since the GenAI API enforces JSON output, parsing is straightforward.
        try:
            regenerated = json.loads(llm_response_json_str)
        except Exception as e:
            logger.error('Failed to JSON parse LLM response for %s: %s', course_id, e)
            raise RuntimeError('LLM did not return parseable JSON as expected')

        # Ensure Do_ID is set (based on new schema)
        regenerated['Do_ID'] = str(course_id)
        
        duration = time.time() - start_time
        
        # Prepare usage metadata for storage
        llm_usage_json_str = json.dumps(llm_usage)

        # 4. PERSIST TO DB (SPLIT INTO TWO COMMANDS)
        async with pool.acquire() as conn:
            
            # Command 1: Insert/Update the regenerated metadata record
            await conn.execute(
                MARK_DONE_METADATA_SQL, 
                regenerated['Do_ID'], # $1
                json.dumps(regenerated, sort_keys=True), # $2
                str(course_folder), # $3
                GENAI_MODEL_NAME, # $4
                LLM_PROMPT_VERSION, # $5
                duration, # $6
                llm_usage_json_str, # $7
                original_meta_str   # $8
            )
            # Command 2: Update the processing checkpoint status
            await conn.execute(
                MARK_DONE_CHECKPOINT_SQL,
                regenerated['Do_ID'] # $1
            )


        logger.info('Course %s processed successfully in %.2fs', course_id, duration)
        return True

    except Exception as e:
        error_msg = str(e)
        logger.exception('Processing failed for %s: %s', course_id, error_msg)
        async with pool.acquire() as conn:
            # SPLIT INTO TWO COMMANDS
            # Command 1: Update the processing checkpoint status with error message
            await conn.execute(
                MARK_FAILED_CHECKPOINT_SQL, 
                course_id, 
                error_msg # $2
            ) 
            # Command 2: Insert into the processing errors log
            await conn.execute(
                MARK_FAILED_ERROR_LOG_SQL, 
                course_id, 
                error_msg # $2
            ) 
        return False


async def claim_and_process_loop(pool: asyncpg.Pool, kcm_json: Dict[str, Any], sgos_json: Dict[str, Any], scorm_flag: bool):
    """Worker loop to claim pending courses and process them sequentially."""
    
    await ensure_checkpoint_populated(pool)

    while True:
        async with pool.acquire() as conn:
            # Select pending/failed courses, prioritizing lower attempt counts and older updates
            rows = await conn.fetch("SELECT course_id, source_folder FROM course_processing_checkpoint WHERE status IN ('pending', 'failed') AND attempts < $2 ORDER BY attempts, last_updated LIMIT $1", WORKER_BATCH_SIZE, MAX_ATTEMPTS)
        
        if not rows:
            logger.info('No pending courses found, sleeping for 30s')
            await asyncio.sleep(30)
            continue

        tasks = []
        for r in rows:
            cid = r['course_id']
            # Default folder path if not stored in DB (for older records)
            folder = Path(r['source_folder']) if r['source_folder'] else COURSES_BASE_PATH / cid
            
            # Atomically claim the course
            async with pool.acquire() as conn:
                res = await conn.fetchrow(CLAIM_COURSE_SQL, cid)
                if not res:
                    logger.info('Could not claim %s, likely already claimed or status changed', cid)
                    continue
            tasks.append(process_single_course(pool, folder, kcm_json, sgos_json, scorm_flag))
        
        if tasks:
            # run tasks with limited concurrency to avoid spawning too many coroutines
            # use asyncio.gather but chunk them
            semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

            async def run_with_local_sem(task_coro):
                async with semaphore:
                    return await task_coro

            wrapped = [run_with_local_sem(t) for t in tasks]
            await asyncio.gather(*wrapped)
        else:
            await asyncio.sleep(5)


async def ensure_checkpoint_populated(pool: asyncpg.Pool):
    """Scans the file system and ensures every course folder has an entry in the checkpoint table."""
    logger.info('Populating checkpoint table from filesystem at %s', COURSES_BASE_PATH)
    # list_course_folders is synchronous and is wrapped in a thread using asyncio.to_thread
    folders = await asyncio.to_thread(list_course_folders, COURSES_BASE_PATH) 
    
    async with pool.acquire() as conn:
        # Use an explicit transaction for efficiency
        async with conn.transaction():
            for f in folders:
                cid = f.name
                try:
                    # Use a single query for upsert/insert
                    await conn.execute(CREATE_CHECKPOINT_SQL, cid, str(f))
                except Exception as e:
                    logger.error('Failed to upsert checkpoint for %s: %s', cid, e)
    logger.info('Checkpoint population complete. Found %d courses.', len(folders))


# ---------------------- Entrypoint ----------------------

async def main():
    """Application entry point."""
    
    # 1. Load external reference data
    try:
        kcm_json = load_json_file(KCM_PATH)
        sgos_json = load_json_file(SGOS_PATH)
        scorm_flag = os.environ.get(SCORM_FLAG_ENV, 'false').lower() in ('true', '1')
    except (FileNotFoundError, ValueError) as e:
        logger.critical("Failed to load critical files: %s", e)
        return
    
    logger.info('Starting course regeneration worker with model: %s', GENAI_MODEL_NAME)
    logger.info('SCORM_FLAG is set to: %s', scorm_flag)
    
    # 2. Create DB pool
    pool = await asyncpg.create_pool(dsn=DB_DSN, min_size=2, max_size=MAX_CONCURRENCY)
    try:
        # Step 3: Ensure tables exist before trying to use them
        await setup_database(pool)
        
        # Step 4: Start the main processing loop
        await claim_and_process_loop(pool, kcm_json, sgos_json, scorm_flag)
    finally:
        # Ensure the pool is closed upon exit
        await pool.close()


if __name__ == '__main__':
    try:
        # Use asyncio.run to start the main async function
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info('Shutting down gracefully')
    except Exception as e:
        logger.error('Fatal error in main execution: %s', e)
