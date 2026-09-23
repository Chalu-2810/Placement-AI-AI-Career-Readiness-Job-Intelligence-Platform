"""
utils/resume.py — PDF resume text extraction + skill detection + ATS scoring.
Uses PyMuPDF (fitz) if available, falls back to pdfminer / raw bytes.
"""

import re

# ── PDF extraction engine ─────────────────────────────────────
try:
    import fitz  # PyMuPDF
    _ENGINE = "pymupdf"
except ImportError:
    try:
        from pdfminer.high_level import extract_text as _pm_extract
        _ENGINE = "pdfminer"
    except ImportError:
        _ENGINE = "none"


def extract_text(file_bytes: bytes) -> str:
    """Extract plain text from PDF bytes."""
    if _ENGINE == "pymupdf":
        doc  = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text

    if _ENGINE == "pdfminer":
        import io
        try:
            return _pm_extract(io.BytesIO(file_bytes))
        except Exception:
            pass

    raw = file_bytes.decode("latin-1", errors="ignore")
    return re.sub(r"[^\x20-\x7E\n]", " ", raw)


# ── Skill taxonomy ────────────────────────────────────────────
SKILL_GROUPS = {
    "Programming Languages": [
        "python", "java", "c++", "c#", "javascript", "typescript", "kotlin", "swift",
        "go", "rust", "scala", "r", "matlab", "php", "ruby", "perl"
    ],
    "Web Technologies": [
        "html", "css", "react", "angular", "vue", "node", "express", "django",
        "flask", "spring", "fastapi", "bootstrap", "tailwind", "graphql", "rest", "soap"
    ],
    "Data & ML": [
        "machine learning", "deep learning", "nlp", "computer vision", "pandas", "numpy",
        "scikit-learn", "tensorflow", "pytorch", "keras", "opencv", "sql", "nosql",
        "data analysis", "data visualization", "tableau", "power bi", "matplotlib", "seaborn"
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "jenkins", "github actions",
        "linux", "bash", "terraform", "ansible", "nginx", "apache"
    ],
    "Databases": [
        "mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle", "firebase", "cassandra"
    ],
    "Tools & Practices": [
        "git", "agile", "scrum", "jira", "figma", "postman", "swagger", "junit", "pytest"
    ],
    "Soft Skills": [
        "leadership", "communication", "teamwork", "problem solving", "analytical",
        "presentation", "time management", "critical thinking"
    ],
}

ALL_SKILLS = {s: g for g, skills in SKILL_GROUPS.items() for s in skills}


def detect_skills(text: str) -> dict:
    """Return {skill: group} for every skill found in text."""
    lower = text.lower()
    found = {}
    for skill, group in ALL_SKILLS.items():
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, lower):
            found[skill] = group
    return found


# ── ATS scoring ───────────────────────────────────────────────
SECTION_KEYWORDS = [
    "education", "experience", "projects", "skills", "certifications",
    "achievements", "internship", "summary", "objective", "contact"
]

def ats_score(text: str, skills_found: dict) -> dict:
    """Compute ATS score 0-100 + suggestions."""
    lower = text.lower()
    words = len(text.split())

    skill_score   = min(20, len(skills_found) * 2)
    section_score = min(20, sum(4 for kw in SECTION_KEYWORDS if kw in lower))
    length_score  = 20 if 200 <= words <= 800 else (10 if words < 200 else 15)
    email_score   = 10 if re.search(r"[\w.+-]+@[\w-]+\.\w+", text) else 0
    phone_score   = 10 if re.search(r"\+?[\d\s\-()]{10,}", text) else 0

    action_words = [
        "developed", "built", "designed", "led", "improved", "optimized",
        "created", "implemented", "managed", "achieved", "reduced", "increased"
    ]
    action_score = min(20, sum(4 for w in action_words if w in lower))

    total = skill_score + section_score + length_score + email_score + phone_score + action_score

    suggestions = []
    if skill_score < 12:
        suggestions.append("Add more technical skills (aim for 8+).")
    if section_score < 16:
        suggestions.append("Include all key sections: Education, Experience, Projects, Skills, Certifications.")
    if words < 200:
        suggestions.append("Resume is too short. Expand project descriptions.")
    if words > 800:
        suggestions.append("Resume is too long. Keep it to 1 page for freshers.")
    if not email_score:
        suggestions.append("No email address detected — add contact info.")
    if not phone_score:
        suggestions.append("No phone number detected.")
    if action_score < 12:
        suggestions.append("Use strong action verbs: Developed, Built, Led, Optimized…")
    if not suggestions:
        suggestions.append("Great resume! Ready for ATS systems. ✓")

    return {
        "total": total,
        "breakdown": {
            "Skills Found": skill_score,
            "Key Sections": section_score,
            "Length"      : length_score,
            "Contact Info": email_score + phone_score,
            "Action Verbs": action_score,
        },
        "suggestions": suggestions,
    }


# ── Full pipeline ─────────────────────────────────────────────
def analyse_resume(file_bytes: bytes) -> dict:
    """Full pipeline: extract → detect skills → score."""
    text   = extract_text(file_bytes)
    skills = detect_skills(text)
    score  = ats_score(text, skills)

    grouped = {}
    for skill, group in skills.items():
        grouped.setdefault(group, []).append(skill)

    return {
        "raw_text"      : text[:3000],
        "skills"        : skills,
        "grouped_skills": grouped,
        "ats"           : score,
        "resume_score"  : score["total"],
    }
