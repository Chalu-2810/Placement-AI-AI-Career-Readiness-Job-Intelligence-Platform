"""
utils/ats_score.py
------------------
ATS scoring utilities — uses the rich resume analysis pipeline from utils/resume.py.
calculate_ats()    → integer score 0-100
get_ats_breakdown()→ dict with breakdown, suggestions, skills_found
"""

from utils.resume import analyse_resume, detect_skills, ats_score as _ats_score


def calculate_ats(text: str) -> int:
    """
    Return an integer ATS score (0-100) from resume plain text.
    Called after extract_text() has already pulled text from the PDF.
    """
    skills = detect_skills(text)
    result = _ats_score(text, skills)
    return result['total']


def get_ats_breakdown(text: str) -> dict:
    """
    Return full breakdown dict compatible with resume.html template.
    Keys returned:
      - breakdown   : {category: score}
      - suggestions : [str, ...]
      - skills_found: [str, ...]
    """
    skills = detect_skills(text)
    result = _ats_score(text, skills)
    return {
        'breakdown'   : result['breakdown'],
        'suggestions' : result['suggestions'],
        'skills_found': list(skills.keys()),
    }
