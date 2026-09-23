"""
prediction.py
-------------
Core prediction engine.
Handles ML inference, readiness scoring, skill gap, domain recommendation, roadmap.
"""

import joblib
import json
import hashlib
import numpy as np
import pandas as pd
import os

from feature_engineering import build_feature_row, row_as_vector, FEATURE_NAMES
from explainability import compute_feature_contributions

# ============================================================
# LOAD ARTIFACTS
# ============================================================
MODEL_DIR = 'models'

def _load(name):
    path = os.path.join(MODEL_DIR, name)
    return joblib.load(path) if os.path.exists(path) else None

model         = _load('trained_model.pkl')
scaler        = _load('scaler.pkl')
label_encoder = _load('label_encoder.pkl')
feature_names = _load('feature_names.pkl') or FEATURE_NAMES
all_models    = _load('all_models.pkl')

_metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
if os.path.exists(_metadata_path):
    with open(_metadata_path) as f:
        MODEL_METADATA = json.load(f)
else:
    MODEL_METADATA = {}

# The name of the model actually selected during training (e.g. "Random
# Forest", "Gradient Boosting"). Falls back to a generic label if
# model_metadata.json hasn't been generated yet (older artifact set).
SELECTED_MODEL_NAME = MODEL_METADATA.get('selected_model', 'Unknown (retrain to populate)')

# Average benchmark scores for placed students
PLACED_AVG = {
    'aptitude_score'     : 70.5,
    'technical_score'    : 72.3,
    'communication_score': 68.8
}

# ============================================================
# PREPROCESS INPUT
# ============================================================
def _build_feature_vector(data: dict):
    """
    Builds a feature vector compatible with the trained scaler/model.

    Uses feature_engineering.build_feature_row() -- the SAME function
    model_training.py's build_features() uses (via the same
    compute_academic_score / compute_skill_score / compute_backlog_penalty
    helpers) -- so training and inference can never compute academic_score
    or skill_score with different weights again. See
    tests/test_feature_consistency.py, which asserts this directly.
    """
    row = build_feature_row(
        tenth=data['tenth'],
        twelfth=data['twelfth'],
        cgpa=data['cgpa'],
        backlogs=data['backlogs'],
        aptitude=data['aptitude'],
        technical=data['technical'],
        communication=data['communication'],
        resume_score=data.get('resume_score', 0),
        domain=data.get('domain', 'Data Science'),
        label_encoder=label_encoder,
    )

    # Return as a named DataFrame, columns in the exact order the saved
    # scaler/model expect, so sklearn does not emit "X does not have valid
    # feature names" warnings and column order can never silently drift.
    vector = row_as_vector(row)
    cols = feature_names if feature_names else FEATURE_NAMES
    return pd.DataFrame([vector], columns=cols)


# ============================================================
# MAIN PREDICTION
# ============================================================
def predict_placement(data: dict) -> dict:
    X        = _build_feature_vector(data)
    X_scaled = scaler.transform(X)

    prediction_raw   = model.predict(X_scaled)[0]
    prediction_label = "Placed" if prediction_raw == 1 else "Not Placed"

    proba      = model.predict_proba(X_scaled)[0]
    confidence = round(float(max(proba)) * 100, 2)

    # All model comparisons
    all_preds = {}
    if all_models:
        for mname, m in all_models.items():
            try:
                p    = m.predict(X_scaled)[0]
                prob = m.predict_proba(X_scaled)[0]
                all_preds[mname] = {
                    'prediction': "Placed" if p == 1 else "Not Placed",
                    'confidence': round(float(max(prob)) * 100, 2)
                }
            except Exception:
                pass

    readiness  = compute_readiness_score(data, confidence)
    skill_gap  = compute_skill_gap(data)
    domain_rec = get_domain_recommendation(data['domain'])
    job_roles  = get_job_roles(data['domain'], prediction_label)
    roadmap    = get_career_roadmap(data['domain'], skill_gap['weak_areas'])
    courses    = get_courses(data['domain'])

    # Phase 4: per-prediction explainability. Reuses X_scaled (the SAME
    # scaled vector already passed to model.predict()/predict_proba()
    # above) -- no re-scaling, no separate feature computation. See
    # explainability.py and docs/ml-methodology.md for the full method
    # and the model-contribution-vs-causation distinction.
    explainability = compute_feature_contributions(
        model, X_scaled[0], feature_names=(feature_names or FEATURE_NAMES)
    )

    return {
        'prediction'           : prediction_label,
        'confidence'           : confidence,
        'readiness_score'      : readiness['score'],
        'readiness_level'      : readiness['level'],
        'all_model_preds'      : all_preds,
        'skill_gap'            : skill_gap,
        'domain_recommendation': domain_rec,
        'job_roles'            : job_roles,
        'roadmap'              : roadmap,
        'courses'              : courses,
        # FIX: previously app.py hardcoded 'Random Forest' when saving to the
        # DB regardless of which model save_artifacts() actually selected.
        # This now reflects the real selected model from model_metadata.json.
        'model_used'           : SELECTED_MODEL_NAME,
        # Scientifically-defensible label for this output. The training
        # dataset is 100% synthetic and formula-generated (see
        # docs/ml-methodology.md) so this is NOT a validated real-world
        # probability of employment -- callers/templates should use this
        # phrase rather than "chance of getting a job".
        'result_label'         : 'Model-Based Placement Readiness Estimate',
        # Phase 4: see explainability.py. 'method' is one of
        # 'linear_coefficient' (exact, per-prediction), 'tree_importance_global'
        # (global, unsigned -- NOT per-prediction), or 'unavailable'.
        'explainability'       : explainability,
    }


# ============================================================
# READINESS SCORE
# ============================================================
def compute_readiness_score(data: dict, confidence: float) -> dict:
    resume_score = float(data.get('resume_score', 0))

    academic = (
        (float(data['tenth'])   / 100) * 15 +
        (float(data['twelfth']) / 100) * 15 +
        (float(data['cgpa'])    / 10)  * 20
    )  # max 50

    skills = (
        (float(data['aptitude'])      / 100) * 10 +
        (float(data['technical'])     / 100) * 12 +
        (float(data['communication']) / 100) * 8  +
        (resume_score                 / 100) * 5
    )  # max 35

    model_conf      = (confidence / 100) * 15   # max 15
    backlog_penalty = int(data['backlogs']) * 3

    score = round(academic + skills + model_conf - backlog_penalty, 2)
    score = max(0, min(100, score))

    if score >= 75:
        level = "Placement Ready 🎯"
    elif score >= 50:
        level = "Almost There 🚀"
    else:
        level = "Needs Improvement 🔧"

    return {'score': score, 'level': level}


# ============================================================
# SKILL GAP ANALYSIS
# ============================================================
def compute_skill_gap(data: dict) -> dict:
    apt  = float(data['aptitude'])
    tech = float(data['technical'])
    comm = float(data['communication'])
    resume_score = float(data.get('resume_score', 0))

    apt_gap  = round(PLACED_AVG['aptitude_score']      - apt,  2)
    tech_gap = round(PLACED_AVG['technical_score']     - tech, 2)
    comm_gap = round(PLACED_AVG['communication_score'] - comm, 2)

    apt_pct  = round((apt_gap  / PLACED_AVG['aptitude_score'])      * 100, 1) if apt_gap  > 0 else 0
    tech_pct = round((tech_gap / PLACED_AVG['technical_score'])     * 100, 1) if tech_gap > 0 else 0
    comm_pct = round((comm_gap / PLACED_AVG['communication_score']) * 100, 1) if comm_gap > 0 else 0

    weak_areas  = []
    suggestions = []

    if apt_gap > 5:
        weak_areas.append("Aptitude")
        suggestions.append("Practice 30 aptitude questions daily on IndiaBix / PrepInsta.")
    if tech_gap > 5:
        weak_areas.append("Technical Skills")
        suggestions.append("Solve 2 LeetCode / HackerRank problems per day.")
    if comm_gap > 5:
        weak_areas.append("Communication")
        suggestions.append("Join Toastmasters or practice GD/PI mock sessions daily.")
    if resume_score < 60:
        weak_areas.append("Resume Quality")
        suggestions.append("Improve resume with ATS keywords, proper formatting, and quantified achievements.")
    elif resume_score < 75:
        suggestions.append("Enhance resume by adding measurable achievements and strong action verbs.")
    else:
        suggestions.append("Resume is strong. Keep it updated with latest projects and skills.")

    if not weak_areas:
        suggestions.insert(0, "Great! Maintain your current performance level.")

    return {
        'apt_gap'     : apt_gap,
        'tech_gap'    : tech_gap,
        'comm_gap'    : comm_gap,
        'apt_pct'     : apt_pct,
        'tech_pct'    : tech_pct,
        'comm_pct'    : comm_pct,
        'resume_score': resume_score,
        'apt_score'   : apt,
        'tech_score'  : tech,
        'comm_score'  : comm,
        'weak_areas'  : weak_areas,
        'suggestions' : suggestions,
    }


# ============================================================
# DOMAIN DATA
# ============================================================
DOMAIN_DATA = {
    "Data Science": {
        'skills'        : ['Python', 'Pandas/NumPy', 'Machine Learning', 'SQL', 'Data Visualization'],
        'certifications': ['IBM Data Science (Coursera)', 'Google Data Analytics', 'Kaggle Certifications'],
        'platforms'     : ['Kaggle', 'DataCamp', 'Google Colab', 'Analytics Vidhya'],
        'projects'      : ['EDA on Real Dataset', 'ML Prediction Model', 'NLP Sentiment Analysis', 'Sales Dashboard'],
        'resources'     : ['Towards Data Science', 'StatQuest YouTube', 'Hands-On ML Book']
    },
    "Web Development": {
        'skills'        : ['HTML/CSS/JS', 'React.js', 'Node.js', 'MongoDB', 'REST APIs'],
        'certifications': ['Meta Frontend Developer', 'freeCodeCamp Full Stack', 'The Odin Project'],
        'platforms'     : ['Frontend Mentor', 'Scrimba', 'JavaScript30'],
        'projects'      : ['Portfolio Website', 'E-commerce App', 'Blog Platform', 'Chat Application'],
        'resources'     : ['MDN Docs', 'CSS Tricks', 'JavaScript.info']
    },
    "Cybersecurity": {
        'skills'        : ['Network Security', 'Ethical Hacking', 'Cryptography', 'OWASP', 'Linux'],
        'certifications': ['CEH', 'CompTIA Security+', 'Cisco CyberOps'],
        'platforms'     : ['TryHackMe', 'Hack The Box', 'OverTheWire'],
        'projects'      : ['Vulnerability Scanner', 'Password Strength Checker', 'Secure Web App', 'Network Monitor'],
        'resources'     : ['OWASP Guide', 'Cybrary', 'Krebs on Security']
    },
    "Cloud Computing": {
        'skills'        : ['AWS', 'Azure', 'GCP', 'Virtualization', 'Cloud Security'],
        'certifications': ['AWS Solutions Architect', 'Azure Fundamentals', 'Google Cloud Cert'],
        'platforms'     : ['AWS Skill Builder', 'Qwiklabs', 'Microsoft Learn'],
        'projects'      : ['Deploy Web App on AWS', 'Cloud Storage System', 'Serverless Function App', 'CI/CD Pipeline'],
        'resources'     : ['AWS Docs', 'Azure Docs', 'Cloud Academy']
    },
    "DevOps": {
        'skills'        : ['Docker', 'Kubernetes', 'CI/CD', 'Jenkins', 'Linux'],
        'certifications': ['Docker Certified Associate', 'Kubernetes CKAD', 'DevOps on Coursera'],
        'platforms'     : ['Katacoda', 'Play with Docker', 'Kubernetes.io'],
        'projects'      : ['CI/CD Pipeline Setup', 'Dockerized Full-Stack App', 'K8s Deployment', 'Monitoring Dashboard'],
        'resources'     : ['DevOps Roadmap', 'Docker Docs', 'K8s Docs']
    },
    "Internet of Things (IoT)": {
        'skills'        : ['Embedded Systems', 'Arduino', 'Raspberry Pi', 'Sensors', 'MQTT'],
        'certifications': ['Cisco IoT Fundamentals', 'Coursera IoT Specialization'],
        'platforms'     : ['Arduino IDE', 'ThingSpeak', 'Tinkercad'],
        'projects'      : ['Smart Home System', 'IoT Weather Station', 'Smart Irrigation', 'Health Monitor'],
        'resources'     : ['Arduino Docs', 'IoT For Beginners GitHub', 'Raspberry Pi Docs']
    },
    "Blockchain Technology": {
        'skills'        : ['Blockchain Basics', 'Ethereum', 'Smart Contracts', 'Solidity', 'Cryptography'],
        'certifications': ['Blockchain Specialization (Coursera)', 'Ethereum Developer Cert'],
        'platforms'     : ['CryptoZombies', 'Remix IDE', 'Alchemy University'],
        'projects'      : ['Crypto Wallet App', 'Voting DApp', 'NFT Marketplace', 'Supply Chain DApp'],
        'resources'     : ['Ethereum Docs', 'Bitcoin Whitepaper', 'Blockchain Council']
    },
    "Mobile App Development": {
        'skills'        : ['Flutter', 'Android (Kotlin)', 'React Native', 'Firebase', 'UI/UX Design'],
        'certifications': ['Flutter Certification', 'Android Developer Cert', 'React Native Course'],
        'platforms'     : ['Android Studio', 'Flutter Docs', 'Expo'],
        'projects'      : ['To-Do App', 'Real-time Chat App', 'E-commerce App', 'Fitness Tracker App'],
        'resources'     : ['Flutter Docs', 'Android Developers', 'React Native Docs']
    }
}

JOB_ROLES = {
    "Data Science"            : ['Data Analyst', 'Data Scientist', 'ML Engineer', 'AI Engineer', 'BI Analyst'],
    "Web Development"         : ['Frontend Developer', 'Backend Developer', 'Full Stack Developer', 'React Developer'],
    "Cybersecurity"           : ['Security Analyst', 'Ethical Hacker', 'Penetration Tester', 'Security Engineer'],
    "Cloud Computing"         : ['Cloud Engineer', 'Cloud Architect', 'AWS Engineer', 'DevOps Engineer'],
    "DevOps"                  : ['DevOps Engineer', 'Site Reliability Engineer', 'CI/CD Engineer', 'Platform Engineer'],
    "Internet of Things (IoT)": ['IoT Engineer', 'Embedded Systems Engineer', 'Hardware Developer', 'Firmware Engineer'],
    "Blockchain Technology"   : ['Blockchain Developer', 'Smart Contract Engineer', 'Crypto Analyst', 'Web3 Developer'],
    "Mobile App Development"  : ['Android Developer', 'iOS Developer', 'Flutter Developer', 'React Native Developer']
}

COURSES = {
    "Data Science": [
        {"title": "Machine Learning Specialization", "provider": "Coursera / Andrew Ng",  "type": "Paid",       "link": "https://coursera.org/specializations/machine-learning-introduction"},
        {"title": "Data Science Bootcamp",           "provider": "Udemy",                 "type": "Paid",       "link": "https://udemy.com"},
        {"title": "Python for Data Science",         "provider": "freeCodeCamp",          "type": "Free",       "link": "https://freecodecamp.org"},
        {"title": "Kaggle Micro-Courses",            "provider": "Kaggle",                "type": "Free",       "link": "https://kaggle.com/learn"},
        {"title": "IBM Data Science Professional",   "provider": "edX",                   "type": "Free Audit", "link": "https://edx.org"},
    ],
    "Web Development": [
        {"title": "Full Stack Web Dev Bootcamp",  "provider": "Udemy / Angela Yu", "type": "Paid",       "link": "https://udemy.com"},
        {"title": "Meta Frontend Developer",      "provider": "Coursera",          "type": "Paid",       "link": "https://coursera.org"},
        {"title": "Responsive Web Design",        "provider": "freeCodeCamp",      "type": "Free",       "link": "https://freecodecamp.org"},
        {"title": "The Odin Project",             "provider": "Self-paced",        "type": "Free",       "link": "https://theodinproject.com"},
        {"title": "Frontend Basics",              "provider": "Scrimba",           "type": "Free",       "link": "https://scrimba.com"},
    ],
    "Cybersecurity": [
        {"title": "Complete Cyber Security Course",     "provider": "Udemy",         "type": "Paid",      "link": "https://udemy.com"},
        {"title": "Google Cybersecurity Certificate",   "provider": "Coursera",      "type": "Paid",      "link": "https://coursera.org"},
        {"title": "Cybersecurity Essentials",           "provider": "Cisco NetAcad", "type": "Free",      "link": "https://netacad.com"},
        {"title": "TryHackMe Learning Paths",           "provider": "TryHackMe",     "type": "Free Tier", "link": "https://tryhackme.com"},
        {"title": "OWASP Web Security Guide",           "provider": "OWASP",         "type": "Free",      "link": "https://owasp.org"},
    ],
    "Cloud Computing": [
        {"title": "AWS Certified Solutions Architect",  "provider": "Udemy",          "type": "Paid",      "link": "https://udemy.com"},
        {"title": "Google Cloud Professional Cert",     "provider": "Coursera",        "type": "Paid",      "link": "https://coursera.org"},
        {"title": "AWS Cloud Practitioner Essentials",  "provider": "AWS Training",    "type": "Free",      "link": "https://aws.amazon.com/training"},
        {"title": "Azure Fundamentals",                 "provider": "Microsoft Learn", "type": "Free",      "link": "https://learn.microsoft.com"},
        {"title": "GCP Basics on Qwiklabs",             "provider": "Google",          "type": "Free Tier", "link": "https://qwiklabs.com"},
    ],
    "DevOps": [
        {"title": "DevOps Bootcamp",          "provider": "Udemy / TechWorld", "type": "Paid", "link": "https://udemy.com"},
        {"title": "DevOps Engineering on AWS","provider": "Coursera",          "type": "Paid", "link": "https://coursera.org"},
        {"title": "Docker for Beginners",     "provider": "Docker Docs",       "type": "Free", "link": "https://docs.docker.com"},
        {"title": "Kubernetes Basics",        "provider": "Kubernetes.io",     "type": "Free", "link": "https://kubernetes.io/docs/tutorials"},
        {"title": "CI/CD with Jenkins",       "provider": "Jenkins Docs",      "type": "Free", "link": "https://jenkins.io"},
    ],
    "Internet of Things (IoT)": [
        {"title": "IoT Specialization",       "provider": "Coursera / UC San Diego", "type": "Paid",       "link": "https://coursera.org"},
        {"title": "Arduino & IoT Bootcamp",   "provider": "Udemy",                   "type": "Paid",       "link": "https://udemy.com"},
        {"title": "IoT Fundamentals",         "provider": "Cisco NetAcad",           "type": "Free",       "link": "https://netacad.com"},
        {"title": "Raspberry Pi Projects",    "provider": "RPi Foundation",          "type": "Free",       "link": "https://raspberrypi.org"},
        {"title": "IoT Basics",               "provider": "edX",                     "type": "Free Audit", "link": "https://edx.org"},
    ],
    "Blockchain Technology": [
        {"title": "Blockchain Specialization",            "provider": "Coursera / U Buffalo", "type": "Paid",       "link": "https://coursera.org"},
        {"title": "Ethereum & Solidity Complete Course",  "provider": "Udemy",                "type": "Paid",       "link": "https://udemy.com"},
        {"title": "Blockchain Basics",                    "provider": "IBM / edX",            "type": "Free Audit", "link": "https://edx.org"},
        {"title": "CryptoZombies Solidity Course",        "provider": "CryptoZombies",        "type": "Free",       "link": "https://cryptozombies.io"},
        {"title": "Alchemy University Web3",              "provider": "Alchemy",              "type": "Free",       "link": "https://university.alchemy.com"},
    ],
    "Mobile App Development": [
        {"title": "Flutter & Dart Complete Course",       "provider": "Udemy / Angela Yu", "type": "Paid", "link": "https://udemy.com"},
        {"title": "Android Development Specialization",   "provider": "Coursera",          "type": "Paid", "link": "https://coursera.org"},
        {"title": "Android Basics with Compose",          "provider": "Google",            "type": "Free", "link": "https://developer.android.com/courses"},
        {"title": "React Native Tutorial",                "provider": "React Native Docs", "type": "Free", "link": "https://reactnative.dev"},
        {"title": "Flutter Official Docs & Codelabs",     "provider": "Flutter",           "type": "Free", "link": "https://flutter.dev"},
    ],
}


def get_domain_recommendation(domain: str) -> dict:
    return DOMAIN_DATA.get(domain, DOMAIN_DATA['Data Science'])


def get_job_roles(domain: str, prediction: str) -> list:
    roles = JOB_ROLES.get(domain, [])
    return roles[:2] if prediction == "Not Placed" else roles


def get_courses(domain: str) -> list:
    return COURSES.get(domain, COURSES['Data Science'])


def get_career_roadmap(domain: str, weak_areas: list) -> dict:
    """
    Returns {stage: {'focus': [topic strings], 'deliverables': [{'id','text'}]}}
    for '30_day_plan' / '60_day_plan' / '90_day_plan'.

    Deliverables (not Focus topics) get a stable `id` -- a deterministic
    hash of (stage, domain, text) -- so a student can mark one complete
    (see app.py's /roadmap/toggle route and database.py's
    roadmap_progress table) and have that stick across page reloads.
    The id is deliberately content-derived, not an arbitrary counter: if
    the student's domain or weak areas change, the roadmap's actual
    content changes and old progress correctly does NOT carry over to
    different tasks -- that's intended, not a bug (see docs/ml-methodology.md
    is the wrong doc; documented in this function's own comment since
    it's this function's own design choice, not a data-source concern).

    FIX (Phase 10): previously only 3 of the 4 possible weak_areas values
    ('Aptitude', 'Technical Skills', 'Communication') affected the
    roadmap at all -- 'Resume Quality' (set by compute_skill_gap() when
    resume_score < 60) was silently ignored here even though it's a real,
    displayed weak area everywhere else in the app (skill gap panel,
    admin skill-gap-frequency chart). All 4 now drive Focus/Deliverables.
    """
    base = DOMAIN_DATA.get(domain, DOMAIN_DATA['Data Science'])

    has_apt_gap    = 'Aptitude'          in weak_areas
    has_tech_gap   = 'Technical Skills'  in weak_areas
    has_comm_gap   = 'Communication'     in weak_areas
    has_resume_gap = 'Resume Quality'    in weak_areas

    def _deliverable(stage, text):
        raw = f'{stage}:{domain}:{text}'
        task_id = hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]
        return {'id': task_id, 'text': text}

    # ---- 30 days ----
    focus_30 = [f"Fundamentals: {', '.join(base['skills'][:2])}"]
    if has_apt_gap:
        focus_30.append('Quantitative aptitude & logical reasoning')
    if has_resume_gap:
        focus_30.append('Resume content & ATS formatting')
    if not has_apt_gap and not has_resume_gap:
        focus_30.append('Core concept revision')

    deliverables_30 = [
        _deliverable('30_day_plan',
                      '20 aptitude problems solved (IndiaBix/PrepInsta)' if has_apt_gap
                      else 'Core concepts revised and summarized'),
        _deliverable('30_day_plan', f"Beginner track completed on {base['platforms'][0]}"),
        _deliverable('30_day_plan',
                      'Resume revised and ATS-optimized' if has_resume_gap
                      else '2 mock technical interviews completed'),
    ]

    # ---- 60 days ----
    focus_60 = [f"Portfolio project: {base['projects'][0]}"]
    if has_tech_gap:
        focus_60.append(f'Core technical skills for {domain}')
    if has_comm_gap:
        focus_60.append('Verbal & written communication')
    focus_60.append(f"Certification track: {base['certifications'][0]}")

    deliverables_60 = [
        _deliverable('60_day_plan', f"Project built: {base['projects'][0]}"),
        _deliverable('60_day_plan', f"Certification earned: {base['certifications'][0]}"),
        _deliverable('60_day_plan',
                      '3 mock GD/PI sessions completed' if has_comm_gap
                      else 'Open-source contribution made'),
        _deliverable('60_day_plan',
                      '10 medium-level problems solved (LeetCode/HackerRank)' if has_tech_gap
                      else f"Explored {base['platforms'][1]}"),
    ]

    # ---- 90 days ----
    advanced_skills = ', '.join(base['skills'][3:]) or base['skills'][-1]
    focus_90 = ['Job applications', 'Interview preparation', f'Advanced skills: {advanced_skills}']

    deliverables_90 = [
        _deliverable('90_day_plan', f"Second project deployed: {base['projects'][1]}"),
        _deliverable('90_day_plan', '30 targeted applications submitted (LinkedIn/Naukri)'),
        _deliverable('90_day_plan', f"Certification completed: {base['certifications'][1]}"),
        _deliverable('90_day_plan', '5 mock interviews completed (HR + Technical)'),
        _deliverable('90_day_plan', 'GitHub portfolio refined and finalized'),
    ]

    return {
        '30_day_plan': {'focus': focus_30, 'deliverables': deliverables_30},
        '60_day_plan': {'focus': focus_60, 'deliverables': deliverables_60},
        '90_day_plan': {'focus': focus_90, 'deliverables': deliverables_90},
    }
