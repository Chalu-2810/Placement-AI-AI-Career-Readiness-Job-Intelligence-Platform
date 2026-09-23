"""
generate_dataset.py
-------------------
Generates a realistic synthetic dataset for training the ML model.
Run once: python generate_dataset.py
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 1200  # number of records

DOMAINS = [
    'Data Science', 'Web Development', 'Cybersecurity',
    'Cloud Computing', 'DevOps', 'Internet of Things (IoT)',
    'Blockchain Technology', 'Mobile App Development'
]

tenth     = np.round(np.random.uniform(45, 99, N), 2)
twelfth   = np.round(np.random.uniform(45, 99, N), 2)
cgpa      = np.round(np.random.uniform(5.0, 10.0, N), 2)
backlogs  = np.random.choice([0, 1, 2, 3], N, p=[0.65, 0.20, 0.10, 0.05])
aptitude  = np.round(np.random.uniform(20, 100, N), 2)
technical = np.round(np.random.uniform(20, 100, N), 2)
comm      = np.round(np.random.uniform(20, 100, N), 2)
resume_score = np.round(np.random.uniform(20, 100, N), 2)
domains   = np.random.choice(DOMAINS, N)


def label(t, tw, c, b, ap, tech, cm, res):
    score = (
        0.15 * (t   / 100) +
        0.15 * (tw  / 100) +
        0.25 * (c   / 10)  +
        0.15 * (ap  / 100) +
        0.18 * (tech/ 100) +
        0.07 * (cm  / 100) +
        0.05 * (res / 100) -
        0.10 * b
    )
    score += np.random.normal(0, 0.04)
    return 1 if score >= 0.55 else 0


labels = [
    label(tenth[i], twelfth[i], cgpa[i], backlogs[i],
          aptitude[i], technical[i], comm[i], resume_score[i])
    for i in range(N)
]

df = pd.DataFrame({
    'tenth_percentage'   : tenth,
    'twelfth_percentage' : twelfth,
    'cgpa'               : cgpa,
    'backlogs'           : backlogs,
    'aptitude_score'     : aptitude,
    'technical_score'    : technical,
    'communication_score': comm,
    'resume_score'       : resume_score,
    'domain'             : domains,
    'placement_status'   : labels
})

os.makedirs('dataset', exist_ok=True)
df.to_csv('dataset/placement_data.csv', index=False)
print(f"Dataset: {N} records | Placed: {sum(labels)} | Not Placed: {N - sum(labels)}")
print(df.head())
