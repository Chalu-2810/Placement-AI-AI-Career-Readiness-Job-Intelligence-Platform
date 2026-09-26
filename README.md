# 🎓 PlacementAI — AI Career Copilot & Job Market Intelligence Platform

PlacementAI is an **AI-powered career and placement platform** designed to help students and job seekers evaluate their career readiness, identify skill gaps, analyze resumes, explore job opportunities, and follow a personalized career roadmap.

The platform combines **Machine Learning, Generative AI, Resume/ATS Analysis, Skill Gap Analysis, Job Market Intelligence, and Career Roadmaps** into one integrated system.

LIVE DEMO: https://placement-ai-9cvb.onrender.com/login

---

## 🚀 Key Features

### 🎯 Placement & Career Readiness

* ML-powered placement prediction
* Placement confidence score
* Career readiness score
* Readiness-level classification
* Academic and skill-based evaluation
* Multi-model prediction comparison

### 📄 Resume & ATS Analysis

* PDF resume upload
* Resume text extraction
* ATS compatibility scoring
* Resume-to-role comparison
* Identification of relevant and missing skills
* Target-role improvement recommendations

### 🧠 AI Career Guidance

* Personalized career recommendations
* AI-assisted career analysis
* Target-role evaluation
* Skill improvement suggestions
* Career-focused recommendations based on user profile

### 📊 Skill Gap Analysis

Identifies gaps across:

* Aptitude
* Technical skills
* Communication
* Resume/ATS readiness
* Domain-specific skills

The platform converts identified gaps into actionable recommendations.

### 🗺️ Personalized Career Roadmap

Generates a structured roadmap based on the user's:

* Target domain
* Existing skills
* Skill gaps
* Career objectives

Users can track roadmap tasks and monitor their progress.

### 💼 Job Market Intelligence

Provides domain-oriented job-market insights, including:

* Job posting trends
* Domain-wise demand
* Posting counts
* Historical job-market snapshots
* Derived job-market analytics

The system is designed to separate job-market data collection from normal user requests through scheduled/refresh workflows.

### 🔎 "Am I Qualified?" Analysis

Users can evaluate their profile against job requirements.

The system provides:

* Qualification analysis
* Missing-skill identification
* AI-generated explanations
* Rule-based fallback analysis

### 📈 Admin Analytics

The administrative dashboard provides analytics covering areas such as:

* Platform overview
* Usage
* Query/activity metrics
* Latency
* Quality
* Documents
* Cost
* User activity

Interactive charts are used to make the analytics easier to understand.

---

# 🧭 Student Journey

```text
Register
   ↓
Login
   ↓
Complete Profile
   ↓
Enter Academic Details
   ↓
Select Career Domain
   ↓
Take Assessment
   ↓
Upload Resume
   ↓
Analyze Skills & Resume
   ↓
Run Placement Prediction
   ↓
View Career Readiness Report
   ↓
Identify Skill Gaps
   ↓
Explore Job Opportunities
   ↓
Follow Personalized Career Roadmap
```

---

# 🤖 Machine Learning

PlacementAI can use multiple machine-learning models for placement/readiness prediction.

### Models

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* Support Vector Machine
* XGBoost
* LightGBM

### Model Outputs

Depending on the prediction workflow, the system can generate:

* Placement prediction
* Prediction confidence
* Readiness score
* Readiness level
* Predicted career domain

### Explainability

Model explainability can be supported using **SHAP** to identify important factors contributing to predictions.

---

# 📚 Supported Career Domains

The current domain configuration includes:

1. Data Science
2. Web Development
3. Cybersecurity
4. Cloud Computing
5. DevOps
6. Internet of Things (IoT)
7. Blockchain Technology
8. Mobile App Development

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │        User          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Web Interface    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Flask Application  │
                    │      / API Layer     │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   ┌─────────────┐      ┌─────────────┐      ┌──────────────┐
   │ ML Pipeline │      │ AI / LLM    │      │ Job Market   │
   │             │      │ Services    │      │ Intelligence │
   └──────┬──────┘      └──────┬──────┘      └──────┬───────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       Database       │
                    │        MySQL         │
                    └──────────────────────┘
```

---

# 🗄️ Database

The application uses **MySQL** for core application data.

### Core Tables

```text
users
domains
academic_details
quiz_scores
prediction_results
skill_gap_analysis
admins
roadmap_progress
job_market_snapshots
```

### Database Responsibilities

* User accounts
* Academic information
* Domain selection
* Assessment results
* ML predictions
* Skill-gap results
* Administrator accounts
* Roadmap progress
* Job-market snapshots

---

# 🛠️ Technology Stack

| Layer                | Technology                                      |
| -------------------- | ----------------------------------------------- |
| Programming Language | Python                                          |
| Backend              | Flask                                           |
| Database             | MySQL                                           |
| Machine Learning     | Scikit-learn, XGBoost, LightGBM                 |
| Data Processing      | Pandas, NumPy                                   |
| Explainable AI       | SHAP                                            |
| AI / LLM             | LangChain, OpenAI / Gemini-compatible workflows |
| RAG                  | FAISS / ChromaDB                                |
| Resume Processing    | PyMuPDF                                         |
| Visualization        | Plotly, Recharts / Chart.js where applicable    |
| Authentication       | Werkzeug Password Hashing                       |
| API                  | REST                                            |
| Containerization     | Docker                                          |
| Version Control      | Git / GitHub                                    |

---

# 📁 Project Structure

The repository contains the application, ML pipeline, database layer, models, question bank, templates, and supporting utilities.

```text
PlacementAI/
│
├── app.py
├── database.py
├── prediction.py
├── model_training.py
├── generate_dataset.py
├── requirements.txt
├── schema.sql
├── .env.example
│
├── models/
│   ├── trained_model.pkl
│   ├── all_models.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── feature_names.pkl
│
├── questions/
│
├── data/
│
├── dataset/
│
├── utils/
│   ├── resume_parser.py
│   └── ats_score.py
│
├── templates/
│
├── static/
│
└── README.md
```

> The exact repository structure may evolve as additional AI, analytics, and job-market modules are added.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd PlacementAI
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🗄️ Database Setup

Make sure MySQL is running.

Create/configure the required database:

```sql
CREATE DATABASE placement_db;
```

Then configure the database connection through environment variables.

> If you already have an existing `placement_db`, **do not drop or recreate it**. Verify the existing schema before applying additional SQL changes.

---

# 🔐 Environment Variables

Create a `.env` file based on `.env.example`.

Example:

```env
DB_HOST=localhost
DB_PORT=3306
DB_NAME=placement_db
DB_USER=root
DB_PASSWORD=your_mysql_password

SECRET_KEY=your_secret_key
```

Additional AI/API credentials should also be supplied through environment variables when required.

### Never commit:

```text
.env
API keys
Database passwords
Secret keys
Production credentials
```

---

# 🤖 ML Model Setup

If pre-trained model files are not available, generate the training dataset and train the models:

```bash
python generate_dataset.py
```

Then:

```bash
python model_training.py
```

The generated model artifacts are stored in the `models/` directory.

---

# ▶️ Run Locally

Start the Flask application:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

# 🔑 Authentication

### Student

Students can create an account through the registration page and then log in to access their career dashboard.

### Administrator

Administrator credentials should be configured securely in the database.

**No default production password is stored in this README.**

---

# 🔒 Security

PlacementAI follows basic application-security practices including:

* Password hashing using Werkzeug
* Environment-based configuration
* Authentication-protected routes
* Input validation
* Structured application errors
* Separation of credentials from source code
* Protection of sensitive configuration

Before deployment, verify that:

```text
.env
database credentials
API keys
secret keys
private credentials
```

are excluded from Git.

---

# 📊 Example Workflow

A student can:

```text
Create Account
      ↓
Enter Academic Information
      ↓
Select Data Science
      ↓
Complete Assessment
      ↓
Upload Resume
      ↓
Receive ATS Score
      ↓
Run Placement Prediction
      ↓
View Readiness Score
      ↓
Identify Skill Gaps
      ↓
Compare Against Target Roles
      ↓
Explore Job Opportunities
      ↓
Follow Career Roadmap
```

---

# 🎯 Project Objectives

PlacementAI is designed to move beyond a simple **Placed / Not Placed** prediction.

The broader objective is to connect:

```text
Student Profile
      ↓
Assessment
      ↓
ML Prediction
      ↓
Career Readiness
      ↓
Skill Gap Analysis
      ↓
Resume Improvement
      ↓
Job Market Intelligence
      ↓
Career Roadmap
      ↓
Continuous Progress
```

This provides users with both an assessment of their current position and actionable information for improving their career readiness.

---

# 🚧 Future Enhancements

Planned improvements may include:

* More real-time job-market sources
* Expanded career domains
* Advanced multi-tenant support
* Improved RAG pipelines
* Automated model monitoring
* Cloud deployment
* More resume formats
* Advanced job matching
* Personalized learning recommendations
* Additional analytics and reporting

---

# 📌 Project Status

**Active Development**

PlacementAI is being developed as a full-stack AI/ML career platform combining:

* Machine Learning
* Generative AI
* Resume Intelligence
* Skill Gap Analysis
* Career Roadmaps
* Job Market Intelligence
* Analytics

---

# 👨‍💻 Author

**Chalukya B.**

B.E. / B.Tech — Computer Science Engineering

---

## ⭐ Project Vision

> **From placement prediction to continuous career guidance.**

PlacementAI aims to help students understand where they currently stand, identify what they need to improve, discover relevant opportunities, and follow a structured path toward their target career.
