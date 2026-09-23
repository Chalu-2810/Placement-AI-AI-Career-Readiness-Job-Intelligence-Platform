CREATE DATABASE IF NOT EXISTS placement_db;
USE placement_db;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(150) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    register_no VARCHAR(50),
    department VARCHAR(100),
    batch_year INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE domains (
    id INT AUTO_INCREMENT PRIMARY KEY,
    domain_name VARCHAR(100) NOT NULL
);

CREATE TABLE academic_details (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL UNIQUE,
    tenth_percentage FLOAT,
    twelfth_percentage FLOAT,
    cgpa FLOAT,
    backlogs INT DEFAULT 0,
    preferred_domain_id INT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (preferred_domain_id) REFERENCES domains(id)
);

CREATE TABLE quiz_scores (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    aptitude_score FLOAT,
    technical_score FLOAT,
    communication_score FLOAT,
    total_score FLOAT,
    attempt_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE prediction_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    academic_score FLOAT,
    skill_score FLOAT,
    resume_score FLOAT,
    prediction VARCHAR(20),
    confidence_score FLOAT,
    readiness_score FLOAT,
    readiness_level VARCHAR(50),
    model_used VARCHAR(50),
    predicted_domain VARCHAR(100),
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE skill_gap_analysis (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    apt_gap FLOAT,
    tech_gap FLOAT,
    comm_gap FLOAT,
    weak_areas TEXT,
    suggestions TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE admins (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL
);

-- FIX: Admin password is now a proper Werkzeug pbkdf2:sha256 hash (not
-- plaintext). The original schema stored the default admin password as
-- plain text, which caused every admin login attempt to fail because
-- app.py uses check_password_hash() (Werkzeug) to verify it — plain
-- text never matches a Werkzeug hash. It also meant the real default
-- password was sitting in a public/committed file, which defeats the
-- point of hashing it -- fixed here by both hashing it AND no longer
-- stating the plaintext value anywhere in this repo.
--
-- CHANGE THIS BEFORE ANY REAL DEPLOYMENT. To set a new admin password,
-- run this in Python and UPDATE the row:
--   from werkzeug.security import generate_password_hash
--   print(generate_password_hash('YourNewPassword', method='pbkdf2:sha256'))
INSERT INTO admins (username, password) VALUES (
    'admin',
    'pbkdf2:sha256:1000000$gRl3bvlY18CYgWdK$1fc4f572aea3fe0c7f4f4658912137c312486dc44fdf76d204ebe0640660b688'
);

-- Roadmap task completion tracking (Phase 10). Additive only -- does not
-- modify any existing table. task_id is a deterministic hash of
-- (stage, domain, deliverable text) computed by
-- prediction.py::get_career_roadmap() -- see that function's docstring
-- for why content-derived IDs are the correct choice here (progress
-- naturally doesn't carry over when the roadmap's actual content
-- changes, e.g. a different domain or different weak areas).
CREATE TABLE roadmap_progress (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    user_id      INT NOT NULL,
    task_id      VARCHAR(64) NOT NULL,
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_user_task (user_id, task_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Job Market Intelligence snapshots (Phase 5).
-- Additive only -- does not modify any existing table. Each row is one
-- point-in-time aggregate snapshot for one domain, produced by
-- scripts/refresh_job_market.py (never written to by the running app on
-- a user request). raw_aggregates stores the full derived-analytics
-- dict from job_market.build_snapshot() as JSON -- no raw job-posting
-- description text is persisted long-term.
CREATE TABLE job_market_snapshots (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    domain         VARCHAR(100) NOT NULL,
    source         VARCHAR(50)  NOT NULL,
    fetched_at     DATETIME     NOT NULL,
    posting_count  INT          NOT NULL,
    raw_aggregates JSON         NOT NULL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_domain_fetched (domain, fetched_at)
);

-- All 8 supported domains — order matters: domain IDs 1-8 must match
-- the preferred_domain_id values stored by the application and must
-- exactly match the class names in models/label_encoder.pkl.
INSERT INTO domains (domain_name) VALUES
('Data Science'),
('Web Development'),
('Cybersecurity'),
('Cloud Computing'),
('DevOps'),
('Internet of Things (IoT)'),
('Blockchain Technology'),
('Mobile App Development');
