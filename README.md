# Jobs Pipeline

An end-to-end data engineering and NLP pipeline that scrapes job listings,
validates and cleans the data, generates semantic embeddings, and powers
a Streamlit app for resume-based job matching.

---

## Project Structure

```
jobs-pipeline/
├── pipeline/
│   ├── notifications.py                  Shared email alert utility
│   ├── data_diagnostics.py               Stage 1 — Raw data validation
│   ├── Jobsdata_cleaning_final_version.py Stage 2 — Data cleaning
│   ├── cleaning_validation.py            Stage 3 — Post-clean validation
│   ├── embeddings.py                     Stage 4 — Embedding generation
│   └── resume_matcher.py                 Matching engine for Streamlit app
│
├── app/
│   └── app.py                            Streamlit interface
│
├── tests/
│   ├── conftest.py                       Shared fixtures
│   ├── test_cleaning.py                  35 unit tests
│   ├── test_validation.py                17 unit tests
│   ├── test_matching.py                  13 unit tests
│   ├── test_integration.py               11 integration tests
│   └── test_edge_cases.py                10 edge case tests
│
├── .github/workflows/
│   ├── run_tests_dev.yml                 CI — runs on every push to dev
│   └── run_tests_main.yml                CI — runs on every PR to main
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Pipeline Flow

```
Scraper (external)
        ↓
data_diagnostics.py        Raw validation
        ↓ pass / abort + alert
Jobsdata_cleaning_final_version.py    Cleaning
        ↓
cleaning_validation.py     Post-clean validation
        ↓ pass = success email / fail = alert email
embeddings.py              Semantic embedding generation
        ↓
S3 (Embeddings_data/)
        ↓
resume_matcher.py          Resume ↔ job matching
        ↓
app.py                     Streamlit UI
```

---

## Environment Variables

Set these in `/etc/environment` on EC2 before running any pipeline script.
**Never hardcode credentials in code or commit them to GitHub.**

```bash
ALERT_SENDER_MAIL=your_gmail@gmail.com
ALERT_RECEIVER_MAIL=your_gmail@gmail.com
GMAIL_APP_PASSWORD=your_16_char_app_password
GROQ_API_KEY=your_groq_api_key
```

For GitHub Actions, add each as a secret under:
`Settings → Secrets and variables → Actions → New repository secret`

---

## Setup

```bash
# Clone the repo
git clone https://github.com/yourname/jobs-pipeline.git
cd jobs-pipeline

# Install dependencies
pip install -r requirements.txt

# Set environment variables (EC2)
sudo nano /etc/environment
# Add all four variables listed above
```

---

## Running the Pipeline Manually

Each script is run independently in sequence:

```bash
python pipeline/data_diagnostics.py
python pipeline/Jobsdata_cleaning_final_version.py
python pipeline/cleaning_validation.py
python pipeline/embeddings.py
```

The pipeline is scheduled via AWS EventBridge in production.

---

## Running Tests

```bash
# Run all 86 tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_cleaning.py -v

# Run with coverage report
pytest tests/ -v --cov=pipeline --cov-report=term-missing

# Run a specific test by name
pytest tests/test_cleaning.py::TestJobTitleClean::test_none_input_returns_other -v
```

---

## Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable — what EC2 runs in production |
| `dev`  | Active development |

**Day-to-day workflow:**

```bash
# Work on dev
git checkout dev
git add .
git commit -m "your message"
git push origin dev
# GitHub Actions runs all 86 tests automatically

# When ready to deploy
# Open a Pull Request: dev → main on GitHub
# Tests run one final time
# Merge if passing — EC2 pulls from main
```

**On EC2 — update production:**

```bash
cd /home/ec2-user/jobs-pipeline
git pull origin main
```

---

## Key Design Decisions

**Two-stage validation**
Raw validation checks scraper health before cleaning begins.
Post-clean validation checks cleaning correctness before S3 upload.
Each stage has its own alert email with a distinct subject line.

**Groq LLM fallback for experience extraction**
Regex handles explicit patterns. Groq (`llama-3.1-8b-instant`) handles
natural language edge cases. Job descriptions are trimmed to the
requirements section before the API call to minimise token usage.

**Hybrid matching score**
Resume-to-job matching uses a weighted combination of semantic
similarity (65%) and skill overlap (35%). Medium confidence results
are reranked with a cross-encoder for improved accuracy.

**Model consistency**
Resume and job descriptions must always use the same embedding model
(`all-MiniLM-L6-v2`). Cosine similarity is only valid within the
same embedding space.

---

## Alerts Reference

| Email Subject | Meaning |
|---|---|
| `PIPELINE ABORTED — Raw Validation Failed` | Scraper issue, cleaning never ran |
| `PIPELINE WARNING — Post-Clean Validation Failed` | Cleaning issue, S3 upload blocked |
| `PIPELINE SUCCESS — Scraper & Cleaning Good` | Full pipeline completed successfully |
| `PIPELINE CRASHED — Unexpected Error` | Unhandled exception anywhere in pipeline |

Zero emails in inbox = pipeline never ran (EC2 or cron issue).
