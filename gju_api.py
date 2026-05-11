from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="GJU Major Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ─────────────────────────────────────────────────────────────────
model = joblib.load("top_15.pkl")

TOP_15_COLS = ['A3', 'I1', 'I4', 'A8', 'A5', 'R4', 'E5',
               'S8', 'C5', 'S3', 'E3', 'A2', 'R3', 'I8', 'S4']

# ── School → Majors mapping ────────────────────────────────────────────────────
SCHOOL_MAJORS = {
    "School of Computing": [
        "Computer Science (min 75%)",
        "Computer Engineering (min 80%)",
        "Business Intelligence and Data Analytics (min 75%)",
        "Game Design and Media Informatics (min 70%)"
    ],
    "School of Applied Technical Sciences": [
        "Mechatronics Engineering (min 80%)",
        "Industrial Engineering (min 82%)",
        "Mechanical and Maintenance Engineering (min 80%)"
    ],
    "School of Management & Logistics": [
        "Management Science (min 75%)",
        "International Accounting (min 75%)",
        "Logistic Sciences (min 75%)",
        "Digital Marketing (min 65%)",
        "Business Intelligence and Data Analytics (min 75%)"
    ],
    "School of Architecture & Built Environment": [
        "Architecture (min 75%)",
        "Interior Architecture (min 75%)",
        "Design and Visual Communication (min 70%)"
    ],
    "School of Applied Humanities & Social Sciences": [
        "Translation German-English-Arabic (min 70%)",
        "German and English for Business and Communication GEBC (min 70%)"
    ],
    "School of Nursing / Health Sciences": [
        "Nursing Science (min 70%)"
    ],
    "School of Applied Medical Sciences": [
        "Pharmaceutical and Chemical Engineering (min 80%)",
        "Biomedical Engineering (min 80%)"
    ],
    "School of Sustainable Systems & Engineering": [
        "Civil and Environmental Engineering",
        "Electrical Engineering"
    ]
}

# ── Tawjihi filter ─────────────────────────────────────────────────────────────
def filter_by_tawjihi(majors: list, tawjihi: float) -> list:
    filtered = []
    for m in majors:
        if "min 82%" in m and tawjihi < 82:
            continue
        elif "min 80%" in m and tawjihi < 80:
            continue
        elif "min 75%" in m and tawjihi < 75:
            continue
        elif "min 70%" in m and tawjihi < 70:
            continue
        elif "min 65%" in m and tawjihi < 65:
            continue
        filtered.append(m)
    return filtered

# ── Input schema ───────────────────────────────────────────────────────────────
class StudentInput(BaseModel):
    tawjihi: float
    A3: int   # I enjoy sketching, drawing, or painting
    I1: int   # I enjoy solving math or science problems
    I4: int   # I enjoy doing laboratory work
    A8: int   # I enjoy attending art exhibits, concerts, or theatre
    A5: int   # I enjoy creative writing
    R4: int   # I enjoy repairing mechanical equipment
    E5: int   # I enjoy persuading others to do things my way
    S8: int   # I enjoy helping others with personal problems
    C5: int   # I enjoy keeping detailed records
    S3: int   # I enjoy teaching or training others
    E3: int   # I enjoy leading a group toward a goal
    A2: int   # I enjoy playing a musical instrument
    R3: int   # I enjoy working on cars or machines
    I8: int   # I enjoy reading scientific or technical journals
    S4: int   # I enjoy working with the elderly or disabled

# ── Prediction endpoint ────────────────────────────────────────────────────────
@app.post("/predict")
def predict(student: StudentInput):
    # Build input dataframe
    answers = {col: getattr(student, col) for col in TOP_15_COLS}
    input_df = pd.DataFrame([answers])

    # Get prediction and probabilities
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    # Build ranked schools
    ranked = sorted(
        zip(model.classes_, probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    # Get top 3 schools with their majors (filtered by tawjihi)
    recommendations = []
    for school, prob in ranked[:4]:
        majors = SCHOOL_MAJORS.get(school, [])
        available = filter_by_tawjihi(majors, student.tawjihi)
        if available:
            recommendations.append({
                "school": school,
                "confidence": f"{prob:.1%}",
                "available_majors": available
            })
        if len(recommendations) == 3:
            break

    return {
        "top_school": prediction,
        "tawjihi_score": student.tawjihi,
        "recommendations": recommendations,
        "model_accuracy": "77%",
        "note": "Based on RIASEC psychological profile + Tawjihi score filtering"
    }

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "GJU Recommendation API is running", "model": "Random Forest (77% accuracy)"}

# ── Questions endpoint (for frontend) ─────────────────────────────────────────
@app.get("/questions")
def get_questions():
    return {
        "questions": [
            {"id": "A3", "text": "I enjoy sketching, drawing, or painting"},
            {"id": "I1", "text": "I enjoy solving math or science problems"},
            {"id": "I4", "text": "I enjoy doing laboratory work"},
            {"id": "A8", "text": "I enjoy attending art exhibits, concerts, or theatre"},
            {"id": "A5", "text": "I enjoy creative writing"},
            {"id": "R4", "text": "I enjoy repairing mechanical equipment"},
            {"id": "E5", "text": "I enjoy persuading others to do things my way"},
            {"id": "S8", "text": "I enjoy helping others with personal problems"},
            {"id": "C5", "text": "I enjoy keeping detailed records"},
            {"id": "S3", "text": "I enjoy teaching or training others"},
            {"id": "E3", "text": "I enjoy leading a group toward a goal"},
            {"id": "A2", "text": "I enjoy playing a musical instrument"},
            {"id": "R3", "text": "I enjoy working on cars or machines"},
            {"id": "I8", "text": "I enjoy reading scientific or technical journals"},
            {"id": "S4", "text": "I enjoy working with the elderly or disabled"}
        ],
        "scale": {
            "1": "Strongly Dislike",
            "2": "Dislike",
            "3": "Neutral",
            "4": "Like",
            "5": "Strongly Like"
        }
    }

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
