from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="GJU Major Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load("top_15.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

TOP_15_COLS = ['A3', 'I1', 'I4', 'A8', 'A5', 'R4', 'E5',
               'S8', 'C5', 'S3', 'E3', 'A2', 'R3', 'I8', 'S4']

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
    "School of Sustainable Systems & Engineering": []
}

def filter_by_tawjihi(majors, tawjihi):
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

def get_riasec_scores(student):
    return {
        "Artistic": round((student.A3 + student.A8 + student.A5 + student.A2) / 4, 1),
        "Investigative": round((student.I1 + student.I4 + student.I8) / 3, 1),
        "Realistic": round((student.R4 + student.R3) / 2, 1),
        "Enterprising": round((student.E5 + student.E3) / 2, 1),
        "Social": round((student.S8 + student.S3 + student.S4) / 3, 1),
        "Conventional": round(student.C5 / 1, 1)
    }

class StudentInput(BaseModel):
    tawjihi: float
    A3: int
    I1: int
    I4: int
    A8: int
    A5: int
    R4: int
    E5: int
    S8: int
    C5: int
    S3: int
    E3: int
    A2: int
    R3: int
    I8: int
    S4: int

@app.get("/")
def root():
    return {
        "status": "GJU Recommendation API is running",
        "model_loaded": model is not None,
        "accuracy": "83%"
    }

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

@app.post("/predict")
def predict(student: StudentInput):
    if model is None:
        return {"error": "Model not loaded"}
    try:
        answers = {col: getattr(student, col) for col in TOP_15_COLS}
        input_df = pd.DataFrame([answers])
        probabilities = model.predict_proba(input_df)[0]

        # Get RIASEC scores
        riasec = get_riasec_scores(student)
        dominant_traits = sorted(riasec.items(), key=lambda x: x[1], reverse=True)
        top_traits = [t[0] for t in dominant_traits[:2]]

        # Rank all schools by RF probability
        ranked = sorted(
            zip(model.classes_, probabilities),
            key=lambda x: x[1],
            reverse=True
        )

        # Stage 1: Get the RF predicted school and its available majors
        predicted_school = None
        predicted_school_majors = []
        predicted_confidence = 0

        for school, prob in ranked:
            majors = SCHOOL_MAJORS.get(school, [])
            available = filter_by_tawjihi(majors, student.tawjihi)
            if available:
                predicted_school = school
                predicted_school_majors = available
                predicted_confidence = prob
                break

        # Stage 2: Get all available majors from ALL schools for LLM context
        all_available = []
        for school, prob in ranked:
            majors = SCHOOL_MAJORS.get(school, [])
            available = filter_by_tawjihi(majors, student.tawjihi)
            for major in available:
                all_available.append({
                    "major": major,
                    "school": school,
                    "school_confidence": f"{prob:.1%}"
                })

        return {
            "predicted_school": predicted_school,
            "predicted_school_confidence": f"{predicted_confidence:.1%}",
            "predicted_school_majors": predicted_school_majors,
            "tawjihi_score": student.tawjihi,
            "riasec_scores": riasec,
            "dominant_traits": top_traits,
            "all_available_majors": all_available,
            "model_accuracy": "83%",
            "instruction_for_llm": (
                f"STAGE 1 (ML Prediction): The Random Forest model (83% accuracy) predicts "
                f"the student belongs to '{predicted_school}' with {predicted_confidence:.1%} confidence "
                f"based on their RIASEC psychological profile. "
                f"STAGE 2 (Your job as AI): "
                f"1. Recommend the BEST major from '{predicted_school}' as the PRIMARY recommendation. "
                f"2. Recommend 2 more majors from 'all_available_majors' that best match the student's "
                f"dominant traits ({', '.join(top_traits)}) and career goals. "
                f"3. Explain WHY each major fits the student specifically. "
                f"4. NEVER recommend majors the student cannot qualify for based on Tawjihi score."
            )
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
