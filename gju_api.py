from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import httpx
 
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
 
GJU_KNOWLEDGE = """
GJU MAJORS KNOWLEDGE BASE:
 
Computer Science (School of Computing, min 75%):
- Best for: students who love coding, algorithms, AI, software development
- Key skills: programming (Python/Java/C++), data structures, algorithms, AI, databases, networks
- Careers: Software Engineer, AI Engineer, Data Scientist, Cybersecurity Analyst, Web Developer
 
Computer Engineering (School of Computing, min 80%):
- Best for: students who love both hardware and software, electronics, embedded systems
- Key skills: electronics, circuit design, embedded systems, IoT, networking
- Careers: Hardware Engineer, Embedded Systems Engineer, IoT Developer, Network Engineer
 
Business Intelligence and Data Analytics / BIDA (School of Computing, min 75%):
- Best for: students who love data, analytics, business + technology combined
- Key skills: SQL, Python/R, Power BI, machine learning, data visualization, ERP
- Careers: Data Analyst, BI Developer, Data Scientist, Business Analyst
 
Game Design and Media Informatics (School of Computing, min 70%):
- Best for: creative students who also love technology, gaming, 3D, animation
- Key skills: Unity/Unreal, 3D modeling, animation, UX/UI, filmmaking, programming
- Careers: Game Developer, 3D Artist, VR/AR Developer, UX Designer
 
Mechatronics Engineering (School of Applied Technical Sciences, min 80%):
- Best for: students who love robots, automation, AI in machines
- Key skills: control systems, electronics, robotics, AI/ML, PLC, IoT, Python/MATLAB
- Tracks: Applied AI OR Drones & Robotics
- Careers: Robotics Engineer, Automation Engineer, AI Systems Engineer, Drone Engineer
 
Industrial Engineering (School of Applied Technical Sciences, min 82%):
- Best for: students who love optimizing systems, manufacturing, operations
- Key skills: operations research, statistics, simulation, quality engineering, Python/R
- Careers: Industrial Engineer, Operations Manager, Quality Manager, Supply Chain Analyst
 
Mechanical and Maintenance Engineering (School of Applied Technical Sciences, min 80%):
- Best for: students who love machines, engines, manufacturing
- Key skills: thermodynamics, machine design, CAD, maintenance management, AI in maintenance
- Tracks: Thermal Systems, Automotive & E-Mobility, Additive Manufacturing
- Careers: Mechanical Engineer, Maintenance Engineer, Automotive Engineer
 
Management Science (School of Management, min 75%):
- Best for: students who enjoy leadership, business strategy, people management
- Key skills: business strategy, marketing, HR, economics, finance, project management
- Careers: Manager, Business Consultant, Marketing Manager, Entrepreneur
 
International Accounting (School of Management, min 75%):
- Best for: detail-oriented students who love finance and numbers for business purposes
- Key skills: accounting (IFRS), auditing, taxation, financial analysis
- Careers: Accountant, Auditor, Financial Analyst, CFO
 
Logistic Sciences (School of Management, min 75%):
- Best for: students interested in supply chains, global trade, operations
- Key skills: supply chain management, ERP, logistics planning, transportation
- Careers: Logistics Manager, Supply Chain Analyst, Procurement Specialist
 
Digital Marketing (School of Management, min 65%):
- Best for: creative + analytical students who love social media, branding, online campaigns
- Key skills: SEO/SEM, social media, content creation, analytics, multilingual marketing
- Careers: Digital Marketer, Social Media Manager, Content Strategist
 
Architecture (School of Architecture, min 75%, entrance exam required):
- Best for: highly creative students with strong spatial reasoning
- Key skills: architectural design, AutoCAD/Revit, structural systems, urban design
- Careers: Architect, Urban Planner, BIM Coordinator
 
Interior Architecture (School of Architecture, min 75%):
- Best for: students passionate about interior spaces and aesthetics
- Key skills: space planning, 3D visualization, color/materials, lighting design
- Careers: Interior Designer, Space Planner, Furniture Designer
 
Design and Visual Communication (School of Architecture, min 70%, entrance exam required):
- Best for: highly artistic students who love graphic design, branding, typography
- Key skills: Adobe Suite, branding, typography, motion graphics, UI/UX
- Careers: Graphic Designer, Art Director, Brand Designer, Motion Graphics Designer
 
Translation (School of Humanities, min 70%):
- Best for: multilingual students who love Arabic, English, German
- Key skills: translation theory, interpretation, legal/medical/technical translation
- Careers: Translator, Interpreter, Localization Specialist, Diplomat
 
GEBC - German and English for Business (School of Humanities, min 70%):
- Best for: students who love languages AND business combined
- Key skills: business communication in German/English, PR, marketing communication
- Careers: Corporate Communications, PR Manager, Export Manager
 
Nursing Science (School of Nursing, min 70%):
- Best for: empathetic students passionate about healthcare and helping people
- Key skills: clinical nursing, anatomy, pharmacology, patient care, German (8 levels)
- Note: 149 credit hours, 800 hours clinical internship in Germany
- Careers: Registered Nurse, ICU Nurse, Community Health Nurse
 
Pharmaceutical and Chemical Engineering (School of Applied Medical Sciences, min 80%):
- Best for: students who love chemistry, biology, pharmaceutical manufacturing
- Key skills: chemical process design, pharmaceutical technology, bioprocessing
- Careers: Chemical Engineer, Pharma Production Engineer, Process Engineer
 
Biomedical Engineering (School of Applied Medical Sciences, min 80%):
- Best for: students at the intersection of engineering and healthcare
- Key skills: biomedical signals, medical device design, biomechanics, clinical engineering
- Careers: Biomedical Engineer, Medical Device Designer, Clinical Engineer
 
IMPORTANT NOTES FOR ALL PROGRAMS:
- ALL GJU programs require a mandatory German year (6 months study + 6 months internship in Germany)
- Industrial Engineering requires minimum 82% Tawjihi
- Mechatronics, Mechanical, Computer Engineering, Pharma, Biomedical require minimum 80%
- Architecture and Design & Visual Communication require entrance exams
- Digital Marketing has the lowest requirement at 65%
"""
 
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
    career_goal: str = ""
    name: str = "Student"
 
class RecommendRequest(BaseModel):
    student: StudentInput
    predicted_school: str
    predicted_majors: list
    all_available: list
    riasec_scores: dict
    dominant_traits: list
 
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
        riasec = get_riasec_scores(student)
        dominant_traits = sorted(riasec.items(), key=lambda x: x[1], reverse=True)
        top_traits = [t[0] for t in dominant_traits[:2]]
        ranked = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)
 
        predicted_school = None
        predicted_majors = []
        predicted_confidence = 0
 
        for school, prob in ranked:
            majors = SCHOOL_MAJORS.get(school, [])
            available = filter_by_tawjihi(majors, student.tawjihi)
            if available:
                predicted_school = school
                predicted_majors = available
                predicted_confidence = prob
                break
 
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
            "predicted_school_majors": predicted_majors,
            "tawjihi_score": student.tawjihi,
            "riasec_scores": riasec,
            "dominant_traits": top_traits,
            "all_available_majors": all_available,
            "model_accuracy": "83%"
        }
    except Exception as e:
        return {"error": str(e)}
 
@app.post("/recommend")
async def recommend(req: RecommendRequest):
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        return {"error": "OpenAI key not configured"}
 
    prompt = f"""You are an academic advisor for German Jordanian University (GJU).
 
STUDENT PROFILE:
- Name: {req.student.name}
- Tawjihi Score: {req.student.tawjihi}%
- Career Goal: {req.student.career_goal}
- RIASEC Scores: {req.riasec_scores}
- Dominant Personality Traits: {req.dominant_traits}
 
ML MODEL PREDICTION (83% accuracy Random Forest):
- Predicted School: {req.predicted_school}
- Available majors in predicted school: {req.predicted_majors}
 
ALL AVAILABLE MAJORS (filtered by Tawjihi score):
{req.all_available}
 
GJU KNOWLEDGE BASE:
{GJU_KNOWLEDGE}
 
YOUR TASK:
Based on the student's complete profile, recommend exactly 3 majors:
1. PRIMARY: The best major from the ML-predicted school ({req.predicted_school})
2. SECOND: Best match from any school based on RIASEC profile and career goal
3. THIRD: Another good match from any school
 
For each recommendation provide:
- Major name and school
- Why it matches this specific student (reference their RIASEC scores and career goal)
- Key courses they will study
- Career paths available
 
IMPORTANT RULES:
- NEVER recommend majors requiring higher Tawjihi than the student has
- Base recommendations on the student's complete profile, not just one factor
- For tech/data students: prioritize Computing school majors
- For engineering/machines: prioritize Technical Sciences majors
- For creative students: prioritize Architecture school majors
- For business students: prioritize Management school majors
- For healthcare: prioritize Nursing/Medical Sciences
- For languages: prioritize Humanities school
- Remind the student that ALL GJU programs require a mandatory German year
 
Format your response in a clear, friendly, encouraging way. Use emojis sparingly."""
 
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 1000
                },
                timeout=30.0
            )
            data = response.json()
            recommendation = data["choices"][0]["message"]["content"]
            return {"recommendation": recommendation}
    except Exception as e:
        return {"error": str(e)}
 
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
