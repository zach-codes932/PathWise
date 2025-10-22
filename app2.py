import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import re
import os

# --- Google Gemini API integration ---
import google.generativeai as genai

# --- Enter your Gemini API key here (REQUIRED) ---
GEMINI_API_KEY = "AIzaSyBDB7CFOaTNcCQUPiAMk5uksfFKdX2J4Wg"  # Replace with your actual key

genai.configure(api_key=GEMINI_API_KEY)

# ---- App Configuration ----
st.set_page_config(page_title="AI Learning Path Generator", page_icon="🎓", layout="wide")
st.title("🎓 AI-Powered Personalized Learning Path Generator")

# ---- MCA Curriculum ----
MCA_CURRICULUM = {
    "Semester 1": [
        "Algorithm_&_Program_Design",
        "Digital_Logic_&_Computer_Architecture",
        "Database_Management_Systems",
        "Programming_with_Java",
        "Software_Engineering"
    ],
    "Semester 2": [
        "Theoretical_Computer_Science",
        "Advanced_Data_Structures",
        "Operating_Systems_&_Shell_Programming",
        "Data_Communication_&_Networks",
        "AI_&_ML"
    ],
    "Semester 3": [
        "Analysis_&_Design_of_Algorithms",
        "Software_Testing_&_QA",
        "Data_Mining_&_Warehousing",
        "Cyber_Security",
        "Cloud_Computing"
    ]
}
subject_list = [subj for sem in MCA_CURRICULUM.values() for subj in sem]

@st.cache_resource
def load_models_and_scalers():
    try:
        model_minors = tf.keras.models.load_model("WeaknessPredictor_MinorsOnly.h5")
        model_full = tf.keras.models.load_model("WeaknessPredictor_FullData.h5")
        scaler_minor = joblib.load("minmax_scaler_minor.pkl")
        scaler_full = joblib.load("minmax_scaler_full.pkl")
        return model_minors, model_full, scaler_minor, scaler_full
    except Exception as e:
        st.error(f"❌ Error loading models/scalers: {e}")
        return None, None, None, None

model_minors, model_full, scaler_minor, scaler_full = load_models_and_scalers()

# ---- Sidebar: User Inputs ----
st.sidebar.header("Enter Your Scores")
selected_semester = st.sidebar.selectbox("Select a Semester", options=list(MCA_CURRICULUM.keys()))
subjects_for_semester = MCA_CURRICULUM[selected_semester]
score_mode = st.sidebar.radio(
    "Which marks?", ["Minors Only", "Minors + EndSem"], horizontal=True
)
include_endsem = (score_mode == "Minors + EndSem")
entered_marks = {}

with st.sidebar.form(key="marks_form"):
    st.write(f"**Enter marks for {selected_semester}:**")
    for subject in subjects_for_semester:
        st.write(f"**{subject.replace('_', ' ')}**")
        col1, col2, col3 = st.columns(3)
        with col1:
            m1 = st.number_input("Minor 1", 0.0, 20.0, step=0.5, key=f"{subject}_m1")
        with col2:
            m2 = st.number_input("Minor 2", 0.0, 20.0, step=0.5, key=f"{subject}_m2")
        with col3:
            es = st.number_input("End Sem", 0.0, 60.0, step=0.5, key=f"{subject}_es") if include_endsem else 0
        entered_marks[subject] = {"Minor1": m1, "Minor2": m2, "EndSem": es}
    submitted = st.form_submit_button("🔍 Generate Learning Path")

def prepare_input_vector(subject_list, entered_marks, include_endsem):
    feats = []
    for subj in subject_list:
        if subj in entered_marks:
            feats.append(entered_marks[subj]["Minor1"])
            feats.append(entered_marks[subj]["Minor2"])
            if include_endsem:
                feats.append(entered_marks[subj]["EndSem"])
        else:
            feats.append(0)
            feats.append(0)
            if include_endsem:
                feats.append(0)
    return np.array(feats).reshape(1, -1)

def predict_weak_subjects_show_only_entered(model, scaler, subject_list, entered_marks, include_endsem):
    X = prepare_input_vector(subject_list, entered_marks, include_endsem)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)[0]
    pred_dict = {subj: preds[i] for i, subj in enumerate(subject_list)}
    out = {subj: pred_dict[subj] for subj in entered_marks}
    return out

def improved_prompt(subject, marks, semester):
    return (
        f"A student is weak in {subject.replace('_', ' ')} "
        f"(Minor1: {marks.get('Minor1', 0)}, Minor2: {marks.get('Minor2', 0)}, End Sem: {marks.get('EndSem', 0)}) for MCA {semester}.\n"
        "Give a practical 2-week improvement plan with:\n"
        "- Key topics to revise each week\n"
        "- Practice activities\n"
        "- Recommended online resources\n"
        "- Motivational advice\n"
        "Reply only with bullet points under '**Week 1**' and '**Week 2**'. Do not repeat the question or add any headers."
    )

def clean_llm_output(ai_plan):
    plan = re.split(r"\*\*Week 1\*\*", ai_plan, maxsplit=1)
    if len(plan) == 2:
        ai_plan = "**Week 1**" + plan[1]
    ai_plan = re.sub(r"(assistant[ :]*|AI mentor[ :]*)", "", ai_plan, flags=re.I)
    ai_plan = ai_plan.replace('\n- ', '\n\n- ')
    ai_plan = re.sub(r"\n\s*\n", "\n\n", ai_plan)
    ai_plan = ai_plan.strip()
    return ai_plan

def gemini_generate_plan(prompt):
    # Use Gemini 2.5 Pro Preview model for text generation
    model = genai.GenerativeModel('models/gemini-2.5-flash')
    response = model.generate_content(prompt)
    return response.text


def generate_llm_plan(weak_subjects, entered_marks, semester):
    if not weak_subjects:
        return "✅ Great job! No weak subjects detected."
    plan = "### Your Personalized AI-Generated Learning Path\n"
    for subject in weak_subjects:
        marks = entered_marks.get(subject.replace(' ', '_'), {})
        prompt = improved_prompt(subject, marks, semester)
        with st.spinner(f"Gemini is generating your personalized plan for {subject}..."):
            ai_raw = gemini_generate_plan(prompt)
        ai_plan = clean_llm_output(ai_raw)
        plan += f"\n\n#### {subject.replace('_',' ')}\n\n{ai_plan}\n\n---\n"
    plan += "*This plan was generated by Google Gemini API (free tier).*"
    return plan

if submitted:
    st.header("Analysis Results")
    marks_list = []
    for subject, scores in entered_marks.items():
        marks_list.append({
            "Subject": subject.replace("_", " "),
            "Minor1": scores["Minor1"],
            "Minor2": scores["Minor2"],
            "EndSem": scores["EndSem"] if include_endsem else None
        })
    marks_df = pd.DataFrame(marks_list).set_index("Subject")
    st.write("**Here are the scores you entered:**")
    st.dataframe(marks_df)

    with st.spinner("AI is analyzing your performance..."):
        if include_endsem:
            preds = predict_weak_subjects_show_only_entered(
                model_full, scaler_full, subject_list, entered_marks, include_endsem)
        else:
            preds = predict_weak_subjects_show_only_entered(
                model_minors, scaler_minor, subject_list, entered_marks, include_endsem)
        weak_subjects = [k.replace("_", " ") for k, v in preds.items() if v > 0.5]

    st.write("**Model Results (only for entered subjects):**")
    for subj, prob in preds.items():
        label = "❌ Weak" if prob > 0.5 else "✅ Not Weak"
        st.markdown(f"- {subj.replace('_', ' ')}: {label} (Confidence: {prob:.2f})")
    if not weak_subjects:
        st.success("No weak subjects detected for entered marks!")
    st.markdown("---")
    st.subheader("Your AI Learning Path")
    st.markdown(generate_llm_plan(weak_subjects, entered_marks, selected_semester))

st.caption("Powered by Google Gemini Pro API and Tensorflow. All logic runs securely in your app.")