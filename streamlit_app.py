import os
import re
import json
import time
import hashlib
import pandas as pd
import openai
from PyPDF2 import PdfReader
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
from PIL import Image

# --- CONFIGURATION ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

CACHE_DIR = ".cv_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- SIDEBAR OPTIONS ---
show_summary = st.sidebar.checkbox("Show 1-line summary for each CV", value=False)

# --- HELPER FUNCTIONS ---
def extract_text_from_pdf(file):
    text = ""
    try:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.warning(f"⚠️ Error reading {file.name}: {e}")
    return text.strip()

def preprocess_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def safe_json_extract(text):
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            cleaned = re.sub(r"[\n\r\t]+", " ", match.group())
            cleaned = re.sub(r"[^{}:,\"'\w\s.-]", "", cleaned)
            try:
                return json.loads(cleaned)
            except Exception:
                pass
    return None

def cache_result(file_hash, data):
    path = os.path.join(CACHE_DIR, f"{file_hash}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def load_from_cache(file_hash):
    path = os.path.join(CACHE_DIR, f"{file_hash}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def extract_resume_data(text):
    text = preprocess_text(text)
    prompt = f"""
You are an expert AI CV parser. Extract structured information from this resume text.

1. Extract Full Name, Mobile Number, Email.
2. Extract Skills (8–15), but you can **infer relevant skills** from work experience, tools, and projects even if not explicitly listed.
3. Extract Total Years of Experience.
4. Extract Education_1 (highest degree) and Education_2 (second highest or additional degree if any).
5. Extract Certifications.
6. Extract LinkedIn URL.

Return a valid JSON object like:
{{
    "name": "",
    "mobile": "",
    "email": "",
    "skills": "",
    "experience_years": "",
    "education_1": "",
    "education_2": "",
    "certifications": "",
    "linkedin": ""
}}

Resume Text:
{text}
"""
    response = openai.responses.create(model="gpt-4o-mini", input=prompt)
    result_text = response.output[0].content[0].text.strip()
    data = safe_json_extract(result_text)
    if not data:
        data = {k: "" for k in ["name","mobile","email","skills","experience_years",
                                "education_1","education_2","certifications","linkedin"]}
    return data

def evaluate_suitability(resume_data, job_requirement):
    prompt = f"""
Evaluate this candidate against the following job requirements and notes:

Mandatory Skills: {job_requirement['mandatory_skills']}
Nice-to-have Skills: {job_requirement['nice_to_have_skills']}
Required Certifications: {job_requirement['required_certifications']}
Nice-to-have Certifications: {job_requirement['nice_to_have_certifications']}
Required Education: {job_requirement['required_education']}
Nice-to-have Education: {job_requirement['nice_to_have_education']}
Experience Range: {job_requirement['experience_range']}

Job Notes / Description:
{job_requirement.get('notes_description', '')}

Candidate:
Name: {resume_data.get('name', '')}
Skills: {resume_data.get('skills', '')}
Experience: {resume_data.get('experience_years', '')} years
Education: {resume_data.get('education_1', '')}; {resume_data.get('education_2', '')}
Certifications: {resume_data.get('certifications', '')}

Respond ONLY with a valid JSON object:
{{
    "suitable": "Yes", "No", or "Manual Review",
    "conclusion": "Brief reason for suitability, rejection, or why manual review is needed"
}}
"""
    response = openai.responses.create(model="gpt-3.5-turbo", input=prompt)
    result_text = response.output[0].content[0].text.strip()
    decision = safe_json_extract(result_text)
    if not decision:
        decision = {"suitable": "Manual Review", "conclusion": "Parsing error — AI did not return clean JSON"}
    elif decision.get("suitable") not in ["Yes", "No", "Manual Review"]:
        decision["suitable"] = "Manual Review"
    return decision

def normalize_for_dataframe(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: ", ".join([str(i) for i in x]) if isinstance(x, list) else str(x) if x is not None else "")
    return df

# Load the logo
logo = Image.open("logo.png")
st.image(logo, width=500)

# --- PAGE TITLE ---
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 10px;">
    <h1 style="margin: 0;">CV Extractor & Evaluator (Enhanced)</h1>
</div>
""", unsafe_allow_html=True)

# --- FLOATING SMALL BUTTONS ---
st.markdown("""
<style>
.floating-btns {
    position: fixed;
    top: 10px;
    left: 10px;
    z-index: 1000;
    display: flex;
    gap: 0.25rem;
}
.floating-btns button {
    padding: 0.25rem 0.5rem;
    font-size: 0.75rem;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="floating-btns">', unsafe_allow_html=True)

if st.button("Clear Cache", key="clear_cache_btn"):
    shutil.rmtree(".cv_cache")
    os.makedirs(".cv_cache", exist_ok=True)
    st.success("✅ Cache cleared! All resumes will be reprocessed on next run.")

if st.button("Clear Form", key="clear_form_btn"):
    for key in ["mandatory_skills","nice_to_have_skills","required_certs","nice_to_have_certs",
                "required_education","nice_to_have_education","experience_min","experience_max","notes_description"]:
        st.session_state[key] = ""
    st.success("✅ Form cleared!")

st.markdown('</div>', unsafe_allow_html=True)

# --- MODE SELECTION ---
st.subheader("Select Mode")
mode = st.radio(
    "Choose what you want to do:",
    ("Extract CV Details Only", "Extract + Screen CVs")
)

# --- JOB REQUIREMENT INPUT (only if Extract + Screen) ---
job_requirement = {}
if mode == "Extract + Screen CVs":
    st.subheader("Job Requirement Details")
    mandatory_skills = st.text_input("Mandatory Skills (comma-separated)")
    nice_to_have_skills = st.text_input("Nice-to-have Skills (comma-separated)")
    required_certs = st.text_input("Required Certifications (comma-separated)")
    nice_to_have_certs = st.text_input("Nice-to-have Certifications (comma-separated)")
    required_education = st.text_input("Required Education (comma-separated)")
    nice_to_have_education = st.text_input("Nice-to-have Education (comma-separated)")
    experience_min = st.number_input("Minimum Experience (years)", min_value=0, step=1)
    experience_max = st.number_input("Maximum Experience (years)", min_value=0, step=1)
    notes_description = st.text_area("Job Notes / Description")

    job_requirement = {
        "mandatory_skills": mandatory_skills,
        "nice_to_have_skills": nice_to_have_skills,
        "required_certifications": required_certs,
        "nice_to_have_certifications": nice_to_have_certs,
        "required_education": required_education,
        "nice_to_have_education": nice_to_have_education,
        "experience_range": f"{experience_min}-{experience_max}",
        "notes_description": notes_description
    }

# --- MAIN APP ---
# --- MAIN APP ---
uploaded_files = st.file_uploader("Upload PDF resumes", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.info(f"{len(uploaded_files)} file(s) uploaded. Click 'Run' to process.")
    
    if st.button("Run"):
        all_data = []
        total_tokens = 0
        seen_hashes = set()
        progress_bar = st.progress(0)
        status_text = st.empty()

        def process_file(file):
            file_hash = hashlib.md5(file.read()).hexdigest()
            file.seek(0)
            if file_hash in seen_hashes:
                return None, 0
            seen_hashes.add(file_hash)
            cached = load_from_cache(file_hash)
            if cached:
                return cached, 0
            text = extract_text_from_pdf(file)
            if not text:
                return None, 0
            data = extract_resume_data(text)

            # Only run suitability evaluation if mode is "Extract + Screen CVs"
            if mode == "Extract + Screen CVs":
                decision = evaluate_suitability(data, job_requirement)
                data["suitable"] = decision.get("suitable", "Manual Review")
                data["conclusion"] = decision.get("conclusion", "")
            else:
                data["suitable"] = ""
                data["conclusion"] = ""

            if show_summary:
                data["summary"] = f"{data.get('experience_years', '')} yrs, Skills: {data.get('skills', '')}, Education: {data.get('education_1', '')}/{data.get('education_2', '')}"
            cache_result(file_hash, data)
            return data, 1250

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_file, f): f for f in uploaded_files}
            for i, future in enumerate(as_completed(futures)):
                result, tokens = future.result()
                if result:
                    all_data.append(result)
                    total_tokens += tokens
                progress_bar.progress((i+1)/len(uploaded_files))
                status_text.text(f"Processed {i+1}/{len(uploaded_files)} resumes")

        if all_data:
            df = pd.DataFrame(all_data)
            df = normalize_for_dataframe(df)
            st.subheader("Extracted CV Data")
            height_px = min(1200, len(df) * 40)
            st.dataframe(df, height=height_px)

            # --- Download CSV ---
            output_path = f"extracted_cv_data_{int(time.time())}.csv"
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            st.download_button("Download CSV", data=open(output_path, "rb"), file_name=output_path)

            st.info(f"Total tokens used: {total_tokens}")