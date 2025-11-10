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

# --- CONFIGURATION ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
st.title("CV Extractor & Evaluator (Optimized)")

CACHE_DIR = ".cv_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

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

    Extract: Full Name, Mobile Number, Email, Skills (8–15), Total Years of Experience,
    Education_1, Education_2, Certifications, LinkedIn URL.

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
        data = {k: "" for k in ["name","mobile","email","skills","experience_years","education_1","education_2","certifications","linkedin"]}
    return data

def evaluate_suitability(resume_data, requirement):
    prompt = f"""
    Evaluate this candidate for the following job requirement:
    Requirement: "{requirement}"

    Candidate:
    Name: {resume_data.get('name', '')}
    Skills: {resume_data.get('skills', '')}
    Experience: {resume_data.get('experience_years', '')} years
    Education: {resume_data.get('education_1', '')}; {resume_data.get('education_2', '')}
    Certifications: {resume_data.get('certifications', '')}

    Respond ONLY with a valid JSON object:
    {{
        "suitable": "Yes" or "No",
        "conclusion": "Brief reason for suitability or rejection"
    }}
    """
    response = openai.responses.create(model="gpt-3.5-turbo", input=prompt)
    result_text = response.output[0].content[0].text.strip()
    decision = safe_json_extract(result_text)
    if not decision:
        decision = {"suitable": "No", "conclusion": "Parsing error — AI did not return clean JSON"}
    return decision

def normalize_for_dataframe(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x) if x is not None else "")
    return df

# --- CACHE CLEAR BUTTON ---
if st.button("Clear cached resume data"):
    import shutil
    shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)
    st.success("✅ Cache cleared! All resumes will be reprocessed on next run.")

# --- MAIN APP ---
uploaded_files = st.file_uploader("Upload PDF resumes", type="pdf", accept_multiple_files=True)
requirement = st.text_input("Enter job requirement (e.g., 'Data analyst with SQL and Python')")

if uploaded_files and requirement:
    all_data = []
    total_tokens = 0
    progress_bar = st.progress(0)
    status_text = st.empty()

    def process_file(file):
        file_hash = hashlib.md5(file.read()).hexdigest()
        file.seek(0)
        cached = load_from_cache(file_hash)
        if cached:
            return cached, 0
        text = extract_text_from_pdf(file)
        if not text:
            return None, 0
        data = extract_resume_data(text)
        decision = evaluate_suitability(data, requirement)
        data["suitable"] = decision.get("suitable", "No")
        data["conclusion"] = decision.get("conclusion", "")
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

        # --- Pagination: 10 CVs per page ---
        rows_per_page = 10
        total_pages = (len(df) - 1) // rows_per_page + 1
        page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)

        start_idx = (page_number - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        st.dataframe(df.iloc[start_idx:end_idx].reset_index(drop=True), height=600)

        # --- Download CSV ---
        output_path = f"extracted_cv_data_{int(time.time())}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        st.download_button("Download CSV", data=open(output_path, "rb"), file_name=output_path)

        # --- Show tokens only ---
        st.info(f"Total tokens used: {total_tokens}")
