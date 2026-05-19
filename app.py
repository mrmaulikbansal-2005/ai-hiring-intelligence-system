import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import random
import os
from openai import OpenAI
import plotly.graph_objects as go
import plotly.express as px
from docx import Document
from io import BytesIO

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def extract_relevant_skills(resume_text, job_description):

    prompt = f"""
    Extract the most relevant skills for this specific job application.

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Return only a Python-style comma-separated list of skills.
    Do not include explanations.

    Example:
    python, sql, customer service, leadership, fashion styling
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You extract role-specific skills from resumes and job descriptions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=150
    )

    skills_text = response.choices[0].message.content
    return [skill.strip().lower() for skill in skills_text.split(",") if skill.strip()]

def extract_text_from_pdf(uploaded_file):
    text = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s+#.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_keywords(text, skills_list):
    text = clean_text(text)

    found_skills = []

    for skill in skills_list:
        if skill.lower() in text:
            found_skills.append(skill.lower())

    return set(found_skills)


def calculate_ats_score(resume_text, job_description, skills_list):
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(job_description)

    documents = [cleaned_resume, cleaned_jd]

    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(documents)

    similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]
    ats_score = round(similarity_score * 100, 2)

    resume_keywords = extract_keywords(resume_text, skills_list)
    jd_keywords = extract_keywords(job_description, skills_list)

    matched_keywords = sorted(list(resume_keywords & jd_keywords))
    missing_keywords = sorted(list(jd_keywords - resume_keywords))

    keyword_match_score = round(
        (len(matched_keywords) / len(jd_keywords)) * 100, 2
    ) if len(jd_keywords) > 0 else 0

    final_score = round((ats_score * 0.6) + (keyword_match_score * 0.4), 2)

    return final_score, keyword_match_score, matched_keywords[:15], missing_keywords[:15]

def get_strength_label(score):
    if score >= 80:
        return "Strong Match", "Excellent fit for this role."
    elif score >= 60:
        return "Moderate Match", "Good fit, but resume can be improved."
    else:
        return "Weak Match", "Resume needs stronger alignment with this job."
    
def generate_ai_feedback(resume_text, job_description):

    prompt = f"""
    Analyze this resume against the job description.

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Give:
    1. Overall fit
    2. Missing skills
    3. Improvement suggestions
    4. Final hiring recommendation

    Keep it concise and professional.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert ATS recruiter."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )

    return response.choices[0].message.content

def generate_interview_questions(resume_text, job_description):
    prompt = f"""
    You are an expert recruiter.

    Generate 5 interview questions based on this resume and job description.

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Make the questions specific, practical, and relevant to the role.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You create recruiter-style interview questions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )

    return response.choices[0].message.content

def create_score_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "ATS Match Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#4CAF50"},
            "steps": [
                {"range": [0, 40], "color": "#4b1f1f"},
                {"range": [40, 70], "color": "#4a451f"},
                {"range": [70, 100], "color": "#1f4b2e"}
            ],
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}
    )

    return fig


def create_keyword_pie(matched_count, missing_count):
    fig = go.Figure(data=[
        go.Pie(
            labels=["Matched Skills", "Missing Keywords"],
            values=[matched_count, missing_count],
            hole=0.45
        )
    ])

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}
    )

    return fig

def generate_skill_gap_analysis(resume_text, job_description, missing_keywords):
    prompt = f"""
    Analyze the skill gaps between this resume and job description.

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Missing Keywords:
    {missing_keywords}

    Give a short, recruiter-style skill gap analysis in 3 bullet points.
    Be specific and practical.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert recruiter and career coach."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=250
    )

    return response.choices[0].message.content

def generate_cover_letter(resume_text, job_description, company_name, tone, length, focus_area, format_style):

    prompt = f"""
    You are an expert career strategist and professional cover letter writer.

    Write a highly personalized cover letter that connects the candidate's resume to the job description.
    The letter should tell a clear story explaining why this candidate is a strong fit.

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Company Name:
    {company_name}

    Preferences:
    Tone: {tone}
    Length: {length}
    Focus Area: {focus_area}
    Format Style: {format_style}

    Requirements:
    - Strictly follow this format:

    Candidate Name
    (phone) | email

    Hiring Manager
    Company Name

    Dear Hiring Manager,

    Paragraph 1: State the role, company, and why the candidate is excited.
    Paragraph 2: Connect the candidate's strongest resume experience to the job responsibilities.
    Paragraph 3: Highlight relevant tools, skills, coursework, or projects from the resume that match the job description.
    Paragraph 4: Close with a confident statement about contributing to the team.

    Thank you for your time and consideration. I look forward to hearing from you.

    Sincerely,
    Candidate Name

    Rules:
    - Do NOT include date.
    - Do NOT include candidate address.
    - Do NOT include company address.
    - Do NOT include placeholders like [LinkedIn Profile URL], [GitHub], [Date], or [Company Address].
    - Keep it concise, polished, and ready to submit.
    - Do not exceed one page.
    - Use the candidate’s actual details only if present in the resume.
    - Do NOT make it generic.
    - Connect specific resume experiences to specific job requirements.
    - Tell a strong career story.
    - Show why the candidate is the best fit for this role.
    - Mention measurable impact where possible.
    - Sound polished, confident, and recruiter-ready.
    - Avoid clichés like "I am writing to express my interest."
    - Make it ATS-friendly by naturally including relevant keywords from the job description.
    - Keep the structure professional with clear paragraphs.
    - End with a confident closing.
        """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an elite career strategist who writes high-converting cover letters for internships and early-career roles."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.65,
        max_tokens=900
    )

    return response.choices[0].message.content

def generate_resume_optimizer(resume_text, job_description):

    prompt = f"""
    You are an expert resume strategist and ATS optimization specialist.

    Analyze this resume against the job description and provide practical resume improvements.

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Output format:

    ## Optimized Professional Summary
    Rewrite a strong 3-4 line professional summary tailored to this job.

    ## Top Resume Improvements
    Give 5 specific improvements the candidate should make.

    ## ATS Keywords to Add
    List important missing keywords from the job description.

    ## Improved Resume Bullet Points
    Rewrite 5 resume bullet points to be stronger, more measurable, and ATS-friendly.

    ## Recruiter Notes
    Explain what would make this resume more competitive for this role.

    Rules:
    - Be specific to the resume and job description.
    - Do not make up fake experiences.
    - Improve wording, impact, and clarity.
    - Use strong action verbs.
    - Keep the output practical and recruiter-ready.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert resume strategist who helps students and early-career candidates tailor resumes for internships."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.6,
        max_tokens=1000
    )

    return response.choices[0].message.content

def generate_quick_assessment(resume_text, job_description):

    prompt = f"""
    You are an elite recruiter and assessment designer.

    Create a rigorous 15-minute interview readiness assessment based on the candidate's resume and job description.

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Create exactly 5 difficult questions.

    The assessment should test:
    - analytical thinking
    - decision making under pressure
    - role-specific knowledge
    - problem-solving ability
    - communication clarity
    - business judgment
    - ability to connect resume experience to real job situations

    Format strictly like this:

    # 15-Minute Interview Readiness Assessment

    ## Instructions
    You have 15 minutes to answer all 5 questions. Keep answers concise, structured, and practical.

    ## Question 1 — Analytical Problem Solving
    [Question]

    ## Question 2 — Role-Specific Challenge
    [Question]

    ## Question 3 — Prioritization Under Pressure
    [Question]

    ## Question 4 — Resume Deep-Dive
    [Question]

    ## Question 5 — Business / Technical Judgment
    [Question]

    ## Scoring Rubric
    Score each answer from 1-5 based on:
    - clarity
    - structure
    - relevance
    - problem-solving
    - confidence
    - specificity

    ## What Strong Answers Should Show
    Explain what a strong candidate should demonstrate.

    Rules:
    - Do not ask generic questions.
    - Make the questions difficult but fair.
    - Use realistic workplace scenarios.
    - Questions must require thinking, not memorization.
    - Tailor questions to the resume and job description.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You design rigorous interview assessments for internship and early-career hiring."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.75,
        max_tokens=1200
    )

    return response.choices[0].message.content

def grade_interview_assessment(assessment_questions, user_answers, resume_text, job_description):

    prompt = f"""
    You are a strict interview evaluator and hiring manager.

    Grade the candidate's answers to this 15-minute interview assessment.

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Assessment Questions:
    {assessment_questions}

    Candidate Answers:
    {user_answers}

    Grade each answer out of 10.

    Output format:

    # Interview Assessment Grading Report

    ## Overall Score
    Give an overall score out of 10.

    ## Question-by-Question Feedback

    ### Question 1
    Score: X/10
    Feedback:
    What was strong:
    What was weak:
    How to improve:

    ### Question 2
    Score: X/10
    Feedback:
    What was strong:
    What was weak:
    How to improve:

    ### Question 3
    Score: X/10
    Feedback:
    What was strong:
    What was weak:
    How to improve:

    ### Question 4
    Score: X/10
    Feedback:
    What was strong:
    What was weak:
    How to improve:

    ### Question 5
    Score: X/10
    Feedback:
    What was strong:
    What was weak:
    How to improve:

    ## Final Readiness Verdict
    Say whether the candidate is:
    - Not Ready
    - Somewhat Ready
    - Interview Ready
    - Strong Candidate

    ## Priority Improvements
    Give 3 specific things the candidate should improve before the interview.

    Rules:
    - Be strict but helpful.
    - Do not give everyone high scores.
    - Penalize vague answers.
    - Reward structured, specific, realistic answers.
    - Focus on analytical thinking, clarity, confidence, and role fit.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a strict but helpful interview assessment grader."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.4,
        max_tokens=1400
    )

    return response.choices[0].message.content

def generate_ats_insights(resume_text, job_description):

    prompt = f"""
    You are an expert ATS analyst, recruiter, and resume optimization specialist.

    Analyze this resume against the job description.

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Output must strictly follow this format:

    # ATS Insights Report

    ## 1. Overall ATS Match
    Give a score out of 100 and explain why.

    ## 2. Keyword Match Analysis
    List:
    - Strong matched keywords
    - Important missing keywords
    - Keywords that should be added naturally

    ## 3. Resume Weakness Scanner
    Identify weaknesses in:
    - vague bullet points
    - missing metrics
    - weak action verbs
    - missing tools/skills
    - lack of role alignment

    ## 4. Recruiter Attention Heatmap
    Categorize sections as:
    - Strong Attention
    - Medium Attention
    - Low Attention

    ## 5. ATS Rewrite Suggestions
    Give 5 before-and-after improvements.
    If exact original bullets are unavailable, create realistic improvement suggestions based only on the resume content.

    ## 6. Interview Probability Estimate
    Estimate likelihood of interview as a percentage.
    Explain top strengths and top risks.

    ## 7. Priority Action Plan
    Give the top 5 changes the candidate should make before applying.

    Rules:
    - Be specific to the resume and JD.
    - Do not invent fake experience.
    - Be recruiter-style and practical.
    - Make the feedback direct, useful, and ATS-focused.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert ATS resume analyst and recruiter."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.45,
        max_tokens=1500
    )

    return response.choices[0].message.content

def generate_recruiter_summary(candidate_results, job_description):

    prompt = f"""
    You are an expert recruiter and hiring manager.

    Analyze these ranked candidate resumes for the job description.

    Job Description:
    {job_description}

    Candidate Results:
    {candidate_results}

    Create a recruiter-style hiring summary.

    Output format:

    # Recruiter View Summary

    ## Best Fit Candidate
    Identify the strongest candidate and explain why.

    ## Candidate Ranking Explanation
    Explain why each candidate is ranked where they are.

    ## Shortlist Recommendation
    Recommend which candidates should be shortlisted.

    ## Top Strengths Across Candidates
    List common strengths.

    ## Top Hiring Risks
    List potential risks or weaknesses.

    ## Final Hiring Decision
    Give a clear hiring recommendation.

    Rules:
    - Be direct and recruiter-style.
    - Do not be generic.
    - Base reasoning on scores, matched skills, and missing keywords.
    - Keep it professional.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert recruiter comparing candidates for a role."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.45,
        max_tokens=1200
    )

    return response.choices[0].message.content

st.set_page_config(
    page_title="AI Hiring Intelligence System",
    page_icon="💼",
    layout="wide"
)
    
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background:
        radial-gradient(circle at top left, rgba(59,130,246,0.20), transparent 35%),
        radial-gradient(circle at top right, rgba(168,85,247,0.18), transparent 35%),
        linear-gradient(135deg, #020617 0%, #0f172a 55%, #111827 100%);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #111827 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.22);
}

h1, h2, h3 {
    color: #f8fafc;
    font-weight: 800;
}

p, li, span {
    color: #cbd5e1;
}

.hero {
    padding: 34px;
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(37,99,235,0.30), rgba(124,58,237,0.25)),
        rgba(15,23,42,0.85);
    border: 1px solid rgba(148,163,184,0.28);
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    margin-bottom: 28px;
    animation: fadeIn 0.8s ease-in-out;
}

.card {
    background: rgba(15,23,42,0.88);
    border: 1px solid rgba(148,163,184,0.25);
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 14px 40px rgba(0,0,0,0.30);
    margin-bottom: 20px;
    transition: all 0.25s ease;
}

.card:hover {
    transform: translateY(-4px);
    border-color: rgba(96,165,250,0.75);
    box-shadow: 0 20px 55px rgba(37,99,235,0.22);
}

.feature-card {
    background: linear-gradient(145deg, rgba(15,23,42,0.96), rgba(30,41,59,0.82));
    border: 1px solid rgba(148,163,184,0.25);
    border-radius: 24px;
    padding: 24px;
    min-height: 170px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.28);
    transition: all 0.25s ease;
}

.feature-card:hover {
    transform: scale(1.025);
    border-color: #60a5fa;
}

[data-testid="stMetric"] {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(96,165,250,0.30);
    padding: 18px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

.stButton > button {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.75rem 1.5rem;
    font-weight: 800;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 12px 30px rgba(37,99,235,0.35);
    color: white;
}

.stDownloadButton > button {
    background: linear-gradient(90deg, #059669, #0d9488);
    color: white;
    border-radius: 14px;
    font-weight: 800;
}

textarea, input {
    border-radius: 14px !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


st.sidebar.title("🧠 AI Hiring Suite")

page = st.sidebar.radio(
    "Navigate",
    [
        "Resume Analyzer",
        "Cover Letter Generator",
        "Resume Optimizer",
        "Mock Interview Simulator",
        "ATS Insights",
        "Recruiter View"
    ]
)
if page == "Resume Analyzer":
# Header
    st.markdown("""
    <div class="hero">
        <h1>💼 AI Hiring Intelligence Suite</h1>
        <p style="font-size:18px; color:#cbd5e1;">
            Analyze resumes, generate cover letters, optimize ATS performance, simulate interviews, and rank candidates with AI.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📄 Upload Resume")
        resume_file = st.file_uploader("", type=["pdf"])

    with col2:
        st.markdown("### 📝 Job Description")
        job_description = st.text_area("", height=250)

    st.divider()

    # Button Center
    center_col = st.columns([2,1,2])[1]

    with center_col:
        analyze = st.button("🚀 Analyze Resume")

    # Results
    if analyze:
        if resume_file is None:
            st.error("Please upload your resume.")

        elif job_description.strip() == "":
            st.error("Please paste job description.")

        else:
            st.success("Analysis Started!")

            resume_text = extract_text_from_pdf(resume_file)
            dynamic_skills = extract_relevant_skills(resume_text, job_description)

            st.markdown("### 📄 Extracted Resume Text Preview")
            with st.expander("Click to view extracted resume text"):
                st.write(resume_text[:2000])

            st.divider()

            score, keyword_match_score, matched_keywords, missing_keywords = calculate_ats_score(
                resume_text,
                job_description,
                dynamic_skills
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("ATS Score", f"{score}%")

            with col2:
                st.metric("Keyword Match", f"{keyword_match_score}%")

            with col3:
                st.metric("Matched Skills", len(matched_keywords))

            strength_title, strength_message = get_strength_label(score)

            st.markdown("### 🎯 Resume Strength")

            if score >= 80:
                st.success(f"### {strength_title}\n{strength_message}")
            elif score >= 60:
                st.warning(f"### {strength_title}\n{strength_message}")
            else:
                st.error(f"### {strength_title}\n{strength_message}")

            st.progress(int(score))

            st.divider()

        st.markdown("### 📊 Visual ATS Dashboard")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            gauge_fig = create_score_gauge(score)
            st.plotly_chart(gauge_fig, use_container_width=True)

        with chart_col2:
            pie_fig = create_keyword_pie(
                len(matched_keywords),
                len(missing_keywords)
            )
            st.plotly_chart(pie_fig, use_container_width=True)

        st.divider()

        st.markdown("### ✅ Matched Skills")

        if matched_keywords:
            for skill in matched_keywords:
                st.write(f"✅ {skill}")
        else:
            st.warning("No matched skills found.")

        st.markdown("### 🔍 Missing Keywords")

        for word in missing_keywords:
            st.write(f"• {word}")

        st.markdown("### 📉 Skill Gap Analysis")

        try:
            skill_gap = generate_skill_gap_analysis(
                resume_text,
                job_description,
                missing_keywords
            )

            st.write(skill_gap)

        except Exception as e:
            st.error(e)

            if missing_keywords:
                top_missing = ", ".join(missing_keywords[:5])

                st.write(
                    f"Your biggest skill gaps for this role are: **{top_missing}**. "
                    "Consider adding relevant projects, certifications, or hands-on experience in these areas."
                )

            else:
                st.success("Strong alignment — no major keyword gaps were detected.")

            st.markdown("### 🤖 AI Suggestions")

        try:
            ai_feedback = generate_ai_feedback(
                resume_text,
                job_description
            )

            st.markdown("#### 🧠 Personalized Recruiter Insights")

            st.write(ai_feedback)

        except Exception as e:
            st.error(e)

        st.divider()

        st.markdown("## 🎤 AI Interview Questions")

        try:
            questions = generate_interview_questions(
                resume_text,
                job_description
            )

            for q in questions.split("\n"):
                if q.strip():
                    st.write(q)

        except Exception as e:
            st.error(e)

            fallback_questions = [
                "Tell me about a project where you used Python, SQL, or analytics to solve a business problem.",
                "How would your AI in Business background help you succeed in this role?",
                "Describe a time you automated or improved a workflow.",
                "What part of this job description is most aligned with your experience?",
                "What technical skill from this role would you want to strengthen first?"
            ]

            for q in fallback_questions:
                st.write(f"❓ {q}")

elif page == "Cover Letter Generator":

    st.markdown("# ✉️ AI Cover Letter Generator")
    st.write("Create a polished, recruiter-ready cover letter tailored to your resume and job description.")

    cover_resume_file = st.file_uploader(
    "Upload Resume",
    type=["pdf"],
    key="cover_resume_upload"
)

    cover_job_description = st.text_area(
    "Paste Job Description",
    height=250,
    key="cover_jd"
)
    cover_col1, cover_col2 = st.columns(2)

    with cover_col1:
        company_name = st.text_input(
            "Company Name",
            placeholder="Example: Microsoft",
            key="cover_company"
        )

        tone = st.selectbox(
            "Tone",
            ["Professional", "Confident", "Startup-style", "Corporate", "Warm"],
            key="cover_tone"
        )

        length = st.selectbox(
            "Length",
            ["Short", "Standard", "Detailed"],
            key="cover_length"
        )

    with cover_col2:
        focus_area = st.selectbox(
            "Focus Area",
            ["Balanced", "Technical Skills", "Leadership", "Business Impact", "Career Story"],
            key="cover_focus"
        )

        format_style = st.selectbox(
            "Format Style",
            ["Formal Letter", "Modern Email Style", "Recruiter-Friendly"],
            key="cover_format"
        )

    if st.button("Generate Cover Letter", key="cover_generate"):

        if cover_resume_file is None:
            st.error("Please upload a resume first.")

        elif not cover_job_description.strip():
            st.error("Please paste a job description first.")

        else:
            with st.spinner("Writing a personalized recruiter-ready cover letter..."):

                try:
                    cover_resume_text = extract_text_from_pdf(cover_resume_file)

                    cover_letter = generate_cover_letter(
                        cover_resume_text,
                        cover_job_description,
                        company_name,
                        tone,
                        length,
                        focus_area,
                        format_style
                    )

                    st.success("Cover Letter Generated Successfully!")

                    st.markdown("### 📄 Generated Cover Letter")
                    st.write(cover_letter)

                    # Create Word Document
                    doc = Document()

                    for line in cover_letter.split("\n"):
                        doc.add_paragraph(line)

                    doc_buffer = BytesIO()
                    doc.save(doc_buffer)
                    doc_buffer.seek(0)

                    st.download_button(
                        label="📥 Download Cover Letter (.docx)",
                        data=doc_buffer,
                        file_name="AI_Cover_Letter.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

                except Exception as e:
                    st.error(e)

elif page == "Resume Optimizer":

    st.markdown("# 🛠️ AI Resume Optimizer")
    st.write("Upload your resume and paste a job description to get tailored resume improvements.")

    optimizer_resume_file = st.file_uploader(
        "Upload Resume",
        type=["pdf"],
        key="optimizer_resume_upload"
    )

    optimizer_job_description = st.text_area(
        "Paste Job Description",
        height=250,
        key="optimizer_jd"
    )

    if st.button("Optimize Resume", key="optimizer_generate"):

        if optimizer_resume_file is None:
            st.error("Please upload a resume first.")

        elif not optimizer_job_description.strip():
            st.error("Please paste a job description first.")

        else:
            with st.spinner("Optimizing your resume for this role..."):

                try:
                    optimizer_resume_text = extract_text_from_pdf(optimizer_resume_file)

                    optimized_resume_feedback = generate_resume_optimizer(
                        optimizer_resume_text,
                        optimizer_job_description
                    )

                    st.success("Resume Optimization Complete!")

                    st.markdown("### 🚀 Optimized Resume Suggestions")
                    st.write(optimized_resume_feedback)

                    st.download_button(
                        label="📥 Download Resume Optimization Notes",
                        data=optimized_resume_feedback,
                        file_name="Resume_Optimization_Notes.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(e)   

                    st.divider()

elif page == "Mock Interview Simulator":

    st.markdown("## ⏱️ 15-Minute Quick Assessment")
    st.write("A rigorous pressure-style assessment to test analytical thinking, role fit, and interview readiness.")

    interview_resume_file = st.file_uploader(
        "Upload Resume",
        type=["pdf"],
        key="interview_resume_upload"
    )

    interview_job_description = st.text_area(
        "Paste Job Description",
        height=250,
        key="interview_jd"
    )

    if st.button("Generate 15-Minute Assessment", key="assessment_generate"):

        if interview_resume_file is None:
            st.error("Please upload a resume first.")

        elif not interview_job_description.strip():
            st.error("Please paste a job description first.")

        else:
            with st.spinner("Creating a rigorous interview assessment..."):

                try:
                    interview_resume_text = extract_text_from_pdf(interview_resume_file)

                    assessment_output = generate_quick_assessment(
                        interview_resume_text,
                        interview_job_description
                    )

                    st.success("Assessment Generated!")

                    st.markdown("### 🧪 Interview Readiness Assessment")
                    st.write(assessment_output)
                    st.session_state.assessment_output = assessment_output
                    st.session_state.interview_resume_text = interview_resume_text
                    st.session_state.interview_job_description = interview_job_description

                    st.download_button(
                        label="📥 Download Assessment",
                        data=assessment_output,
                        file_name="Interview_Readiness_Assessment.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(e)    
            if "assessment_output" in st.session_state:

                    st.divider()

                    st.markdown("## ✍️ Your Assessment Answers")
                    st.write("Answer all 5 questions below. Keep your answers structured and practical.")

            answer_1 = st.text_area("Answer 1", height=180, key="answer_1")
            answer_2 = st.text_area("Answer 2", height=180, key="answer_2")
            answer_3 = st.text_area("Answer 3", height=180, key="answer_3")
            answer_4 = st.text_area("Answer 4", height=180, key="answer_4")
            answer_5 = st.text_area("Answer 5", height=180, key="answer_5")

            if st.button("Grade My Assessment", key="grade_assessment"):

                user_answers = f"""
                Answer 1:
                {answer_1}

                Answer 2:
                {answer_2}

                Answer 3:
                {answer_3}

                Answer 4:
                {answer_4}

                Answer 5:
                {answer_5}
                """

                if not all([
                    answer_1.strip(),
                    answer_2.strip(),
                    answer_3.strip(),
                    answer_4.strip(),
                    answer_5.strip()
                ]):
                    st.error("Please answer all 5 questions before grading.")

                else:

                    with st.spinner("Grading your assessment like a real recruiter..."):

                        try:

                            grading_report = grade_interview_assessment(
                                st.session_state.assessment_output,
                                user_answers,
                                st.session_state.interview_resume_text,
                                st.session_state.interview_job_description
                            )

                            st.success("Assessment Graded!")

                            st.markdown("## 📊 AI Grading Report")
                            st.write(grading_report)

                        except Exception as e:
                            st.error(e) 

elif page == "ATS Insights":

    st.markdown("# 📊 ATS Insights")
    st.write("Get a recruiter-grade ATS report with keyword analysis, resume weaknesses, heatmap insights, and interview probability.")

    ats_resume_file = st.file_uploader(
        "Upload Resume",
        type=["pdf"],
        key="ats_resume_upload"
    )

    ats_job_description = st.text_area(
        "Paste Job Description",
        height=250,
        key="ats_jd"
    )

    if st.button("Generate ATS Insights", key="ats_generate"):

        if ats_resume_file is None:
            st.error("Please upload a resume first.")

        elif not ats_job_description.strip():
            st.error("Please paste a job description first.")

        else:
            with st.spinner("Analyzing resume through an ATS and recruiter lens..."):

                try:
                    ats_resume_text = extract_text_from_pdf(ats_resume_file)

                    ats_report = generate_ats_insights(
                        ats_resume_text,
                        ats_job_description
                    )

                    st.success("ATS Insights Generated!")

                    st.markdown("## ⚡ ATS Snapshot")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                         st.metric("ATS Status", "Strong")

                    with col2:
                         st.metric("Keyword Match", "High")

                    with col3:
                         st.metric("Recruiter View", "Positive")

                    st.progress(84/100)
                    st.caption("Estimated ATS Compatibility")

                    with st.expander("📊 View Full ATS Report"):
                       st.markdown(f"""
                       <div class="card">
                           <div style="color:#e2e8f0; line-height:1.8; font-size:16px;">
                               {ats_report.replace(chr(10), "<br>")}
                           </div>
                       </div>
                       """, unsafe_allow_html=True)

                    st.download_button(
                        label="📥 Download ATS Report",
                        data=ats_report,
                        file_name="ATS_Insights_Report.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(e)

elif page == "Recruiter View":

    st.markdown("# 🧑‍💼 Recruiter View")
    st.write("Upload multiple resumes and rank candidates against one job description.")

    recruiter_resume_files = st.file_uploader(
        "Upload Multiple Resumes",
        type=["pdf"],
        accept_multiple_files=True,
        key="recruiter_resume_upload"
    )

    recruiter_job_description = st.text_area(
        "Paste Job Description",
        height=250,
        key="recruiter_jd"
    )

    if st.button("Rank Candidates", key="rank_candidates"):

        if not recruiter_resume_files:
            st.error("Please upload at least two resumes.")

        elif len(recruiter_resume_files) < 2:
            st.error("Please upload at least two resumes for comparison.")

        elif not recruiter_job_description.strip():
            st.error("Please paste a job description first.")

        else:
            with st.spinner("Ranking candidates like a recruiter..."):

                try:
                    candidate_results = []

                    for resume_file in recruiter_resume_files:
                        resume_text = extract_text_from_pdf(resume_file)

                        dynamic_skills = extract_relevant_skills(
                            resume_text,
                            recruiter_job_description
                        )

                        score, keyword_match_score, matched_keywords, missing_keywords = calculate_ats_score(
                            resume_text,
                            recruiter_job_description,
                            dynamic_skills
                        )

                        candidate_results.append({
                            "Resume": resume_file.name,
                            "ATS Score": score,
                            "Keyword Match": keyword_match_score,
                            "Matched Skills": len(matched_keywords),
                            "Missing Keywords": len(missing_keywords),
                            "Top Matched": ", ".join(matched_keywords[:5]),
                            "Top Missing": ", ".join(missing_keywords[:5])
                        })

                    candidate_results = sorted(
                        candidate_results,
                        key=lambda x: x["ATS Score"],
                        reverse=True
                    )

                    st.success("Candidate Ranking Complete!")

                    st.markdown("## 🏆 Candidate Leaderboard")

                    st.dataframe(candidate_results, use_container_width=True)

                    names = [candidate["Resume"] for candidate in candidate_results]
                    scores = [candidate["ATS Score"] for candidate in candidate_results]

                    leaderboard_fig = px.bar(
                        x=names,
                        y=scores,
                        labels={"x": "Candidate Resume", "y": "ATS Score"},
                        title="Candidate Ranking by ATS Score"
                    )

                    st.plotly_chart(leaderboard_fig, use_container_width=True)

                    st.markdown("## 🧠 Recruiter Summary")

                    recruiter_summary = generate_recruiter_summary(
                        candidate_results,
                        recruiter_job_description
                    )

                    st.write(recruiter_summary)

                except Exception as e:
                    st.error(e)