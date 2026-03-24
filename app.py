import os
import random
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
keys = [
    st.secrets["GROQ_API_KEY_1"],
    st.secrets["GROQ_API_KEY_2"],
    st.secrets["GROQ_API_KEY_3"]
]

client = Groq(api_key=random.choice(keys))

with open("prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()



st.set_page_config(page_title="Resume Optimizer", page_icon="📄", layout="wide")
st.title("📄 Resume Optimizer")
st.caption("Powered by Groq + llama-3.3-70b-versatile")
uploaded_file = st.file_uploader("Upload your resume (.txt)", type=["txt"])
jd = st.text_area("Paste Job Description here", height=300, placeholder="Copy the full job description and paste it here...")

if st.button("Optimize My Resume", type="primary"):
    if not uploaded_file:
        st.warning("Please upload your resume first.")
    elif not jd.strip():
        st.warning("Please paste a job description.")
    else:
        RESUME = uploaded_file.read().decode("utf-8")
        with st.spinner("Optimizing your resume..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Here is my resume:\n\n{RESUME}\n\nHere is the job description:\n\n{jd}\n\nOptimize my resume for this role."
                    }
                ],
                max_tokens=4000
            )
            result = response.choices[0].message.content

        st.success("Done!")
        st.markdown("### Optimized Resume")
        st.text_area("Copy your optimized resume below:", value=result, height=600)
        st.download_button(
            label="Download as .txt",
            data=result,
            file_name="optimized_resume.txt",
            mime="text/plain"
        )
password = st.text_input("Enter password", type="password")
if password != st.secrets["APP_PASSWORD"]:
    st.stop()
