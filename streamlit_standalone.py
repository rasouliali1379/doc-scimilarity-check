import os
import numpy as np
import streamlit as st
import subprocess
import json
import tempfile

# Set page configuration for RTL
st.set_page_config(
    page_title="سامانه بررسی شباهت اسناد",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply RTL CSS
st.markdown("""
<style>
    .main, .sidebar, .stMarkdown, .stButton, .stTextInput, .stFileUploader, .stHeader, .stExpander {
        direction: rtl;
        text-align: right;
    }
    .stExpander .streamlit-expanderContent {
        direction: rtl;
        text-align: right;
    }
    button {
        float: right;
    }
    div[data-testid="stFileUploader"] label {
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit GUI with Persian labels
def main():
    st.title("سامانه بررسی شباهت اسناد")
    st.markdown("یک سند جدید را بارگذاری کنید و آن را با اسناد موجود در پوشه مقایسه کنید تا شباهت‌ها با استفاده از FAISS و LaBSE تشخیص داده شوند.")

    # Sidebar for configuration
    st.sidebar.header("تنظیمات")
    existing_docs_dir = st.sidebar.text_input("مسیر پوشه اسناد موجود", value="./Documents/")
    uploaded_file = st.sidebar.file_uploader("بارگذاری سند جدید (DOCX یا PDF)", type=["docx", "pdf"])

    if st.sidebar.button("بررسی شباهت"):
        if not os.path.isdir(existing_docs_dir):
            st.error("لطفاً یک مسیر معتبر برای پوشه اسناد موجود وارد کنید.")
        elif not uploaded_file:
            st.error("لطفاً یک سند جدید برای مقایسه بارگذاری کنید.")
        else:
            try:
                with st.spinner("در حال پردازش..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name
                    
                    # Run similarity check in a subprocess
                    result = subprocess.run(
                        [".venv/bin/python", "faiss_standalone.py", existing_docs_dir, "--new_doc_path", tmp_file_path],
                        capture_output=True,
                        text=True
                    )
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                    if result.returncode != 0:
                        st.error(f"خطا در پردازش: {result.stderr}")
                        return
                    
                    # Parse results (assuming standalone script prints JSON)
                    try:
                        results = json.loads(result.stdout)
                    except json.JSONDecodeError:
                        # Fallback: parse plain text output
                        lines = result.stdout.strip().split('\n')
                        results = []
                        for i in range(0, len(lines), 3):
                            file_name = lines[i].replace("Existing Document: ", "")
                            avg_sim = float(lines[i+1].replace("  Average Similarity: ", ""))
                            prop_high_sim = float(lines[i+2].replace("  Proportion of Highly Similar Sentences (threshold > 0.8): ", ""))
                            results.append((file_name, avg_sim, prop_high_sim))
                
                st.success("بررسی شباهت با موفقیت انجام شد!")
                st.header("نتایج (مرتب شده بر اساس میانگین شباهت)")
                
                for file_name, avg_sim, prop_high_sim in results:
                    with st.expander(f"سند موجود: {file_name}"):
                        st.write(f"**میانگین شباهت**: {avg_sim:.4f}")
                        st.write(f"**نسبت جملات با شباهت بالا (آستانه > 0.8)**: {prop_high_sim:.4f}")
            except Exception as e:
                st.error(f"خطایی رخ داده است: {str(e)}")

if __name__ == "__main__":
    main()