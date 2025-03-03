import os
import numpy as np
import streamlit as st
import subprocess
import json
import tempfile

# Streamlit GUI
def main():
    st.title("Document Similarity Checker")
    st.markdown("Upload a new document and compare it against a directory of existing documents to detect similarities using FAISS and LaBSE embeddings.")

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    existing_docs_dir = st.sidebar.text_input("Existing Documents Directory", value="./Documents/")
    uploaded_file = st.sidebar.file_uploader("Upload New Document (DOCX or PDF)", type=["docx", "pdf"])

    if st.sidebar.button("Check Similarity"):
        if not os.path.isdir(existing_docs_dir):
            st.error("Please provide a valid directory containing existing documents.")
        elif not uploaded_file:
            st.error("Please upload a new document to compare.")
        else:
            try:
                with st.spinner("Processing..."):
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
                        st.error(f"Error during processing: {result.stderr}")
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
                
                st.success("Similarity check completed!")
                st.header("Results (Sorted by Average Similarity)")
                
                for file_name, avg_sim, prop_high_sim in results:
                    with st.expander(f"Existing Document: {file_name}"):
                        st.write(f"**Average Similarity**: {avg_sim:.4f}")
                        st.write(f"**Proportion of Highly Similar Sentences (threshold > 0.8)**: {prop_high_sim:.4f}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
