import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# LOAD TRAINED MODELS
# -------------------------------
cpu_model = joblib.load("cpu_model.pkl")
mem_model = joblib.load("mem_model.pkl")

st.set_page_config(page_title="HPC Resource Predictor", layout="centered")

st.title("HPC File-Based Resource Prediction System")
st.markdown("Upload a file to estimate CPU time and memory requirements")

# -------------------------------
# FILE TYPE ENCODING
# -------------------------------
file_type_map = {
    "csv": 1,
    "xlsx": 2,
    "py": 3,
    "sql": 4,
    "txt": 5,
    "json": 6
}

# -------------------------------
# FEATURE EXTRACTION FUNCTION
# -------------------------------
def extract_features(uploaded_file):
    # File size (KB)
    file_size_kb = len(uploaded_file.getvalue()) / 1024

    # File extension
    file_name = uploaded_file.name
    ext = file_name.split('.')[-1].lower()

    file_type = file_type_map.get(ext, 0)

    num_rows = 0
    num_columns = 0

    try:
        # Structured files
        if ext == "csv":
            df = pd.read_csv(uploaded_file)
            num_rows, num_columns = df.shape

        elif ext == "xlsx":
            df = pd.read_excel(uploaded_file)
            num_rows, num_columns = df.shape

        # Code / text files
        elif ext in ["txt", "py", "sql", "json"]:
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            num_rows = content.count("\n")

            # Rough complexity estimate
            num_columns = len(content) // max(1, num_rows)

    except:
        pass

    return np.array([[file_size_kb, file_type, num_rows, num_columns]]), ext, file_size_kb, num_rows, num_columns


# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload a file",
    type=["csv", "xlsx", "py", "sql", "txt", "json"]
)

# -------------------------------
# MAIN LOGIC
# -------------------------------
if uploaded_file is not None:

    features, ext, size, rows, cols = extract_features(uploaded_file)

    st.subheader("Extracted Features")
    st.write(f"File Type: {ext}")
    st.write(f"File Size: {size:.2f} KB")
    st.write(f"Rows/Lines: {rows}")
    st.write(f"Columns/Complexity: {cols}")

    if st.button("Predict Resource Requirements"):

        cpu_time = cpu_model.predict(features)[0]
        memory = mem_model.predict(features)[0]

        st.subheader("Prediction Results")

        st.success(f"Estimated CPU Time: {cpu_time:.2f} seconds")
        st.info(f"Estimated Memory Required: {memory:.2f} MB")

        # -------------------------------
        # HPC INTERPRETATION
        # -------------------------------
        st.subheader("HPC Recommendation")

        # Memory-based recommendation
        if memory < 1024:
            st.write("→ Suitable for Standard Node")
        elif memory < 4096:
            st.write("→ Use High-Memory Node")
        else:
            st.warning("→ Requires HPC Cluster Allocation")

        # CPU-based recommendation
        if cpu_time < 10:
            st.write("→ Short Job (Low Priority Queue)")
        elif cpu_time < 60:
            st.write("→ Medium Job")
        else:
            st.write("→ Long Job (Batch Scheduling Required)")