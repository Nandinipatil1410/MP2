import streamlit as st
import os
import tempfile
from modules.orchestrator import HybridOrchestrator

# Page Config
st.set_page_config(page_title="Secure Hybrid LLM", layout="wide")

st.title("🛡️ Privacy-Preserving Hybrid LLM System")
st.markdown("### Secure Document Reasoning with Local Execution & Cloud Planning")

# Initialize Orchestrator
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = HybridOrchestrator()

# Sidebar for settings and document upload
with st.sidebar:
    st.header("Settings & Upload")
    uploaded_files = st.file_uploader("Upload sensitive PDF documents", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Indexing documents locally..."):
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    st.session_state.orchestrator.vector_store.process_pdf(tmp_path)
                    os.unlink(tmp_path)
                st.success(f"Processed {len(uploaded_files)} documents.")
        else:
            st.warning("Please upload at least one PDF.")

    st.divider()
    st.info("Cloud Provider: Groq (Reasoning Only)\nLocal Provider: Ollama (Execution)")

# Main Query Interface
query = st.text_input("Enter your query about the documents:", placeholder="e.g., Analyze the potential risks in the security logs...")

if query:
    tab1, tab2 = st.tabs(["🔒 Hybrid Reasoning (Secure)", "🏠 Local-Only Baseline"])
    
    with tab1:
        with st.spinner("Generating secure reasoning plan in cloud..."):
            result = st.session_state.orchestrator.process_query_hybrid(query)
            
        st.subheader("Final Answer")
        st.write(result["final_answer"])
        
        with st.expander("Show Privacy & Reasoning Details"):
            st.markdown("**Masked Query (Sent to Cloud):**")
            st.code(result["masked_query"])
            
            st.markdown("**Cloud-Generated Reasoning Plan:**")
            st.info(result["reasoning_plan"])
            
            st.markdown("**Sources used:**")
            st.write(list(set(result["context_sources"])))
            
    with tab2:
        with st.spinner("Processing locally..."):
            local_result = st.session_state.orchestrator.process_query_local_only(query)
        st.subheader("Local Model Output")
        st.write(local_result)
