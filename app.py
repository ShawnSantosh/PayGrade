import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import json
import os

# --- UTILITY IMPORTS ---
from utils.parser import extract_text_from_pdf
from utils.llm_handler import analyze_document_text
from utils.rag_handler import setup_vector_store, get_targeted_salary_estimation
from utils.api_handler import get_jooble_job_openings
from models.embeddings import get_embedding_model

# --- AGENT IMPORTS ---
from agents.agent_handler import get_agent_llm, create_agent_executor
from agents.tool_defs import get_tools
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

load_dotenv()

@st.cache_resource
def load_ai_resources():
    """Loads all AI models, tools, and the agent executor."""
    try:
        llm = get_agent_llm()
        embedding_model = get_embedding_model()
        vector_store = setup_vector_store(embedding_model)

        agent_executor = None
        if vector_store:
            tools = get_tools(vector_store)
            agent_executor = create_agent_executor(llm, tools)
        
        # --- FIX 1: Return the vector_store as well ---
        return llm, vector_store, agent_executor
    except Exception as e:
        st.error(f"A critical error occurred during AI resource initialization: {e}")
        return None, None, None

def main():
    st.set_page_config(page_title="PayGrade AI", page_icon="")
    st.title("PayGrade AI")

    # --- FIX 2: Receive the vector_store here ---
    llm, vector_store, agent_executor = load_ai_resources()

    st.sidebar.title("Configuration")
    st.sidebar.subheader("Response Style")
    response_mode = st.sidebar.radio(
        "Select response verbosity:", ("Detailed", "Concise"),
        key="response_mode", horizontal=True
    )
    st.sidebar.write("---")
    st.sidebar.title("Features")
    
    if not llm:
        st.error("Fatal Error: Could not initialize the Language Model. Please check your API keys.")
        st.stop()

    feature_list = ["Resume Analysis & Job Matching", "Compare Multiple Offers"]
    if agent_executor:
        st.sidebar.success("AI Agent is ONLINE", icon="✅")
        feature_list.append("AI Agent & Simulator")
    else:
        st.sidebar.warning("AI Agent is OFFLINE", icon="⚠️")
        st.sidebar.info("The Agent is offline. Please ensure the `documents` folder exists and contains at least one PDF.")
        
    app_mode = st.sidebar.selectbox("Select a Feature", feature_list)

    if app_mode == "Resume Analysis & Job Matching":
        st.header("Analyze Resume, Estimate Salary & Find Jobs")
        
        uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

        if uploaded_file:
            if 'last_file' not in st.session_state or st.session_state.last_file != uploaded_file.name:
                st.session_state.update({
                    'analysis': None, 'salary_estimation': None, 'job_openings': None,
                    'last_file': uploaded_file.name
                })

            text = extract_text_from_pdf(uploaded_file)
            
            if text and "Error" not in text:
                with st.spinner("Analyzing resume profile..."):
                    analysis_result = analyze_document_text(text, llm, response_mode=st.session_state.response_mode)
                
                if analysis_result:
                    try:
                        result_json = json.loads(analysis_result.strip())
                        st.session_state.analysis = result_json
                    except (json.JSONDecodeError, TypeError):
                        st.session_state.analysis = None

        if st.session_state.get('analysis'):
            st.success("Resume analyzed successfully!")
            with st.expander("View Full Resume Analysis (JSON)"):
                st.json(st.session_state['analysis'])

            st.subheader("Enter the details of the job you are applying for:")
            target_role_input = st.text_input("Target Job Title", placeholder="e.g., Senior Data Scientist")
            target_location_input = st.text_input("Target Location", value="Bengaluru, India")

            col1, col2 = st.columns(2)
            with col1:
                # The error was here because 'vector_store' was not defined in this scope. It is now.
                if st.button("Estimate Salary & Analyze Fit"):
                    if target_role_input and vector_store:
                        with st.spinner("Comparing your profile..."):
                            st.session_state.salary_estimation = get_targeted_salary_estimation(
                                st.session_state.analysis, target_role_input, target_location_input,
                                vector_store, llm, response_mode=st.session_state.response_mode
                            )
            with col2:
                if st.button("Find Jobs for This Target Role"):
                    if target_role_input:
                        with st.spinner(f"Searching for '{target_role_input}' jobs..."):
                            st.session_state.job_openings = get_jooble_job_openings(target_role_input, target_location_input)

            if st.session_state.get('salary_estimation'):
                st.subheader("✅ Salary & Profile Fit Analysis:")
                st.info(st.session_state.salary_estimation)
            
            if st.session_state.get('job_openings'):
                st.subheader("🔍 Live Job Openings:")
                jobs = st.session_state.job_openings
                if isinstance(jobs, list) and jobs:
                    for job in jobs:
                        with st.expander(f"{job.get('title')} at {job.get('company', 'N/A')}"):
                            st.markdown(f"**Location:** {job.get('location', 'N/A')}")
                            st.markdown(f"**Description:** {job.get('snippet', '').strip()}")
                            st.markdown(f"**[View Full Job Posting]({job.get('link')})**")
                elif isinstance(jobs, dict) and 'error' in jobs:
                    st.error(f"Could not fetch jobs: {jobs['error']}")
                else:
                    st.write("No matching job openings found for this search.")

    elif app_mode == "Compare Multiple Offers":
        st.header("📂 Compare Multiple Offers")
        uploaded_files = st.file_uploader("Upload 2 or more offer letters (PDFs)", type="pdf", accept_multiple_files=True)
        
        if uploaded_files and len(uploaded_files) > 1:
            all_analyses = []
            with st.spinner(f"Analyzing {len(uploaded_files)} offers..."):
                for file in uploaded_files:
                    text = extract_text_from_pdf(file)
                    if text and "Error" not in text:
                        analysis_str = analyze_document_text(text, llm)
                        try:
                            analysis_json = json.loads(analysis_str.strip())
                            analysis_json['File Name'] = file.name
                            all_analyses.append(analysis_json)
                        except (json.JSONDecodeError, TypeError):
                            st.warning(f"Could not parse analysis for {file.name}.")
            
            if all_analyses:
                st.subheader("Offer Comparison Table")
                df = pd.DataFrame(all_analyses).set_index('File Name')
                st.dataframe(df)

    elif app_mode == "AI Agent & Simulator":
        st.header("🤖 Chat With Kariar")
        history = StreamlitChatMessageHistory(key="agent_chat_history")
        
        if not history.messages:
            st.info("Ask me anything! I can search your documents or the live web for answers.")

        for msg in history.messages:
            st.chat_message(msg.type).write(msg.content)

        if prompt := st.chat_input("Ask a question..."):
            st.chat_message("human").write(prompt)
            with st.spinner("Thinking..."):
                if st.session_state.response_mode == "Concise":
                    prompt_for_agent = f"{prompt} (Please keep your answer concise)"
                else:
                    prompt_for_agent = f"{prompt} (Please provide a detailed answer)"
                
                response = agent_executor.invoke({
                    "input": prompt_for_agent,
                    "chat_history": history.messages
                })
            st.chat_message("ai").write(response["output"])
            history.add_user_message(prompt)
            history.add_ai_message(response["output"])

if __name__ == "__main__":
    keys_to_init = ['analysis', 'salary_estimation', 'job_openings', 'last_file', 'agent_chat_history']
    for key in keys_to_init:
        if key not in st.session_state:
            st.session_state[key] = [] if 'history' in key else None
    main()