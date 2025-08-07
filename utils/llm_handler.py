import os
from langchain_groq import ChatGroq
import traceback

def get_llm():
    """
    Returns an instance of the Groq LLM with the Llama3 model.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file. Please set it.")
        
    return ChatGroq(
        model_name="llama3-70b-8192",
        groq_api_key=groq_api_key,
        temperature=0.2
    )

def analyze_document_text(text, llm, response_mode="Detailed"):
    """
    Analyzes resume text to extract structured data, with error handling.
    """
    try:
        if response_mode == "Concise":
            style_instruction = "Be concise and brief in all text fields (e.g., summary, responsibilities)."
        else:
            style_instruction = "Provide detailed and comprehensive information in all text fields (e.g., summary, responsibilities)."

        prompt = f"""
        You are an expert HR and technical recruiter. Your task is to parse the following resume text. {style_instruction}
        A crucial part of your task is to calculate the "total_experience_years" by looking at the dates in the work experience section. For "Present" dates, assume the current date is August 2025.
        You MUST return the output as a clean JSON object. Do not add any text or markdown formatting before or after the JSON.

        Use this exact JSON format for your response:
        {{
          "personal_details": {{
            "name": "string",
            "email": "string",
            "phone": "string",
            "location": "string",
            "links": [
              {{ "site": "GitHub", "url": "string" }},
              {{ "site": "LinkedIn", "url": "string" }}
            ]
          }},
          "summary": "string",
          "total_experience_years": "string (e.g., '2.5 years', '5 years')",
          "work_experience": [
            {{
              "job_title": "string",
              "company": "string",
              "location": "string",
              "start_date": "string",
              "end_date": "string",
              "responsibilities": [
                "string",
                "string"
              ]
            }}
          ],
          "education": [
            {{
              "institution": "string",
              "degree": "string",
              "location": "string",
              "start_date": "string",
              "end_date": "string"
            }}
          ],
          "projects": [
            {{
              "name": "string",
              "technologies": "string",
              "description": "string"
            }}
          ],
          "technical_skills": {{
            "languages": ["string"],
            "libraries_and_technologies": ["string"]
          }},
          "certifications": ["string"]
        }}

        Resume Text to Parse:
        ---
        {text}
        ---
        """
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"--- ERROR in analyze_document_text ---")
        print(f"LLM call failed. Error: {e}")
        traceback.print_exc()
        return None # Return None on failure

def get_resume_improvement_suggestions(resume_text, llm, response_mode="Detailed"):
    """
    Analyzes resume text for improvements with error handling.
    """
    try:
        if response_mode == "Concise":
            style_instruction = "Provide a very brief, summarized list of the top 3 most important improvement points."
        else:
            style_instruction = """
            Provide a detailed, in-depth analysis. Give specific, actionable suggestions for each section of the resume (Summary, Work Experience, Projects, Skills). 
            Use bullet points for clarity and offer examples of improved phrasing where appropriate. Maintain a constructive and encouraging tone.
            """

        prompt = f"""
        You are an expert career coach and senior technical recruiter in India. Your task is to review the following resume text and provide actionable suggestions for improvement.
        {style_instruction}

        Resume Text to Analyze:
        ---
        {resume_text}
        ---
        """
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"--- ERROR in get_resume_improvement_suggestions ---")
        print(f"LLM call failed. Error: {e}")
        traceback.print_exc()
        return "Error: Could not get resume suggestions due to an API failure."
