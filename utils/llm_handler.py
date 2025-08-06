import os
from langchain_groq import ChatGroq

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
    Analyzes resume text to extract structured data, adjusting for verbosity.
    """
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