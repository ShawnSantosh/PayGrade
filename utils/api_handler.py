import os
import requests
import certifi

# The get_market_salary_data function remains the same
def get_market_salary_data(job_title, location):
    """
    Searches for market salary data using the JSearch API on RapidAPI.
    """
    # ... (This function's code is unchanged) ...


# --- THIS IS THE UPDATED FUNCTION ---
def get_jooble_job_openings(job_title, location):
    """
    Fetches and fine-tunes similar job openings from the Jooble API.
    """
    api_key = os.getenv("JOOBLE_API_KEY")
    if not api_key:
        return {"error": "JOOBLE_API_KEY not found in .env file."}

    url = f"https://jooble.org/api/{api_key}"
    headers = {"Content-Type": "application/json"}
    
    # 1. Refine the search query by putting the job title in quotes
    # This asks the API to prioritize an exact phrase match.
    body = {
       "keywords": f'"{job_title}"',
       "location": location
    }

    try:
        response = requests.post(url, headers=headers, json=body, verify=certifi.where())
        response.raise_for_status()
        
        results = response.json()
        jobs = results.get('jobs', [])
        
        if not jobs:
            return []

        # 2. Post-filter the results to ensure they are highly relevant
        filtered_jobs = []
        # Create a list of essential keywords from the user's input (e.g., ['senior', 'data', 'scientist'])
        title_keywords = job_title.lower().split()

        for job in jobs:
            api_job_title = job.get('title', '').lower()
            
            # Check if all the user's keywords are present in the job title from the API
            if all(keyword in api_job_title for keyword in title_keywords):
                filtered_jobs.append(job)
                
        return filtered_jobs

    except requests.exceptions.HTTPError as http_err:
        return {"error": f"An HTTP error occurred: {http_err} - {response.text}"}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}
