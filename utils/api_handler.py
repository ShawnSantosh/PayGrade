import os
import requests
import certifi

def get_market_salary_data(job_title, location):
    """
    Searches for market salary data using the JSearch API on RapidAPI.
    """
    api_key = os.getenv("JSEARCH_API_KEY")
    if not api_key:
        return "Error: JSEARCH_API_KEY not found in .env file."

    url = "https://jsearch.p.rapidapi.com/search"
    querystring = {"query": f"salary for {job_title} in {location}", "num_pages": "1"}
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    try:
        response = requests.get(url, headers=headers, params=querystring, verify=certifi.where())
        response.raise_for_status()
        results = response.json()
        jobs = results.get('data', [])
        
        if not jobs:
            return "No job listings found on JSearch for this role to determine salary."

        salary_info = []
        for job in jobs:
            min_salary = job.get('job_min_salary')
            max_salary = job.get('job_max_salary')
            salary_period = job.get('job_salary_period')
            employer = job.get('employer_name', 'an employer')

            if min_salary and max_salary and salary_period:
                salary_text = f"₹{min_salary:,.0f} - ₹{max_salary:,.0f} per {salary_period.lower()}"
                salary_info.append(f"- {employer}: {salary_text}")

        if salary_info:
            return "Found the following salary ranges in recent job listings:\n" + "\n".join(salary_info[:5])
        else:
            return "Found job listings, but could not extract specific salary range data."

    except requests.exceptions.HTTPError as http_err:
        return f"An HTTP error occurred: {http_err} - {response.text}"
    except Exception as e:
        return f"An error occurred: {e}"

def get_jooble_job_openings(job_title, location):
    """
    Fetches similar job openings from the Jooble API, using certifi for SSL verification.
    """
    api_key = os.getenv("JOOBLE_API_KEY")
    if not api_key:
        return {"error": "JOOBLE_API_KEY not found in .env file."}

    url = f"https://jooble.org/api/{api_key}"
    headers = {"Content-Type": "application/json"}
    body = {
       "keywords": job_title,
       "location": location
    }

    try:
        response = requests.post(url, headers=headers, json=body, verify=certifi.where())
        response.raise_for_status()
        results = response.json()
        return results.get('jobs', [])
    except requests.exceptions.HTTPError as http_err:
        return {"error": f"An HTTP error occurred: {http_err} - {response.text}"}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}
