import os
import re
import requests

def parse_repo_name(repo_name: str) -> str:
    """Helper to parse owner/repo from either URL, raw string, or key-value labels."""
    repo_name = repo_name.strip()
    
    # 1. Handle key-value format (e.g. owner=SachinMishra-ux, repo=Financial_Pdfs)
    if "owner=" in repo_name or "repo=" in repo_name:
        owner_match = re.search(r"owner\s*=\s*([^,\s]+)", repo_name, re.IGNORECASE)
        repo_match = re.search(r"repo\s*=\s*([^,\s]+)", repo_name, re.IGNORECASE)
        if owner_match and repo_match:
            return f"{owner_match.group(1)}/{repo_match.group(1)}"
        # Fallback raw extraction
        clean = repo_name.replace("owner=", "").replace("repo=", "").replace(" ", "").replace(",", "")
        return clean

    # 2. Handle full GitHub URLs (e.g. https://github.com/google/jax)
    if "github.com/" in repo_name:
        parts = repo_name.split("github.com/")[-1].split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
            
    # 3. Default raw owner/repo shortcode
    return repo_name


def get_github_repo_info(repo_name: str) -> str:
    """Fetches key repository details like stars, description, fork count, and open issues/PRs. The input 'repo_name' MUST be a simple string in the format 'owner/repo' (e.g. 'google/jax') or a full GitHub URL (e.g. 'https://github.com/google/jax'). Do NOT format as 'owner=X, repo=Y'."""
    owner_repo = parse_repo_name(repo_name)
    url = f"https://api.github.com/repos/{owner_repo}"
    headers = {"User-Agent": "PurePythonAgent"}
    
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
        
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return (
                f"Repository: {data.get('full_name')}\n"
                f"Description: {data.get('description', 'No description available')}\n"
                f"Stars: {data.get('stargazers_count')}\n"
                f"Forks: {data.get('forks_count')}\n"
                f"Open Issues & PRs: {data.get('open_issues_count')}\n"
                f"License: {data.get('license', {}).get('name') if data.get('license') else 'No License'}\n"
                f"HTML URL: {data.get('html_url')}"
            )
        elif response.status_code == 404:
            return f"Error: Repository '{owner_repo}' not found."
        elif response.status_code == 403 and "rate limit" in response.text.lower():
            return "Error: GitHub API rate limit exceeded. Please configure a GITHUB_TOKEN environment variable."
        else:
            return f"Error: GitHub API returned status code {response.status_code}."
    except Exception as e:
        return f"Error connecting to GitHub API: {str(e)}"


def get_github_repo_prs(repo_name: str) -> str:
    """Fetches the titles, authors, and URLs of the 5 most recent open Pull Requests (PRs) in a repository. The input 'repo_name' MUST be a simple string in the format 'owner/repo' (e.g. 'google/jax') or a full GitHub URL (e.g. 'https://github.com/google/jax'). Do NOT format as 'owner=X, repo=Y'."""
    owner_repo = parse_repo_name(repo_name)
    url = f"https://api.github.com/repos/{owner_repo}/pulls?state=open&per_page=5"
    headers = {"User-Agent": "PurePythonAgent"}
    
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
        
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            prs = response.json()
            if not prs:
                return f"No open Pull Requests found for '{owner_repo}'."
            
            result = f"Recent Open Pull Requests for '{owner_repo}':\n"
            for pr in prs:
                result += f"- #{pr.get('number')}: '{pr.get('title')}' by @{pr.get('user', {}).get('login')} (URL: {pr.get('html_url')})\n"
            return result
        elif response.status_code == 404:
            return f"Error: Repository '{owner_repo}' not found."
        else:
            return f"Error: GitHub API returned status code {response.status_code}."
    except Exception as e:
        return f"Error connecting to GitHub API: {str(e)}"


def get_weather(city_name: str) -> str:
    """Fetches the current live weather conditions for a city without requiring an API key."""
    city_name = city_name.strip()
    url = f"https://wttr.in/{city_name}?format=j1"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            current = data['current_condition'][0]
            temp_c = current.get('temp_C')
            temp_f = current.get('temp_F')
            desc = current.get('weatherDesc')[0].get('value')
            wind = current.get('windspeedKmph')
            humidity = current.get('humidity')
            
            area = data.get('nearest_area', [{}])[0]
            region = area.get('region', [{}])[0].get('value')
            country = area.get('country', [{}])[0].get('value')
            resolved_loc = f"{city_name} ({region}, {country})" if region and country else city_name
            
            return (
                f"Location: {resolved_loc}\n"
                f"Condition: {desc}\n"
                f"Temperature: {temp_c}°C ({temp_f}°F)\n"
                f"Wind Speed: {wind} km/h\n"
                f"Humidity: {humidity}%"
            )
        else:
            return f"Error: Weather service wttr.in returned status code {response.status_code}."
    except Exception as e:
        return f"Error fetching weather from wttr.in: {str(e)}"
