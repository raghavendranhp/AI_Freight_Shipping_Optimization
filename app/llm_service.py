import requests
import json
import os

#Configuration for local Ollama instance
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma:2b"

def load_prompt(filename):
    """
    Loads a prompt template from the 'prompts' directory.
    Uses relative paths so it works seamlessly inside the 'app' folder.
    """
    #Get the parent directory of 'app' (the project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, 'prompts', filename)
    
    try:
        with open(filepath, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Warning: {filename} not found in prompts directory.")
        return ""

def get_ai_insight(shipment_data, eta_hours, risk_label, confidence):
    """
    Combines shipment input data and ML predictions to prompt the local 
    Gemma model for actionable operational insights.
    """
    #Load prompts from text files
    system_prompt = load_prompt('system_prompt.txt')
    user_prompt_template = load_prompt('user_prompt.txt')

    #Fallback prompts just in case the files haven't been created yet
    if not system_prompt:
        system_prompt = "You are an expert AI logistics operations analyst. Provide concise, actionable, and highly professional advice."
    
    if not user_prompt_template:
        user_prompt_template = """
        Analyze this freight shipment:
        - Origin: {origin} -> Destination: {destination}
        - Mode: {mode} | Weather: {weather} | Traffic: {traffic}
        
        ML Pipeline Predictions:
        - Estimated Time of Arrival: {eta} hours
        - Delay Risk Category: {risk}
        - Model Confidence: {confidence}%
        
        Provide a 2-3 sentence insight for the dispatch team on how to handle this specific shipment.
        """

    #Inject the live API variables into the template
    formatted_user_prompt = user_prompt_template.format(
        origin=shipment_data.get('origin'),
        destination=shipment_data.get('destination'),
        mode=shipment_data.get('mode'),
        weather=shipment_data.get('weather'),
        traffic=shipment_data.get('traffic'),
        eta=round(eta_hours, 2),
        risk=risk_label,
        confidence=round(confidence * 100, 1)
    )

    #Build the Ollama API payload
    payload = {
        "model": MODEL_NAME,
        "system": system_prompt,
        "prompt": formatted_user_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3 # Low temperature for factual, analytical outputs
        }
    }

    #Make the request to local Gemma
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to local Ollama instance: {e}")
        return "AI Insight currently unavailable. Please ensure Ollama is running ('ollama run gemma:2b')."