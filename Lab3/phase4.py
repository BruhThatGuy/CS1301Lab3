import streamlit as st
import requests
import json
import time
from datetime import datetime
import functools

# --- Configuration & State ---
# NOTE: Replace with your actual Gemini API key, or load from environment variable
API_KEY = "" 
MODEL_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
MAX_RETRIES = 5
DEFAULT_MODEL = "gemini-2.5-flash-preview-09-2025"

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "model", "text": "Hello! I am a general-purpose assistant. You can ask me anything, or try asking for the **current weather or a 7-day forecast** for any city!"}]

# --- Tool Definitions for LLM Function Calling ---

def geocode_city(city):
    """Fetches latitude and longitude for a city."""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    response = requests.get(url)
    data = response.json()
    
    if "results" not in data or len(data["results"]) == 0:
        return None, None
    
    # Select the most populated city match for better accuracy
    best_match = max(data["results"], key=lambda c: c.get("population", 0))
    return best_match["latitude"], best_match["longitude"]

def get_current_and_forecast_weather(city: str, units: str = 'celsius') -> str:
    """
    Provides the current temperature and a summary of the 7-day weather forecast 
    for a given city. Use this tool only when the user explicitly asks for the 
    current or future weather or forecast for a specific location.
    The 'units' parameter must be 'celsius' or 'fahrenheit'.
    """
    if units not in ['celsius', 'fahrenheit']:
        return json.dumps({"error": "Invalid units. Must be 'celsius' or 'fahrenheit'."})

    lat, lon = geocode_city(city)
    if lat is None:
        return json.dumps({"error": f"Could not find coordinates for city: {city}"})

    # Construct the forecast API URL
    temp_unit = "temperature_unit=fahrenheit" if units == "fahrenheit" else ""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,weather_code&current=temperature_2m,weather_code,wind_speed_10m&timezone=auto&{temp_unit}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Extract Current Data
        current = data.get('current', {})
        current_temp = current.get('temperature_2m')
        current_code = current.get('weather_code')
        wind_speed = current.get('wind_speed_10m')
        
        unit_symbol = "°F" if units == "fahrenheit" else "°C"

        # Extract Forecast Data (first 7 days)
        daily = data.get('daily', {})
        forecasts = []
        for i in range(min(7, len(daily.get('time', [])))):
            forecasts.append({
                "date": daily['time'][i],
                "max_temp": daily['temperature_2m_max'][i],
                "min_temp": daily['temperature_2m_min'][i],
            })

        # Return structured data for the LLM to synthesize
        return json.dumps({
            "city": city,
            "units": unit_symbol,
            "current_conditions": {
                "temperature": current_temp,
                "wind_speed": wind_speed,
                "note": "Weather code is not provided for simplification."
            },
            "seven_day_forecast": forecasts
        })

    except Exception as e:
        return json.dumps({"error": f"Error fetching weather data: {e}"})

# Define the callable tool for the LLM to use
AVAILABLE_TOOLS = {
    "get_current_and_forecast_weather": get_current_and_forecast_weather,
}

# --- LLM Communication Logic ---

def extract_sources(attributions):
    """Converts the raw grounding attribution array into a simplified array of objects."""
    if not attributions:
        return []
    sources = []
    for attr in attributions:
        if 'web' in attr and 'uri' in attr['web'] and 'title' in attr['web']:
            sources.append({'uri': attr['web']['uri'], 'title': attr['web']['title']})
    return sources

def fetch_with_exponential_backoff(payload, url, api_key, max_retries=MAX_RETRIES):
    """Implements exponential backoff for API retries using requests."""
    if not api_key:
        raise ValueError("Gemini API_KEY is missing.")

    full_url = f"{url}?key={api_key}"
    headers = {'Content-Type': 'application/json'}

    for i in range(max_retries):
        try:
            response = requests.post(full_url, headers=headers, data=json.dumps(payload), timeout=60)
            
            if response.ok:
                return response.json()

            if response.status_code == 429 or response.status_code >= 500:
                print(f"Attempt {i + 1} failed with status {response.status_code}. Retrying...")
                if i < max_retries - 1:
                    delay = (2 ** i) + (time.time() % 1)
                    time.sleep(delay)
                continue
            
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            print(f"Request attempt {i + 1} failed: {e}. Retrying...")
            if i < max_retries - 1:
                delay = (2 ** i) + (time.time() % 1)
                time.sleep(delay)
            else:
                raise

    raise requests.exceptions.RequestException("API call failed after maximum retries.")

def call_gemini_api(messages):
    """Handles the main chat loop with tool calling logic."""
    
    # 1. Define the tools for the LLM
    # The search tool is used for general grounding/knowledge.
    tools_config = [{"google_search": {}}] 
    
    # The weather tool is defined via its function structure
    weather_tool_definition = {
        "functionDeclarations": [
            {
                "name": "get_current_and_forecast_weather",
                "description": get_current_and_forecast_weather.__doc__.strip(),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "city": {"type": "STRING", "description": "The name of the city for the weather query (e.g., 'Paris')."},
                        "units": {"type": "STRING", "description": "The temperature unit, either 'celsius' or 'fahrenheit'. Defaults to 'celsius' if not specified."}
                    },
                    "required": ["city"],
                }
            }
        ]
    }
    
    # Combine search and the function tool
    all_tools = [tools_config[0], {"functionDeclarations": weather_tool_definition["functionDeclarations"]}]

    # 2. Construct the payload
    # Convert Streamlit messages history to the Gemini API format
    contents = []
    for message in messages:
        if message["role"] == "user":
            contents.append({"role": "user", "parts": [{"text": message["text"]}]})
        elif message["role"] == "model":
            # This logic might need refinement for complex tool calls in history
            contents.append({"role": "model", "parts": [{"text": message["text"]}]})
        elif message["role"] == "function_response":
            contents.append({"role": "function", "parts": [{"functionResponse": message["functionResponse"]}]})
        elif message["role"] == "tool_call":
            contents.append({"role": "model", "parts": [{"functionCall": message["functionCall"]}]})

    payload = {
        "contents": contents,
        "config": {
            "tools": all_tools
        }
    }

    # 3. Initial API Call (LLM decides text, search, or tool)
    with st.spinner("Thinking..."):
        try:
            result = fetch_with_exponential_backoff(payload, MODEL_URL, API_KEY)
        except Exception as e:
            st.error(f"Error during API call: {e}")
            return

    candidate = result.get('candidates', [{}])[0]
    
    # 4. Check for Function Call
    if 'functionCall' in candidate.get('content', {}).get('parts', [{}])[0]:
        
        function_call = candidate['content']['parts'][0]['functionCall']
        func_name = function_call['name']
        func_args = dict(function_call['args'])
        
        st.session_state["messages"].append({
            "role": "tool_call",
            "functionCall": function_call
        })
        
        st.info(f"LLM decided to call tool: {func_name}({func_args})")
        
        # Execute the function locally
        if func_name in AVAILABLE_TOOLS:
            
            with st.spinner(f"Executing {func_name}..."):
                func_to_call = AVAILABLE_TOOLS[func_name]
                function_output = func_to_call(**func_args)
            
            # Add function response to history
            st.session_state["messages"].append({
                "role": "function_response",
                "functionResponse": {
                    "name": func_name,
                    "response": {"content": function_output}
                }
            })
            
            # Recursive call with function result to get final LLM response
            with st.spinner("Synthesizing tool output..."):
                final_response = call_gemini_api(st.session_state["messages"])
                return final_response
        else:
            return "Error: Unknown tool requested by model."

    # 5. Handle Text Response (with or without grounding)
    text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'No response generated.')
    sources = extract_sources(candidate.get('groundingMetadata', {}).get('groundingAttributions'))

    response_data = {"text": text, "sources": sources}
    st.session_state["messages"].append({"role": "model", "text": response_data})
    return response_data

# --- Streamlit Application Layout ---

def main():
    st.set_page_config(page_title="Gemini Chatbot with Weather Tool", layout="centered")
    st.title("🤖 General Chatbot with Open-Meteo Tool")
    
    # Check for API Key
    if not API_KEY:
        st.warning("⚠️ Please enter your Gemini API Key in the Python file (`API_KEY = ...`) to enable the chatbot.")
        return

    # Display chat history
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["text"])
        elif message["role"] == "model":
            with st.chat_message("assistant"):
                st.markdown(message["text"]["text"] if isinstance(message["text"], dict) else message["text"])
                if isinstance(message["text"], dict) and message["text"]["sources"]:
                    with st.expander("Grounded Sources (via Google Search)"):
                        for source in message["text"]["sources"]:
                            st.markdown(f"[{source['title']}]({source['uri']})")
        # Tool call messages are displayed as info banners
        elif message["role"] == "tool_call":
            with st.chat_message("assistant"):
                st.info(f"Tool Call: {message['functionCall']['name']}({dict(message['functionCall']['args'])})")
        elif message["role"] == "function_response":
            with st.chat_message("assistant"):
                st.code(message['functionResponse']['response']['content'], language='json')
                
    # Handle user input
    if prompt := st.chat_input("Ask a general question, or a weather query for a city..."):
        
        # Add user message to history
        st.session_state["messages"].append({"role": "user", "text": prompt})
        
        # Display the user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call the API and get the response
        response_data = call_gemini_api(st.session_state["messages"])
        
        # Rerun to display the model's response and any tool execution steps
        st.rerun()

if __name__ == "__main__":
    main()
