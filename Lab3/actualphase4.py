import streamlit as st
import requests
import json
import time
from datetime import datetime
import functools

# --- Configuration & State ---
# NOTE: Replace with your actual Gemini API key, or load from environment variable
# If running in a secure environment, consider using st.secrets or environment variables.
API_KEY = "AIzaSyBg7BL-ACkEFFkSHjTxXk_trTOJu1vON5I" 
MODEL_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
DEFAULT_MODEL = "gemini-2.5-flash-preview-09-2025"

if "messages" not in st.session_state:
    # Initial message text is stored as a simple string for simplicity
    st.session_state["messages"] = [{"role": "model", "text": "Hello! I am a general-purpose assistant. You can ask me anything, or try asking for the **current weather or a 7-day forecast** for any city!"}]

# --- Tool Definitions for LLM Function Calling ---

@st.cache_data(ttl=3600) # Cache the geocoding results for 1 hour
def geocode_city(city):
    """Fetches latitude and longitude for a city."""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() 
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Geocoding API Request Error for {city}: {e}")
        return None, None
    except json.JSONDecodeError as e:
        print(f"Geocoding API JSON Decode Error for {city}: {e}")
        return None, None
    
    if "results" not in data or len(data["results"]) == 0:
        return None, None
    
    # Select the city with the highest population for the best match
    best_match = None
    max_population = -1 

    for city_data in data["results"]:
        current_population = city_data.get("population", 0)
        if current_population > max_population:
            max_population = current_population
            best_match = city_data
            
    if best_match is None:
        return None, None
        
    return best_match.get("latitude"), best_match.get("longitude")

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
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract Current Data
        current = data.get('current', {})
        
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
                "temperature": current.get('temperature_2m'),
                "wind_speed": current.get('wind_speed_10m'),
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

def call_gemini_api(messages):
    """Handles the main chat loop with tool calling logic and exponential backoff."""
    
    if not API_KEY:
        st.error("Gemini API_KEY is missing.")
        return

    full_url = f"{MODEL_URL}?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}
    
    # --- 1. Define the tools for the LLM ---
    tools_config = [{"google_search": {}}] 
    
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
    
    all_tools = [tools_config[0], {"functionDeclarations": weather_tool_definition["functionDeclarations"]}]

    # --- 2. Construct the payload ---
    # Convert Streamlit messages history to the Gemini API format (role mapping)
    contents = []
    for message in messages:
        if message["role"] == "user":
            contents.append({"role": "user", "parts": [{"text": message["text"]}]})
        elif message["role"] == "model":
            # Check if the message contains a tool call instead of plain text
            if "functionCall" in message:
                contents.append({"role": "model", "parts": [{"functionCall": message["functionCall"]}]})
            else:
                contents.append({"role": "model", "parts": [{"text": message["text"]}]})
        elif message["role"] == "function":
             # This handles the function response back to the model
            contents.append({"role": "function", "parts": [{"functionResponse": message["functionResponse"]}]})


    payload = {
        "contents": contents,
        "config": {
            "tools": all_tools
        }
    }

    # --- 3. Initial API Call (with simple retry/backoff) ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with st.spinner(f"Thinking... (Attempt {attempt + 1}/{max_retries})"):
                # Use a small backoff delay
                if attempt > 0:
                    time.sleep(2 ** attempt)

                response = requests.post(full_url, headers=headers, data=json.dumps(payload), timeout=90)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                result = response.json()
                
                # Success
                break 

        except requests.exceptions.HTTPError as e:
            if response.status_code in [429, 500, 503] and attempt < max_retries - 1:
                st.warning(f"Rate limit or server error ({response.status_code}). Retrying...")
                continue
            else:
                st.error(f"Error during API call: {response.status_code} - {e}")
                return {"text": f"Error: Could not complete API request. Status code {response.status_code}."}
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                st.warning(f"Network error. Retrying... ({e})")
                continue
            else:
                st.error(f"Fatal Network Error: {e}")
                return {"text": "Error: Could not connect to the Gemini API."}
        except json.JSONDecodeError as e:
            st.error(f"Error decoding API response: {e}")
            return {"text": "Error: Could not decode response from Gemini API."}
    else:
        # This branch runs if the loop completes without 'break' (all attempts failed)
        return {"text": "Error: Failed to get a response from Gemini after multiple attempts."}


    candidate = result.get('candidates', [{}])[0]
    
    # --- 4. Check for Function Call ---
    # The response structure for function call is nested deeply
    content_parts = candidate.get('content', {}).get('parts', [{}])
    
    if content_parts and 'functionCall' in content_parts[0]:
        
        function_call = content_parts[0]['functionCall']
        func_name = function_call['name']
        func_args = dict(function_call['args'])
        
        # Add the model's intent (function call) to the history
        st.session_state["messages"].append({
            "role": "model",
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
                "role": "function",
                "functionResponse": {
                    "name": func_name,
                    "response": {"content": function_output}
                }
            })
            
            # Recursive call with function result to get final LLM response
            # The recursive call uses the updated st.session_state["messages"]
            final_response = call_gemini_api(st.session_state["messages"]) 
            return final_response
        else:
            return {"text": "Error: Unknown tool requested by model."}

    # --- 5. Handle Text Response (with or without grounding) ---
    text = content_parts[0].get('text', 'No response generated.')
    sources = extract_sources(candidate.get('groundingMetadata', {}).get('groundingAttributions'))

    response_data = {"text": text, "sources": sources}
    # Do NOT add to history here, it is added in the main loop to ensure correct order
    return response_data

# --- Streamlit Application Layout ---

def main():
    st.set_page_config(page_title="Gemini Weather Chatbot", layout="centered")
    st.title("🤖 Gemini Chatbot with Weather Tool")
    
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
                # Check if the model message is a plain text response or a tool call
                if "functionCall" in message:
                    st.info(f"Tool Call: {message['functionCall']['name']}({dict(message['functionCall']['args'])})")
                
                # FIXED: Check if 'text' is a dictionary before trying to access 'sources'
                elif isinstance(message["text"], dict):
                    # This branch handles responses from call_gemini_api that include sources
                    st.markdown(message["text"]["text"])
                    # Display sources if available
                    if message["text"]["sources"]:
                        with st.expander("Grounded Sources (via Google Search)"):
                            for source in message["text"]["sources"]:
                                st.markdown(f"[{source['title']}]({source['uri']})")
                else:
                    # This branch handles the initial string greeting message and other simple strings
                    st.markdown(message["text"])
        
        # Function response (tool output) is displayed as code block
        elif message["role"] == "function":
            with st.chat_message("assistant"):
                st.code(message['functionResponse']['response']['content'], language='json')
                
    # Handle user input
    if prompt := st.chat_input("Ask a general question, or a weather query for a city..."):
        
        # 1. Add user message to history
        st.session_state["messages"].append({"role": "user", "text": prompt})
        
        # 2. Display the user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # 3. Call the API and handle the response/tool-calling loop
        response_data = call_gemini_api(st.session_state["messages"])
        
        # 4. Only add the final text response to the history here.
        if response_data and "text" in response_data:
            st.session_state["messages"].append({"role": "model", "text": response_data})
        
        # 5. Rerun to display the model's response and any tool execution steps
        st.rerun()

if __name__ == "__main__":
    main()
