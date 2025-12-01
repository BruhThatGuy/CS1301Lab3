import streamlit as st
import datetime
import requests
import google.generativeai as genai

st.set_page_config(page_title="Weather Chat Assistant", page_icon="🌤️")
st.title("🌤️ Weather Chat Assistant")
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except:
    st.error("⚠️ Please add GEMINI_API_KEY to your secrets!")
    st.stop()
model = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
user_input = st.chat_input("Ask me about weather in any city...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    weather_words = ['weather', 'temperature', 'temp', 'hot', 'cold', 'warm', 'climate']
    is_weather_question = any(word in user_input.lower() for word in weather_words)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if is_weather_question:
                try:
                    extract_prompt = f'''From this question, extract:
1. The city name
2. When they're asking about (e.g., "tomorrow", "next week", "January 15", "in 30 days")

Question: "{user_input}"

Respond in this exact format:
City: [city name]
When: [time reference]'''
                    
                    extraction = model.generate_content(extract_prompt)
                    extracted = extraction.text.strip()
                    city = ""
                    when = ""
                    for line in extracted.split('\n'):
                        if line.startswith('City:'):
                            city = line.replace('City:', '').strip().replace('"', '').replace("'", "")
                        elif line.startswith('When:'):
                            when = line.replace('When:', '').strip()
                    today = datetime.date.today()
                    date_prompt = f'''Today is {today}. The user asked about: "{when}"
                    
How many days from today is this? Just give me a number.
If it's "tomorrow", say 1.
If it's "next week", say 7.
If it's a specific date, calculate the difference.
If you're not sure or they didn't specify, say 1.

Just respond with a single number, nothing else.'''
                    date_response = model.generate_content(date_prompt)
                    days_ahead = int(date_response.text.strip())
                    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
                    geo_data = requests.get(geo_url, timeout=10).json()
                    if "results" not in geo_data or len(geo_data["results"]) == 0:
                        answer = f"Couldn't find a city called '{city}'."
                    else:
                        lat = geo_data["results"][0]["latitude"]
                        lon = geo_data["results"][0]["longitude"]
                        city_name = geo_data["results"][0]["name"]
                        
                        if days_ahead <= 7:
                            week_ago = today - datetime.timedelta(days=7)
                            weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={week_ago}&end_date={today}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean&temperature_unit=fahrenheit"
                            weather_data = requests.get(weather_url, timeout=10).json()
                            if "daily" in weather_data:
                                temps = weather_data["daily"]["temperature_2m_mean"]
                                dates = weather_data["daily"]["time"]
                                avg_temp = sum(temps) / len(temps)
                                max_temp = max(weather_data["daily"]["temperature_2m_max"])
                                min_temp = min(weather_data["daily"]["temperature_2m_min"])
                                recent_temps = temps[-3:]
                                predicted_temp = sum(recent_temps) / len(recent_temps)
                                target_date = today + datetime.timedelta(days=days_ahead)
                                daily_list = ""
                                for i in range(len(dates)):
                                    daily_list += f"{dates[i]}: {temps[i]}°F\n"
                                
                                weather_prompt = f"""Here's the RECENT weather data for {city_name}:

Average temperature (last 7 days): {avg_temp:.1f}°F
Highest: {max_temp}°F
Lowest: {min_temp}°F
Last 3 days: {recent_temps[0]}°F, {recent_temps[1]}°F, {recent_temps[2]}°F

Daily temperatures:
{daily_list}

Based on the recent trend, {target_date} is predicted to be around {predicted_temp:.1f}°F.

User's question: {user_input}

Answer their question using this recent weather data and prediction."""
                                response = model.generate_content(weather_prompt)
                                answer = response.text
                            else:
                                answer = f"Found {city_name} but couldn't get recent weather data."
                        else:
                            target_date = today + datetime.timedelta(days=days_ahead)
                            historical_data = []
                            years_to_check = 10  # Check last 10 years
                            
                            for year_offset in range(1, years_to_check + 1):
                                historical_date = target_date.replace(year=target_date.year - year_offset)
                                start = historical_date - datetime.timedelta(days=3)
                                end = historical_date + datetime.timedelta(days=3)
                                try:
                                    weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}&daily=temperature_2m_mean&temperature_unit=fahrenheit"
                                    weather_response = requests.get(weather_url, timeout=10).json()
                                    
                                    if "daily" in weather_response:
                                        year_temps = weather_response["daily"]["temperature_2m_mean"]
                                        year_avg = sum(year_temps) / len(year_temps)
                                        historical_data.append({
                                            'year': target_date.year - year_offset,
                                            'avg_temp': year_avg
                                        })
                                except:
                                    continue
                            if len(historical_data) > 0:
                                all_temps = [d['avg_temp'] for d in historical_data]
                                historical_avg = sum(all_temps) / len(all_temps)
                                historical_min = min(all_temps)
                                historical_max = max(all_temps)
                                history_text = ""
                                for entry in historical_data:
                                    history_text += f"{entry['year']}: {entry['avg_temp']:.1f}°F\n"
                                weather_prompt = f"""Here's HISTORICAL data for {city_name} around {target_date.strftime('%B %d')} from the past {years_to_check} years:

Historical average for this time of year: {historical_avg:.1f}°F
Historical range: {historical_min:.1f}°F to {historical_max:.1f}°F

Year-by-year data for this date:
{history_text}

Target prediction date: {target_date}
User's question: {user_input}

Based on this historical pattern, predict what the weather will be like on {target_date}. Use the historical average as your baseline and consider any trends you see in the data."""
                                response = model.generate_content(weather_prompt)
                                answer = response.text
                            else:
                                answer = f"Found {city_name} but couldn't get enough historical data for that time period."
                except Exception as e:
                    answer = f"Had trouble getting weather data: {str(e)}"
            else:
                response = model.generate_content(user_input)
                answer = response.text
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
