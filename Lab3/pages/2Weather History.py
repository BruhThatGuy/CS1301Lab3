import streamlit as st
import datetime
import requests
import pandas as pd

st.header("Weather History Selector")
st.write("")
st.write("---")
city=st.text_input("What city do you want to look at?","Atlanta")
st.write("---")
start = st.date_input("When would you like to start?", min_value=datetime.date(1995, 1, 1), max_value=datetime.date(2024, 10, 31))
st.write("You chose:",start)
st.write("---")
end = st.date_input("When would you like to end?", min_value=datetime.date(1995, 1, 1), max_value=datetime.date(2025, 10, 31))
st.write("You chose:",end)
st.write("---")
units=st.radio("What units do you want for temperature?",["Fahrenheit","Celcius"])
unitsurl=units.lower()
if st.button("Let's See"):
    if end < start:
        st.error("End date should be after start date. Try again")
        st.stop()
    try:
        url=f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
        endpoint=requests.get(url)
        data=endpoint.json()
        pophigh=0
        lat=0
        long=0
        if "results" not in data or len(data["results"])==0:
            st.error("City not found. Check spelling or try with a different city.")
            st.stop()
        for c in data["results"]:
            if "population" in c:
                if c["population"]>pophigh:
                    pophigh=c["population"]
                    lat=c["latitude"]
                    long=c["longitude"]
        if unitsurl=="fahrenheit":
            url2=f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={long}&start_date={start}&end_date={end}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean&hourly=temperature_2m&temperature_unit={unitsurl}"
        else:
            url2=f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={long}&start_date={start}&end_date={end}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean&hourly=temperature_2m"
        endpoint2=requests.get(url2)
        data2=endpoint2.json()
        if "daily" not in data2 or "temperature_2m_mean" not in data2["daily"]:
            st.error("No temperature data in this time frame. Try again.")
            st.stop()
        dates=data2["daily"]["time"]
        y=data2["daily"]["temperature_2m_mean"]
        st.write("---")
        st.write("Here is where you picked!")
        st.map(pd.DataFrame({"lat": [lat], "lon": [long]}))
        st.write("---")
        st.write("Here is the temperature for each day")
        st.line_chart(pd.DataFrame({f'Temperature in {units}':y},index=pd.to_datetime(dates)))
        st.write(f"**Average:** {round(sum(y)/len(y), 1)} °F")
        st.write(f"**Highest:** {max(y)} °F")
        st.write(f"**Lowest:** {min(y)} °F")
    except Exception as excep:
        st.error(f"Something else went wrong: {excep}")
