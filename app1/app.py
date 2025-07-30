import os
import csv
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain.agents import create_pandas_dataframe_agent
from langchain_community.llms import Ollama


st.title("Indian Meal Recommender (30–60 mins) ")


@st.cache_data
def scrape_recipes():
    url = "https://www.harighotra.co.uk/indian-recipes/time/indian-recipes-in-30-60-minutes/view-all"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')

    articles = soup.find_all('article')
    recipe = []
    for a in articles:
        name_tag = a.find("h3")
        time_tag = a.find('h4', attrs={"content": True})
        if name_tag and time_tag:
            name = name_tag.text.strip()
            time = time_tag.text.strip()
            recipe.append([name, time])
    return recipe


def time_min(time_str):
    time_str = str(time_str).lower().strip()
    minutes = 0
    if 'hr' in time_str:
        hr_part = time_str.split('hr')[0].strip()
        if hr_part.isdigit():
            minutes += int(hr_part) * 60
    if 'min' in time_str:
        min_part = ''.join(filter(str.isdigit, time_str.split('hr')[-1]))
        if min_part:
            minutes += int(min_part)
    return minutes


recipes = scrape_recipes()
df = pd.DataFrame(recipes, columns=["Dish", "Time"])
df["Time (mins)"] = df["Time"].apply(time_min)
st.sidebar.header("Filter Recipes by Time ⏱️")
min_time = st.sidebar.slider("Minimum time (minutes)", 0, 120, 0, 5)
max_time = st.sidebar.slider("Maximum time (minutes)", 10, 120, 60, 5)


filtered_df = df[(df["Time (mins)"] >= min_time) & (df["Time (mins)"] <= max_time)]


st.subheader("Filtered Recipes")
st.dataframe(filtered_df)




if st.button("Generate Meal Plan "):
    if filtered_df.empty:
        st.warning("No recipes found in that time range.")
    else:
        with st.spinner("Asking the LLM for today's meals..."):
            try:
                llm = Ollama(model='gemma3')
                agent = create_pandas_dataframe_agent(llm, filtered_df, allow_dangerous_code=True, verbose=True)

                query = """
                Pick 3 different meals for 3 days: one breakfast, one lunch, one dinner.
                Choose randomly and return them in a table with Meal, Dish Name, and Time (mins).
                """
                result = agent.run(query)
                st.success("Here’s your meal plan!")
                st.markdown(result)
            except Exception as e:
                st.error(f"Error: {e}")