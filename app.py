import streamlit as st
import pandas as pd
import requests
import os
import json
import re

st.set_page_config(page_title="ChefSort AI", page_icon="🍳")

st.title("🍳 Ez Cook: Classify Recipes with LLaMA 3")
st.write("This app uses free LLaMA 3 via OpenRouter to classify recipes from the EPICurious dataset.")

# Load CSV directly from the project directory
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("food_ingredients.csv")
        df = df[['Title', 'Ingredients']].dropna().sample(15).reset_index(drop=True)
        return df
    except FileNotFoundError:
        st.error("The file 'food_ingredients.csv' was not found in the project directory.")
        return pd.DataFrame()

# Call OpenRouter API
def classify_recipe_with_llama(recipe_text: str) -> dict | None:
    api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.error("Missing OpenRouter API Key.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://chefsort.streamlit.app",  # replace with your actual Streamlit Cloud app URL
        "Content-Type": "application/json"
    }

    system_prompt = (
        "You are a culinary AI assistant. Classify the given recipe (title and ingredients) "
        "into structured tags: cuisine, meal_type (breakfast/lunch/dinner/snack), "
        "dietary_tags (vegan, keto, gluten-free, etc.), and difficulty (easy, medium, hard). "
        "Respond in JSON format with keys: cuisine, meal_type, dietary_tags, difficulty."
    )

    data = {
        "model": "meta-llama/llama-3.3-8b-instruct:free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": recipe_text}
        ]
    }

    res = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
    if res.status_code == 200:
        return res.json()
    else:
        st.error(f"Error from OpenRouter: {res.status_code}")
        st.code(res.text)
        return None

# Load dataset
df = load_data()

if not df.empty:
    selected_recipe = st.selectbox("Select a recipe to classify:", df['Title'])
    recipe_row = df[df['Title'] == selected_recipe].iloc[0]

    recipe_text = f"Title: {recipe_row['Title']}\nIngredients: {recipe_row['Ingredients']}"
    st.text_area("📋 Recipe Content", recipe_text, height=150)

    if st.button("🔍 Classify Recipe"):
        with st.spinner("Sending to LLaMA 3..."):
            result = classify_recipe_with_llama(recipe_text)

            if result:
                try:
                    # Raw AI text output
                    raw_output = result["choices"][0]["message"]["content"]

                    # Remove code fences ```json ... ```
                    clean_text = re.sub(r"^```(json)?|```$", "", raw_output.strip(), flags=re.IGNORECASE).strip()

                    # Attempt JSON parse
                    parsed = json.loads(clean_text)

                    st.success("AI Classification Result:")
                    st.json(parsed)

                except json.JSONDecodeError:
                    st.warning("Couldn't parse response as JSON. Showing raw result below:")
                    st.code(result["choices"][0]["message"]["content"])

                except Exception as e:
                    st.error("Unexpected error while handling AI output.")
                    st.exception(e)
else:
    st.warning("Dataset not loaded. Please make sure 'food_ingredients.csv' is in the same folder as this script.")
