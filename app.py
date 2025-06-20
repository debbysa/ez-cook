import streamlit as st
import pandas as pd
import requests
import os

st.set_page_config(page_title="ChefSort AI", page_icon="🍳")

st.title("🍳 ChefSort AI: Classify Recipes with LLaMA 3")
st.write("Upload the EPICurious CSV file from Kaggle to classify recipes using free LLaMA 3 via OpenRouter.")

# Load uploaded CSV file
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = df[['Title', 'Ingredients']].dropna().sample(15).reset_index(drop=True)
    return df

# Call OpenRouter API
def classify_recipe_with_llama(recipe_text: str) -> dict | None:
    api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.error("Missing OpenRouter API Key.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://chefsort.streamlit.app",  # replace with your app URL
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

# Upload section
uploaded_file = st.file_uploader("Upload `EPICurious.csv`", type=["csv"])

if uploaded_file:
    df = load_data("EPICurious.csv")

    selected_recipe = st.selectbox("Select a recipe to classify:", df['Title'])
    recipe_row = df[df['Title'] == selected_recipe].iloc[0]

    recipe_text = f"Title: {recipe_row['Title']}\nIngredients: {recipe_row['Ingredients']}"
    st.text_area("📋 Recipe Content", recipe_text, height=150)

    if st.button("🔍 Classify Recipe"):
        with st.spinner("Sending to LLaMA 3..."):
            result = classify_recipe_with_llama(recipe_text)
            if result:
                try:
                    output = result["choices"][0]["message"]["content"]
                    st.success("AI Classification Result:")
                    st.json(output)
                except Exception:
                    st.error("Failed to parse AI response.")
                    st.write(result)
else:
    st.info("Please upload the `EPICurious.csv` file to continue.")
