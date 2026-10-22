import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
import time
from neo4j import GraphDatabase
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import json
import re
import random
from rapidfuzz.distance import JaroWinkler
from sentence_transformers.util import cos_sim

#------ 1 CARGAR E INICIALIZAR OLLAMA Y NEO4J ------

# LLM de Ollama
llm = OllamaLLM(model="qwen2.5",
                    options={"temperature": 0.0})

# Datos de la base de datos de  Neo4j
server = "neo4j://127.0.0.1:17687"
username = "neo4j"
password = "TFGAmadeo" #Contraseña

# Inicializa el cliente de Neo4j
try:
    driver = GraphDatabase.driver(server, auth=(username, password))
except:
    print("Unable to reach Database")

# Función para usar el query de Neo4j
def run_query(query):
    with driver.session() as session:
        result = session.run(query)
        return result.data()

# Prompt para extraer ingredientes de recetas
prompt_ingredients = PromptTemplate(
    input_variables=["topic"],
    template = """
    Generate a recipe with the provided name, then extract each ingredient and its final preparation method.

Rules:
- Each ingredient must have its own preparation. 
- If the preparation method is not explicitly stated, infer the most likely one, preferibly from the list:  'steamed', 'fried', 'raw', 'boiled', 'roasted', 'pan-fried, 'stewed', 'sautéed', 'cooked'. 
- The ingredients have to be in singular, like 'potatoes' → 'potato', 'tomatoes' → 'tomato', 'leaves' → 'leaf', etc.
- Never leave a field empty of the JSON empty.

Return the result strictly as a JSON object with no explanations, no preamble, and no extra text. 
The JSON object must have this exact structure:

{{
  "recipe_name" : "<name of the recipe introduced>",
  "instructions":[
    {{
        "text": "<each one of the steps of the recipe>"
    }}
  ],
  "ingredients": [
    {{
      "name": "<ingredient name only, clean, singular>",
      "preparation": "<one word, chosen strictly from: 'steamed', 'fried', 'raw', 'boiled', 'roasted', 'pan-fried, 'stewed', 'sautéed', 'cooked'>",
      "amount": "<amount with unit, e.g., '1 cup', '200 grams', '2 units'>",
      "weight": "<normalized weight in grams>"
    }}
  ]
}}
---

Now, analyze the following recipe (serving size: one person):
{topic}

"""
)

#------ 2 FUNCIONES DE PROCESAMIENTO ------

def is_raw_like(text):
    text = text.lower()
    return any(x in text for x in ["raw", "fresh", "whole", "unprocessed"])

def is_processed_like(text):
    text = text.lower()
    return any(x in text for x in [
        "cooked", "boiled", "fried", "roasted", "processed", "steamed",
        "grilled", "dehydrated", "dried", "baked", "microwaved", "powdered", "smoked"
    ])


#------ 3 CACHING DE DATOS PESADOS ------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_food_list():
    food_list = run_query(
        "MATCH (f:Composition) RETURN f.food_name AS food_name, f.id AS id, f.FooDB_ID AS food_id"
    )
    df = pd.DataFrame(food_list)
    df['food_name'] = df['food_name'].fillna('').astype(str)
    df['is_raw'] = df['food_name'].apply(is_raw_like)
    df['is_processed'] = df['food_name'].apply(is_processed_like)
    return df

@st.cache_resource
def compute_embeddings(food_names):
    model = load_embedding_model()  # reuse cached model inside
    return model.encode(food_names, convert_to_tensor=True)

@st.cache_data
def load_facts(file_path="facts.txt"):
    with open(file_path, "r", encoding="utf-8") as file:
        facts = [line.strip() for line in file if line.strip()]
    return facts


#------ 4 BUSCADOR DE INGREDIENTES ------
def normalize_name(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-z\s]', '', text)  # elimina puntuación
    text = re.sub(r'\s+', ' ', text)
    return text
    
def find_best_matches(input_ingredients, df, db_embeddings, model, score_threshold=0.4):
    results = []

    for ingredient in input_ingredients:
        input_emb = model.encode(ingredient, convert_to_tensor=True)
        cos_scores = cos_sim(input_emb, db_embeddings)[0].cpu().numpy()

        df["semantic_score"] = cos_scores
        df["string_score"] = df["food_name"].apply(lambda x: JaroWinkler.normalized_similarity(x.lower(), ingredient.lower()))

        # Combinamos scores
        df["final_score"] = 0.7 * df["semantic_score"] + 0.3 * df["string_score"]

        # Filtramos términos procesados
        candidates = df[~df["is_processed"]].copy()

        # Tomamos el mejor
        best = candidates.sort_values(by="final_score", ascending=False).iloc[0]

        results.append({
            "input": ingredient,
            "food_name": best["food_name"],
            "id": int(best["id"]),
            "food_id": int(best["food_id"]),
            "score": round(float(best["final_score"]), 4)
        })

    return results




#------ 5 FUNCIONES AUXILIARES CON CACHE ------

@st.cache_data
def get_composition(id):
    constituents = run_query(f'MATCH (c:Composition {{id: {id}}}) RETURN c.constituents')
    return json.loads(constituents[0]["c.constituents"])

@st.cache_data
def get_disease(id):
    query = f'''
        MATCH (f:Food {{FooDB_id:{id}}})-[r:Affects]->(d:Disease)
        RETURN d.Disease AS Disease,
               r.`Suitable for Disease` AS Suitable,
               r.Link AS link
    '''
    return run_query(query)

def get_healthy_aging(food_id):
    query = f'''
        MATCH (h:`Envejecimiento Saludable` {{FooDB_ID:{food_id}}})
        RETURN h
    '''
    result = run_query(query)
    return result if result else None

def get_preparation(method):
    preparation = run_query(
        f'MATCH (m:Cooking_Methods {{Cooking_Method: "{method}"}}) RETURN m.Health_Impact AS impact, m.Sentence AS sentence, m.link AS link'
    )
    return preparation[0] if preparation else None

def update_search():
    st.session_state.search_query = st.session_state.search_input
    st.session_state.search_input = ""

def extract_numeric(val):
    if isinstance(val, (int, float)):
        return val
    match = re.search(r"[-+]?\d*\.?\d+", str(val))
    return float(match.group()) if match else 0

def get_tags(food_id, preparation=None):
    tags_html = []

    # --- HEALTH IMPACT ---
    diseases = get_disease(food_id)
    if diseases:
        positives = sum(1 for d in diseases if d["Suitable"])
        negatives = sum(1 for d in diseases if not d["Suitable"])
        if positives > 0 and negatives == 0:
            tags_html.append("<span style='background-color:#28a745; color:white; padding:3px 8px; border-radius:8px; font-size:12px; margin-left:5px;'>Healthy</span>")
        elif negatives > 0 and positives == 0:
            tags_html.append("<span style='background-color:#e74c3c; color:white; padding:3px 8px; border-radius:8px; font-size:12px; margin-left:5px;'>Unhealthy</span>")
        else:
            tags_html.append("<span style='background-color:#e67e22; color:white; padding:3px 8px; border-radius:8px; font-size:12px; margin-left:5px;'>Mixed Effects on Health</span>")
    else:
        tags_html.append("<span style='background-color:#5E5E5E; color:white; padding:3px 8px; border-radius:8px; font-size:12px; margin-left:5px;'>No Health Data</span>")
    
    # --- HEALTHY AGING ---
    results = get_healthy_aging(food_id)
    if results:
        levels = []
        for result in results:
            he = result.get("h", {})
            level = he.get("Healthy Aging_level")
            if level is not None:
                levels.append(level)

        if levels:
            positives = sum(1 for l in levels if l in [3, 4])
            negatives = sum(1 for l in levels if l in [0, 1])

            if positives > 0 and negatives == 0:
                tags_html.append("<span style='background-color:#28a745; color:white; padding:3px 8px; border-radius:8px; font-size:12px; margin-left:5px;'>Beneficial for Healthy Aging</span>")
            elif negatives > 0 and positives == 0:
                tags_html.append("<span style='background-color:#e74c3c; color:white; padding:3px 8px; border-radius:8px; font-size:12px; margin-left:5px;'>Harmful for Healthy Aging</span>")
            else:
                tags_html.append("<span style='background-color:#e67e22; color:white; padding:3px 8px; border-radius:8px; font-size:12px; margin-left:5px;'>Mixed Effects on Aging</span>")
        else:
            tags_html.append("<span style='background-color:#5E5E5E; color:white; padding:3px 8px; border-radius:8px; font-size:12px; margin-left:5px;'>No Aging Data</span>")
    else:
        tags_html.append("<span style='background-color:#5E5E5E; color:white; padding:3px 8px; border-radius:8px; font-size:12px; margin-left:5px;'>No Aging Data</span>")
    
    if preparation: 
        prep_data = get_preparation(preparation) 
        if prep_data: 
            impact = (prep_data.get("impact") or "").lower() 
            if impact == "good": 
                color = "#28a745" 
            elif impact == "moderate": 
                color = "#e67e22" 
            elif impact == "bad": 
                color = "#e74c3c" 
            else: 
                color = "#5E5E5E" 
        else: color = "#5E5E5E" 
        tags_html.append( f"<span style='background-color:{color}; color:white; padding:3px 8px; border-radius:8px; font-size:12px; margin-left:5px;'>{preparation.capitalize()}</span>" )

    return " ".join(tags_html)

def show_health_impact(food_id):
    diseases = get_disease(food_id)

    with st.expander("### 🧬 Health impact", expanded=False):
        if diseases:
            for disease in diseases:
                if disease["Suitable"]:
                    st.success(f"✅ Positive impact on **{disease['Disease']}** \n\n[More Info]({disease['link']})")
                else:
                    st.error(f"⚠️ Negative impact on **{disease['Disease']}** \n\n[More Info]({disease['link']})")
        else:
            st.info("No known health impacts.")
            
def show_nutrient_data(id):
    constituents = get_composition(id)

    # --- Diccionario de sinónimos ---
    nutrient_aliases = {
        "protein, total": "Proteins",
        "protein": "Proteins",
        "proteins": "Proteins",
        "carbohydrates, total": "Carbohydrate",
        "carbohydrate": "Carbohydrate",
        "energy": "Energy",
        "fat, total (lipids)": "Fat",
        "fat":"Fat",
        "total lipid (fat)": "Fat",
        "fiber, total dietary": "Fiber",
        "fiber (dietary)": "Fiber",
        "fiber, dietary": "Fiber",
        "Fiber (dietary)": "Fiber"
    }

    def normalize_nutrient_name(name):
        return nutrient_aliases.get(name.strip().lower(), name)
FoodMedKG Evaluation Survey
No valid survey block was found in this link.

Please use the exact link that was shared with you by the study coordinators (it should look like ...?block=3). If you believe this is an error, contact the research team rather than guessing a block number.
    # --- Normalizar y combinar nutrientes, conservando unidades ---
    normalized_constituents = {}
    for k, v in constituents.items():
        norm_key = normalize_nutrient_name(k.lower())

        num_val = extract_numeric(v)  # extrae solo el número
        unit = v.strip().replace(str(num_val), "").strip()  # extrae la unidad

        if norm_key in normalized_constituents:
            prev_val_str = normalized_constituents[norm_key]
            prev_num = extract_numeric(prev_val_str)
            prev_unit = prev_val_str.replace(str(prev_num), "").strip()

            # guardamos el valor mayor, manteniendo la unidad
            if num_val > prev_num:
                normalized_constituents[norm_key] = f"{num_val} {unit}".strip()
            else:
                normalized_constituents[norm_key] = f"{prev_num} {prev_unit}".strip()
        else:
            normalized_constituents[norm_key] = f"{num_val} {unit}".strip()

    # Ordenar por valor numérico
    sorted_constituents = sorted(
        normalized_constituents.items(),
        key=lambda x: extract_numeric(x[1]),
        reverse=True
    )

    with st.expander("🍽️ Nutrient Data", expanded=True):
        if not sorted_constituents:
            st.info("No nutrient data available.")
            return

        # Macronutrientes principales
        important_nutrients = ["Proteins", "Carbohydrate", "Energy", "Fat", "Fiber"]

        top_nutrients = [(nutrient, normalized_constituents[nutrient])
                        for nutrient in important_nutrients if nutrient in normalized_constituents]

        if top_nutrients:
            st.markdown("##### Key macronutrients")
            cols = st.columns(len(top_nutrients))
            for col, (nutrient, value) in zip(cols, top_nutrients):
                with col:
                    st.metric(label=nutrient, value=value)
        else:
            st.info("No key macronutrients found for this ingredient.")

        # Otros nutrientes
        other_nutrients = [(nutrient, val) for nutrient, val in sorted_constituents if nutrient not in important_nutrients]
        if other_nutrients:
            st.markdown("##### Full nutrient list")
            df_nutrients = pd.DataFrame(other_nutrients, columns=["Nutrient", "Value"])
            st.dataframe(df_nutrients, use_container_width=True, hide_index=True)

def show_healthy_aging(food_id):
    results = get_healthy_aging(food_id)

    with st.expander("🧓 Healthy Aging impact", expanded=False):
        if not results:
            st.info("#### 🧓 Healthy Aging impact\nNo data available.")
            return

        # Introductory note
        st.info(
            """
            ⚠️ **Important note**: The same ingredient can appear in different food groups  
            (e.g. The food *milk* will appear in the groups *Total dairy*, 
            *High-Fat and Regular-Fat Dairy*, *Low-Fat dairy*).  
            The results depend on the context, so results may differ.
            """
        )

        st.markdown("#### 🧓 Healthy Aging impact")

        # Mapping impact levels to description and color
        impact_map = {
            4: ("✅ Very beneficial to", "#28a745"),
            3: ("✅ Beneficial to", "#6ab04c"),
            2: ("No significant effect on", "#555555"),
            1: ("⚠️ Negative impact on", "#e67e22"),
            0: ("❌ Very negative impact on", "#e74c3c"),
        }

        card_style = """
            <div style="
                background-color: #f9f9f9;
                border-radius: 12px;
                padding: 12px;
                text-align: center;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                margin: 6px;
            ">
                <div style="font-size:16px; font-weight:600; color:{color};">{desc}</div>
                <div style="font-size:16px; color:#333;">{metric}</div>
            </div><br>
        """

        for result in results:
            he = result.get("h", {})

            group = he.get("Name", "Unknown group")
            st.markdown(
                f"##### The group **{group}** affects how a person ages in this way:"
            )

            # Metrics to display
            metrics = {
                "Healthy Aging": he.get("Healthy Aging_level"),
                "Cognitive Function": he.get("Intact Cognitive Function_level"),
                "Physical Function": he.get("Intact physical function_level"),
                "Mental Health": he.get("Intact mental health_level"),
                "Chronic Diseases": he.get("Free From Chronic Disease_level"),
                "Survived 70+ Years": he.get("Survived For 70 Years Of Age_level"),
            }

            cols = st.columns(len(metrics))
            for col, (metric, level) in zip(cols, metrics.items()):
                desc, color = impact_map.get(level, ("No known effects", "#888888"))
                col.markdown(
                    card_style.format(desc=desc, metric=metric, color=color),
                    unsafe_allow_html=True,
                )

def show_preparation(method, context="recipe"):
    preparation = get_preparation(method)
    if preparation:
        impact = (preparation.get("impact") or "").lower()
        sentence = preparation.get("sentence", "")
        link = preparation.get("link", "")
        with st.expander("👩‍🍳 Preparation Impact", expanded=False):
            if preparation:
                impact = (preparation.get("impact") or "").lower()
                sentence = preparation.get("sentence", "")
                link = preparation.get("link", "")

                if impact == "bad":
                    st.error(f"##### 👩‍🍳 **Impact of preparing the ingredient as {method}:**  \n❌ {sentence} \n\n[More Info]({link})")
                elif impact == "moderate":
                    st.warning(f"##### 👩‍🍳 **Impact of preparing the ingredient as {method}:**  \n⚠️ {sentence} \n\n[More Info]({link})")
                elif impact == "good":
                    st.success(f"##### 👩‍🍳 **Impact of preparing the ingredient as {method}:**  \n✅ {sentence} \n\n[More Info]({link})")
                else:
                    st.info(f"##### 👩‍🍳 **Impact of preparing the ingredient as {method}:** \n(no health info available)")

    else:
        with st.expander("👩‍🍳 Preparation Impact", expanded=False):
            st.info(f"##### 👩‍🍳 **Impact preparing the ingredient as {method}:** \n(no health info available)")

#------ 6 INICIALIZAR APLICACIÓN STREAMLIT ------

st.set_page_config(
    page_title="Healthy Food",
    layout="wide"
)

if "search_mode" not in st.session_state:
    st.session_state.search_mode = "recipes"
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "search_placeholder" not in st.session_state:
    st.session_state.search_placeholder = "Example: spaghetti with meatballs"

# Cargar datos solo una vez (cache)
model = load_embedding_model()
df = load_food_list()
db_embeddings = compute_embeddings(df['food_name'].tolist())
facts = load_facts()


#------ 7 APLICACIÓN ------

st.markdown(f"""
    <h1 style='text-align: center; color: #D9572A;'>
        Healthy Food
    </h1>
    <h3 style='text-align: center; color: #9AA3A8; font-weight: normal;'>
        Find what your meal is composed of
    </h3>
""", unsafe_allow_html=True)

sidebar, divider, main = st.columns([0.5, 0.02, 2.5])

with divider:
    st.markdown("<div style='border-left: 1px solid #ccc; height: 100dvh;'></div>", unsafe_allow_html=True)

with sidebar:
    st.markdown("### 🔍 Search options")
    if st.button("🍅 Search Ingredients"):
        st.session_state.search_mode = "ingredients"
        st.session_state.search_placeholder = "Example: potato"
        st.session_state.search_query = ""
    if st.button("🍲 Search Recipes"):
        st.session_state.search_mode = "recipes"
        st.session_state.search_placeholder = "Example: spaghetti with meatballs"
        st.session_state.search_query = ""

    st.write(f"Currently searching: **{st.session_state.search_mode.capitalize()}**")
    if st.session_state.search_mode == "ingredients":
        st.text_input(
            "Enter your search:",
            key="search_input",
            on_change=update_search,
            placeholder=st.session_state.search_placeholder,
            width=300
        )

        st.session_state.include_raw = st.checkbox("🔍 Search only raw ingredients", value=True)
    else:
        st.text_area(
            "Enter your search:",
            key="search_input",
            on_change=update_search,
            placeholder=st.session_state.search_placeholder,
            height=150,
            width=300
        )

    st.markdown("#### ℹ️ About this app")
    st.markdown(
        """
        This application is made to check the nutritional composition of foods,
        their potential health impacts, and their role in healthy aging.

        - The data comes from various sources like FooDB, Github, Pubmed and other scientific sites and studies.
        - The data was stored and formatted using MongoDB.
        - The data was added later to a graph database to find relationships between foods and diseases, using Neo4j.
        - The way of finding ingredients from a recipe is done thanks to a LLM made with Ollama.
        - The app was created and designed using Streamlit. 

        - This app was made from scratch by **Amadeo Martínez Sánchez** as a final degree project.
        """,
        unsafe_allow_html=True
    )

with main:
    query = st.session_state.search_query.lower()
    if query:
        with st.spinner(f"🔎 Did you know? {random.choice(facts)}"):
            MAX_RETRIES = 3
            if st.session_state.search_mode == "recipes":
                chain = prompt_ingredients | llm
                for attempt in range(MAX_RETRIES):
                    respuesta = chain.invoke({"topic": query})
                    print(respuesta)
                    try:
                        data = json.loads(respuesta)
                        break  # Si se pudo parsear, salimos del bucle
                    except json.JSONDecodeError:
                        if attempt == MAX_RETRIES - 1:
                            st.error("Could not parse recipe data. Try again.")
                            st.stop()

                recipe_title = data.get("recipe_name", query).strip().capitalize()
                st.markdown(f"## 🍲 Recipe: {recipe_title}")
                st.markdown("### 🍅 Ingredients")
                input_ingredients = [i["name"] for i in data["ingredients"]]
                amounts = [i["amount"] for i in data["ingredients"]]
                results = find_best_matches(input_ingredients, df, db_embeddings, model)
                ids = [r["id"] for r in results]
                for i, ing in enumerate(data["ingredients"]):
                    ingredient = ing["name"]
                    amount = ing["amount"]
                    preparation= ing["preparation"]
                    tags = get_tags(results[i]["food_id"], preparation)

                    st.markdown(f"- <span style='color:#000; font-weight:bold'>{ingredient}</span>: {amount} {tags}", unsafe_allow_html=True)

                    # Mostrar impactos de salud/nutrición
                    with st.expander(f"Most similar ingredient found: {results[i]['food_name']}"):
                        show_preparation(preparation, "ingredient")  # quí usamos el método específico del ingrediente
                        show_health_impact(results[i]["food_id"])
                        show_healthy_aging(results[i]["food_id"])
                        show_nutrient_data(results[i]["id"])

                # Mostrar instrucciones de la receta
                st.markdown("### 👨‍🍳 Instructions")

                if "instructions" in data and data["instructions"]:
                    for step in data["instructions"]:
                        st.markdown(f"- <span style='color:#000; font-weight:bold'>{step['text']}</span>", unsafe_allow_html=True)
                else:
                    st.info("No instructions found for this recipe.")



            elif st.session_state.search_mode == "ingredients":
                if st.session_state.get("include_raw", False):
                    query = query + " raw"

                results = find_best_matches([query], df, db_embeddings, model)
                food_name = results[0]["food_name"]
                tags = get_tags(results[0]["food_id"])
                st.markdown(
                    f"## Found ingredient: <span style='color:#000; font-weight:bold'>{food_name}</span> {tags}",
                    unsafe_allow_html=True
                )

                id = results[0]["id"]
                show_health_impact(results[0]["food_id"])
                show_healthy_aging(results[0]["food_id"])
                show_nutrient_data(id)
