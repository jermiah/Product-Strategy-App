from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import streamlit as st
import pandas as pd
import joblib
import faiss
from sentence_transformers import SentenceTransformer
import torch
import os
import requests
import gc
from streamlit_searchbox import st_searchbox
import time
import json
import datetime
import numpy as np
from streamlit_option_menu import option_menu
import shutil
import re
from dotenv import load_dotenv

load_dotenv()

os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# -------------------------------
# Page Config & Custom CSS 
# -------------------------------
st.set_page_config(
    page_title="GrandView by Prosol",
    page_icon=":package:",
    layout="wide"
)

# ------------------------
# Top Navigation Menu
# ------------------------

selected = option_menu(
    menu_title=None,
    options=[
        "Segment & Export Products", 
        "Product Strategy Builder", 
        "Strategy Explorer", 
        "Strategy Feedback Vault", 
        "Maintenance"
    ],
    icons=["cloud-upload", "box", "search", "folder", "tools"],
    menu_icon="cast",
    default_index=1,
    orientation="horizontal",
    styles={
        "container": {
    "padding": "0",
    "background-color": "transparent",
    "border-bottom": "1px solid rgba(0,0,0,0.1)",
    "margin-bottom": "10px",
    "box-shadow": "none",           
    "backdrop-filter": "blur(0px)"  
        },
        "icon": {"color": "#FF5E5B", "font-size": "20px"},
        "nav-link": {
            "font-size": "15px",
            "font-weight": "500",
            "white-space": "nowrap", 
            "text-align": "center",
            "padding": "10px 20px",
            "margin": "0 10px",
        },
        "nav-link-selected": {
            "background-color": "#FF5E5B",
            "color": "white",
            "font-weight": "bold",
            "border-radius": "6px"
        },
    }
)
# Background gradient
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right top, #c8b1b1, #cfb1a6, #cdb399, #c1b990, #abc091, #9ccaa6, #90d2be, #8cd8d6, #abe0ee, #cee8fa, #ecf2fe, #ffffff);
    }

    /* Optionally shrink container margins */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Optional: tighten the top spacing */
    header {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)


# -------------------------------
# Company Logos 
# -------------------------------

col_logo_left, col_title, col_logo_right = st.columns([1, 3, 1])

with col_logo_left:
    st.image("E:/Prosol/emlyon_logo.png", width=120)

with col_title:
    st.markdown(
        """
        <h1 style='text-align: center; font-size: 36px; margin-bottom: 0;'>GrandView by Prosol</h1>
        <p style='text-align: center; font-size: 18px; color: #333; font-weight: 500; margin-top: 10px;'>
        🧭 Strategic clarity for every product on the shelf🧭 
        </p>
        <p style='text-align: center; font-size: 16px; color: #555; margin-top: 4px;'>
        Insights are derived exclusively from Grand Frais store data across five cities: <em>Vaulx-en-Velin, Bron, Meyzieu, Décines, and Mions</em>.
        </p>
        
        """,
        unsafe_allow_html=True
    )


with col_logo_right:
    st.image("E:/Prosol/prosol logo.png", width=150)


# Define Gemini API details
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SIMILARITY_THRESHOLD_RAG = 0.4      # Threshold for retrieving RAG context
SIMILARITY_THRESHOLD_REFERENCE = 0.7 # Threshold for finding past reference strategies


# Function to call Gemini API
def run_gemini_api(prompt):
    if not GEMINI_API_KEY:
        st.error("Gemini API key is not set. Please set the GEMINI_API_KEY environment variable.")
        return None
    
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.6,
            "topP": 0.85,
            "maxOutputTokens": 700
        }
    }
    params = {"key": GEMINI_API_KEY}
    
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            st.error(f"API error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# ===============================
# Load Model  
# ===============================
@st.cache_resource
def load_llm():
    clear_gpu()
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True 
    )

    model.eval()
    return model, tokenizer


model, tokenizer = load_llm()

def run_llm(system_prompt, user_prompt, model, tokenizer, temperature=0.7, top_p=0.9, max_new_tokens=300, repetition_penalty=1.2):
    # Debug checks
    if not isinstance(system_prompt, str):
        raise ValueError(f"❌ system_prompt must be a string, got: {type(system_prompt)}")

    if not isinstance(user_prompt, str):
        raise ValueError(f"❌ user_prompt must be a string, got: {type(user_prompt)}\n\nuser_prompt content:\n{user_prompt}")

    # Apply chat template
    chat_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        add_generation_prompt=True,
        tokenize=False
    )

    if not isinstance(chat_prompt, str):
        raise ValueError(f"❌ chat_prompt must be a string. Got: {type(chat_prompt)}")

    chat_prompt = chat_prompt.strip()

    # Final check before calling tokenizer
    try:
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    except Exception as e:
        raise ValueError(f"🔥 Tokenizer failed. Chat prompt:\n{chat_prompt[:500]}\n\nError: {e}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id  
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def show_feedback_viewer():
    st.markdown("#### Product Feedback Viewer")

    # Load feedback memory
    feedback_path = "Data/feedback_memory.json"
    if not os.path.exists(feedback_path):
        st.info("No feedback has been saved yet.")
        return

    with open(feedback_path, "r") as f:
        feedback_list = json.load(f)

    product_names = sorted(list(set(entry["product_name"] for entry in feedback_list)))


    # -------------------------
    # Feedback Search Viewer
    # -------------------------
    selected_product = st_searchbox(
        lambda term: [p for p in product_names if term.lower() in p.lower()],
        key="feedback_searchbox",
        placeholder="Search product name..."
    )

    if selected_product:
        st.markdown(f"#### Feedback for: **{selected_product}**")
        matching_entries = [fb for fb in feedback_list if fb["product_name"] == selected_product]

        if not matching_entries:
            st.warning("No feedback found for this product.")
        else:
            for i, fb in enumerate(matching_entries):
                with st.expander(f"Feedback #{i+1} | {fb['timestamp']} | {fb['feedback']}", expanded=False):
                    st.markdown(f"**Original Strategy:**\n\n{fb['original_strategy']}")
                    if fb.get("corrected_strategy"):
                        st.markdown(f"**Corrected Strategy:**\n\n{fb['corrected_strategy']}")
                    if fb.get("negative_feedback_reason"):
                        st.warning(f"**Reason for 👎:** {fb['negative_feedback_reason']}")

                    # Delete option
                    if st.button(f"🗑️ Delete this feedback", key=f"delete_{i}"):
                        feedback_list.remove(fb)
                        with open(feedback_path, "w") as f:
                            json.dump(feedback_list, f, indent=2)
                        st.success("Deleted successfully.")
                        st.rerun()



    # -------------------------
    # Manual Strategy Entry
    # -------------------------
    st.markdown("---")
    st.markdown("#### ➕ Manually Add a New Product Strategy")
    st.info("Use this only if a product does not already exist in the feedback records. This is for new strategies, not corrections.")

    with st.form("manual_add_form"):
        new_product = st.text_input("Product Name")
        user_defined_strategy = st.text_area("Strategy You Recommend", height=200)
        submitted = st.form_submit_button("Save New Strategy")

        if submitted:
            if not new_product or not user_defined_strategy.strip():
                st.error("🚫 Product name and strategy are required.")
            else:
                existing_names = [entry["product_name"].lower() for entry in feedback_list]
                if new_product.lower() in existing_names:
                    st.warning("⚠️ This product already exists in feedback. Please update it through the strategy generation interface.")
                else:
                    entry = {
                        "strategy_type": "product",
                        "product_name": new_product,
                        "original_input": "manual_entry",
                        "product_vector": embed_text(new_product).tolist(), 
                        "original_strategy": user_defined_strategy.strip(),
                        "corrected_strategy": user_defined_strategy.strip(),
                        "manual_edit": True,
                        "feedback": "👍",
                        "negative_feedback_reason": None,
                        "timestamp": str(datetime.datetime.now())
                    }
                    save_to_memory(entry, strategy_type="product")
                    st.success("✅ New product strategy saved successfully!")


def show_maintenance_tools():

    st.markdown("##### ⚙️ Maintenance Tools")

    if st.button("🧹 Clear GPU Memory"):
        clear_gpu()
        st.success("✅ GPU memory cleared!")

    if st.button("🔄 Clear App Cache"):
        st.warning("Streamlit cache cleared. Please manually rerun or reload the app.")
        st.cache_resource.clear()
        st.cache_data.clear()

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**2
        st.markdown(f"🔋 **GPU Memory Used:** `{gpu_mem:.1f} MB`")

    st.markdown("##### 🛠️ Feedback File Maintenance")

    feedback_files = {
        "Product Feedback": "Data/feedback_memory.json",
        "General Feedback": "Data/general_feedback_memory.json"
    }

    for label, path in feedback_files.items():
        st.markdown(f"#### {label}")
        col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])

        # 🔍 Check Integrity
        with col1:
            if st.button(f"🔍 Check Integrity [{label}]"):
                if not os.path.exists(path):
                    st.error(f"❌ No such file found: `{path}`")
                else:
                    try:
                        with open(path, "r") as f:
                            json.load(f)
                        st.success(f"✅ {label} file is valid.")
                    except json.JSONDecodeError:
                        st.warning(f"⚠️ {label} file is corrupted!")

        # 🧹 Attempt Repair
        with col2:
            if st.button(f"🧹 Attempt Repair [{label}]"):
                if not os.path.exists(path):
                    st.error(f"❌ No such file found: `{path}`")
                else:
                    try:
                        with open(path, "r") as f:
                            raw = f.read()
                        trimmed = raw.rsplit("}", 1)[0] + "}]"
                        fixed_data = json.loads(trimmed)
                        backup_path = path.replace(".json", "_backup_corrupt.json")
                        shutil.copy(path, backup_path)
                        with open(path, "w") as f:
                            json.dump(fixed_data, f, indent=2)
                        st.success(f"✅ Repaired successfully. Backup saved at `{backup_path}`.")
                    except Exception as e:
                        st.error(f"❌ Repair failed: {e}")

        # 📥 Download File as Backup
        with col3:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{label.replace(' ', '_').lower()}_{timestamp}.json"
            if os.path.exists(path):
                with open(path, "r") as f:
                    file_content = f.read()
                file_bytes = file_content.encode("utf-8")

                st.download_button(
                    label=f"📥 Download Backup of [{label}]",
                    data=file_bytes,
                    file_name=filename,
                    mime="application/json",
                    key=f"download_{label}"
                )
            else:
                st.download_button(
                    label=f"📥 Download Backup [{label}]",
                    data="",
                    file_name="missing.json",
                    mime="application/json",
                    disabled=True,
                    key=f"download_disabled_{label}"
                )
                st.error(f"❌ No such file found:\n`{path}`")

        # ❌ Reset File
        with col4:
            if st.button(f"❌ Reset File [{label}]"):
                if not os.path.exists(path):
                    st.error(f"❌ No such file found: `{path}`")
                else:
                    try:
                        shutil.copy(path, path.replace(".json", "_reset_backup.json"))
                        with open(path, "w") as f:
                            json.dump([], f, indent=2)
                        st.success("File reset to empty list. Previous version backed up.")
                    except Exception as e:
                        st.error(f"❌ Could not reset: {e}")

# ===============================
# Feedback Memory Helpers
# ===============================

def embed_text(text):
    vec = embed_model.encode([text], normalize_embeddings=True)
    return vec.astype("float32")[0]


def save_to_memory(entry, strategy_type="product"):
    """
    Save feedback entry to the appropriate memory file.
    strategy_type: "product" or "general"
    """
    if strategy_type == "general":
        path = "Data/general_feedback_memory.json"
    else:
        path = "Data/feedback_memory.json"

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def search_similar_strategy(query_text_or_vector,
                            product_faiss_index,  
                            general_faiss_index,  
                            strategy_type="product",
                            top_k=1,
                            is_vector=False,
                            similarity_threshold=0.7): # Add a threshold
    """
    Search past feedback strategies for either product or general queries.
    If is_vector=True, `query_text_or_vector` is treated as an embedded vector.
    Uses the appropriate FAISS index based on strategy_type.
    Returns only entries above the similarity_threshold.
    """
    memory_list = []
    active_index = None
    memory_path = ""

    if strategy_type == "general":
        memory_path = "Data/general_feedback_memory.json"
        active_index = general_faiss_index
    else: # product
        memory_path = "Data/feedback_memory.json"
        active_index = product_faiss_index

  
    if active_index is None: 
        return []
    if not os.path.exists(memory_path):
        return []

    try:
        with open(memory_path, "r") as f:
            memory_list = json.load(f)
    except json.JSONDecodeError:
        st.error(f"Memory file {memory_path} is corrupt.")
        return []
    except Exception as e:
        st.error(f"Error reading memory file {memory_path}: {e}")
        return []

    if not memory_list:
        return []


    if is_vector:
        try:
            query_vector = np.array(query_text_or_vector, dtype="float32").reshape(1, -1)
            if query_vector.shape[1] != active_index.d:
                 st.error(f"Query vector dimension ({query_vector.shape[1]}) does not match index dimension ({active_index.d}).")
                 return []
        except Exception as e:
            st.error(f"Error processing input vector: {e}")
            return []
    else:
        if embed_model is None:
            st.error("Embedding model not loaded. Cannot perform text search.")
            return []
        try:
            query_vector = embed_model.encode([query_text_or_vector], normalize_embeddings=True)
            query_vector = query_vector.astype("float32").reshape(1, -1)
            if query_vector.shape[1] != active_index.d:
                 st.error(f"Embedded query vector dimension ({query_vector.shape[1]}) does not match index dimension ({active_index.d}).")
                 return []
        except Exception as e:
            st.error(f"Error embedding query text: {e}")
            return []

    # --- Perform Search ---
    try:
        # D: distances (or similarities for IP), I: indices (or IDs if using IndexIDMap)
        scores, memory_indices = active_index.search(query_vector, top_k)
    except Exception as e:
        st.error(f"FAISS search failed for {strategy_type}: {e}")
        return []

    # --- Filter and Map Results ---
    similar_entries = []
    if memory_indices.size > 0:
        for i in range(memory_indices.shape[1]): # Iterate through the top_k results for the first query
            mem_idx = memory_indices[0, i]
            score = scores[0, i]
            # Check if the index is valid and score meets threshold
            if mem_idx != -1 and score >= similarity_threshold: # FAISS returns -1 for no result or if ID mapping fails
                # mem_idx should directly correspond to the position in memory_list
                # because we used IndexIDMap with original indices.
                if 0 <= mem_idx < len(memory_list):
                    entry = memory_list[mem_idx]
                    entry['_similarity_score'] = float(score) 
                    similar_entries.append(entry)
                    #
                else:
                     print(f"Warning: FAISS returned invalid memory index {mem_idx} for {strategy_type}.")
    return similar_entries


@st.cache_resource
def load_or_build_general_faiss_index(memory_path="Data/general_feedback_memory.json", index_path="Data/general_vectors.index"):
    """Loads or builds a FAISS index for general strategy feedback vectors."""
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    memory_exists = os.path.exists(memory_path)
    index_exists = os.path.exists(index_path)
    general_index = None
    memory_data = []

    # Load memory data first
    if memory_exists:
        try:
            with open(memory_path, "r") as f:
                memory_data = json.load(f)
        except json.JSONDecodeError:
            st.error(f"Error decoding {memory_path}. Cannot build general index.")
            return None
        except Exception as e:
            st.error(f"Error loading {memory_path}: {e}")
            return None

    if not memory_data:
        # If memory is empty, no index needed or possible
        if index_exists: # Clean up old index if memory is now empty
             try: os.remove(index_path)
             except: pass
        return None # Return None if no data

    if index_exists:
        try:
            general_index = faiss.read_index(index_path)
            # Verify if the number of vectors matches the memory list size
            if general_index.ntotal == len(memory_data):
                print(f"Loaded existing general FAISS index from {index_path} with {general_index.ntotal} vectors.")
                return general_index
            else:
                print(f"Index size ({general_index.ntotal}) mismatch with memory size ({len(memory_data)}). Rebuilding.")
                general_index = None 
        except Exception as e:
            print(f"Could not load existing general index {index_path}, rebuilding. Error: {e}")
            general_index = None

    print("Building new general FAISS index...")
    vectors = []
    valid_indices = [] # Keep track of which memory entries had valid vectors
    dimension = -1
    for i, entry in enumerate(memory_data):
        # Use 'retrieved_vector' which stores the embedding of the context used
        vec = entry.get("retrieved_vector")
        if vec and isinstance(vec, list):
            try:
                np_vec = np.array(vec, dtype="float32")
                if dimension == -1:
                    dimension = len(np_vec)
                if len(np_vec) == dimension: # Ensure consistent dimension
                    vectors.append(np_vec)
                    valid_indices.append(i) # Store original index
                else:
                    print(f"Skipping entry {i} due to dimension mismatch.")
            except Exception as e:
                 print(f"Skipping entry {i} due to vector conversion error: {e}")
        else:
            print(f"Skipping entry {i} due to missing or invalid 'retrieved_vector'.")

    if not vectors or dimension == -1:
        st.warning("No valid vectors found in general feedback memory to build index.")
        if index_exists: # Clean up old index if no valid vectors found
            try: os.remove(index_path)
            except: pass
        return None

    try:
        vector_matrix = np.vstack(vectors)
        # FAISS index for inner product (IP) similarity search
        general_index = faiss.IndexFlatIP(dimension)
        general_index = faiss.IndexIDMap(general_index)
        general_index.add_with_ids(vector_matrix, np.array(valid_indices, dtype='int64'))

        faiss.write_index(general_index, index_path)
        print(f"Built and saved general FAISS index to {index_path} with {general_index.ntotal} vectors.")
        return general_index
    except Exception as e:
        st.error(f"Error building or saving general FAISS index: {e}")
        return None



def show_feedback_viewer():
    st.markdown("##### Product Feedback Viewer")

    # Load product and general feedback
    feedback_path = "Data/feedback_memory.json"
    general_feedback_path = "Data/general_feedback_memory.json"

    feedback_list = []
    general_feedbacks = []

    if os.path.exists(feedback_path):
        with open(feedback_path, "r") as f:
            feedback_list = json.load(f)

    if os.path.exists(general_feedback_path):
        with open(general_feedback_path, "r") as f:
            general_feedbacks = json.load(f)

    # -----------------------------
    # Product Feedback Viewer
    # -----------------------------
    product_names = sorted(list(set(entry["product_name"] for entry in feedback_list)))
    selected_product = st_searchbox(
        lambda term: [p for p in product_names if term.lower() in p.lower()],
        key="feedback_searchbox",
        placeholder="Search product name..."
    )

    if selected_product:
        st.markdown(f"#### Feedback for: **{selected_product}**")

        filter_choice = st.radio(
            "Show feedback type:",
            ["All", "Positive", "Negative"],
            horizontal=True,
            key="filter_type_selector"
        )

        matching_entries = [fb for fb in feedback_list if fb.get("product_name") == selected_product]

        # Apply filter properly
        if filter_choice == "Positive":
            matching_entries = [fb for fb in matching_entries if fb.get("feedback") == "👍"]
        elif filter_choice == "Negative":
            matching_entries = [fb for fb in matching_entries if fb.get("feedback") == "👎"]

        if not matching_entries:
            st.warning("No feedback found for this product.")
        else:
            for i, fb in enumerate(matching_entries):
                with st.expander(f"📄 Feedback #{i+1} | {fb['timestamp']} | {fb['feedback']}", expanded=False):
                    if fb.get("feedback") == "👍":
                        st.markdown(f"**✅ Strategy:**\n\n{fb.get('corrected_strategy', 'No strategy available.')}")
                    elif fb.get("feedback") == "👎":
                        st.markdown(f"**📝 Original Strategy:**\n\n{fb.get('original_strategy', 'No strategy available.')}")
                        if fb.get("negative_feedback_reason"):
                            st.warning(f"**Reason for 👎:** {fb['negative_feedback_reason']}")

                    if st.button(f"🗑️ Delete this feedback", key=f"delete_{i}"):
                        feedback_list.remove(fb)
                        with open(feedback_path, "w") as f:
                            json.dump(feedback_list, f, indent=2)
                        st.success("Deleted successfully.")
                        st.rerun()

    # -----------------------------
    # 🧠 General Strategy Viewer
    # -----------------------------
    st.markdown("---")
    st.markdown("##### General Strategy Feedback Viewer")

    if not general_feedbacks:
        st.info("No general strategy feedback found.")
    else:
        dropdown_options = [
            f"#{i+1} | {fb['timestamp']} | {fb.get('original_query', fb.get('original_input', ''))[:60]}..."
            for i, fb in enumerate(general_feedbacks)
        ]
        selected_index = st.selectbox(
            "Select a general feedback entry to view:",
            range(len(dropdown_options)),
            format_func=lambda i: dropdown_options[i],
            key="general_feedback_dropdown"
        )

        toggle_key = f"show_general_{selected_index}"
        if toggle_key not in st.session_state:
            st.session_state[toggle_key] = False

        st.session_state[toggle_key] = st.toggle(
            "👁️ Show/Hide Strategy Details",
            value=st.session_state[toggle_key],
            key=f"toggle_button_{selected_index}"
        )

        if st.session_state[toggle_key]:
            selected_fb = general_feedbacks[selected_index]
            st.markdown("##### Original Query")
            st.code(selected_fb.get("original_query", selected_fb.get("original_input", "N/A")), language="markdown")
            st.markdown("##### Original Strategy")
            st.markdown(selected_fb.get("original_strategy", "N/A"))
            if selected_fb.get("corrected_strategy"):
                st.markdown("##### Corrected Strategy")
                st.success(selected_fb["corrected_strategy"])
            if selected_fb.get("negative_feedback_reason"):
                st.warning(f"**Reason for 👎:** {selected_fb['negative_feedback_reason']}")

            if st.button("🗑️ Delete this general feedback", key=f"delete_general_{selected_index}"):
                general_feedbacks.remove(selected_fb)
                with open(general_feedback_path, "w") as f:
                    json.dump(general_feedbacks, f, indent=2)
                st.success("Deleted successfully.")
                st.rerun()

    # -----------------------------
    # Manual Entry
    # -----------------------------
    st.markdown("---")
    st.markdown("##### Manually Add Feedback")

    with st.form("manual_add_form"):
        strategy_type = st.selectbox("Is this for a product or general query?", ["Product", "General"], key="manual_type")
        name_or_query = st.text_input("Product Name or General Query")
        manual_strategy = st.text_area("Strategy You Recommend", height=200)
        submitted = st.form_submit_button("💾 Save Manual Strategy")

        if submitted:
            if not name_or_query.strip() or not manual_strategy.strip():
                st.error("🚫 Both fields are required.")
            else:
                entry = {
                    "strategy_type": strategy_type.lower(),
                    "original_input": "manual_entry",
                    "original_strategy": manual_strategy.strip(),
                    "corrected_strategy": manual_strategy.strip(),
                    "manual_edit": True,
                    "feedback": "👍",
                    "negative_feedback_reason": None,
                    "timestamp": str(datetime.datetime.now())
                }

                if strategy_type == "Product":
                    entry["product_name"] = name_or_query
                    entry["product_vector"] = embed_text(name_or_query).tolist()
                else:
                    entry["original_query"] = name_or_query
                    entry["retrieved_vector"] = embed_text(name_or_query).tolist()
                    entry["retrieved_context"] = "manual_general_entry"

                save_to_memory(entry, strategy_type=strategy_type.lower())
                st.success("✅ Strategy saved successfully!")
                st.rerun()


# ===============================
# Load Embeddings + RAG DF + Indexes
# ===============================
@st.cache_resource
def load_product_faiss_index():
    try:
        return faiss.read_index("./Data/product_vectors.index")
    except Exception as e:
        st.error(f"Failed to load product FAISS index: {e}")
        return None

@st.cache_data
def load_rag_df():
    try:
        return pd.read_parquet("./Data/rag_df.parquet")
    except Exception as e:
        st.error(f"Failed to load RAG dataframe: {e}")
        return pd.DataFrame() # Return empty DF

@st.cache_resource
def load_embed_model():
    try:
        return SentenceTransformer("BAAI/bge-base-en-v1.5")
    except Exception as e:
        st.error(f"Failed to load SentenceTransformer model: {e}")
        return None

# Load resources
product_index = load_product_faiss_index()
general_index = load_or_build_general_faiss_index() # Load the general index here
rag_df = load_rag_df()
embed_model = load_embed_model()

# Add checks to ensure models/indexes loaded
if product_index is None or rag_df.empty or embed_model is None:
    st.error("Critical error: Failed to load necessary data or models. App cannot function correctly.")
    st.stop() # Stop execution if core components failed


# ===============================
# Prompting & Cleaning
# ===============================

def get_product_prompt(model_choice, context):
    if model_choice == "On-Premise":
        return f"""
You are a working with sales metrics and segmentation data for a specific product from 5 physical Grand Frais stores located in Lyon.

Important context:
- These stores are part of **Prosol**, a French retail group known for its offline Grand Frais stores that specialize in fresh produce, meat, seafood, and groceries.
- The data comes **only from physical (offline) stores**. There is no e-commerce or delivery platform involved.
- All recommendations must be relevant **only to offline retail strategy**.
- Avoid suggesting digital marketing, email campaigns, apps, or social media ads.
- Focus on In-store promotions and pricing tactics, Cross-merchandising and shelf placement, in-store signage, product bundling, physical loyalty programs, timing-based offers and store layout optimization.

Here is the Product data and segmentation insights:

🔹 This is the data for one product. All strategies should be based on this product’s performance. You may suggest bundling or synergies if relevant, but do not describe or analyze other specific products.

{context}

Please respond with a complete strategy in the required format.
"""

    
    elif model_choice == "Gemini API":
        return f"""
You are a seasoned Business Intelligence Strategist specializing in retail product performance. 
You are working with sales and segmentation data from 5 physical Grand Frais stores located in Lyon.

Important context:
- These stores are part of **Prosol**, a French retail group known for its offline Grand Frais stores that specialize in fresh produce, meat, seafood, and groceries.
- The data comes **only from physical (offline) stores**. There is no e-commerce or delivery platform involved.
- All recommendations must be relevant **only to offline retail strategy**.
- Avoid suggesting digital marketing, email campaigns, apps, or social media ads.
- Focus on In-store promotions and pricing tactics, Cross-merchandising and shelf placement, in-store signage, product bundling, physical loyalty programs, timing-based offers and store layout optimization.

Your task is to generate a focused, actionable business strategy for the product described below. This product-level data includes key metrics and segmentation insights.

📌 **Important Instructions:**
- The strategy must be tailored only to the product mentioned. Avoid discussing store-wide or generic strategies.
- Do NOT summarize the input data.
- Do NOT describe the context or your reasoning. Go straight to the recommendations.
- The strategy must be organized into the **five sections** below.

**Format:**
1. **Introduction** (1-2 lines setting context or objective)
2. **Strategy 1: Cross-Selling or Synergies**  
3. **Strategy 2: Marketing & Promotion**  
4. **Strategy 3: Additional Growth Opportunities**  
5. **Conclusion** (1-2 lines to close with clarity or next steps)

**Tone & Style:**
- Sharp, confident, and executive-level.
- Use bullet points only if needed for clarity.
- Avoid over-explaining. Assume the reader understands retail KPIs.

Here is the product data:

🔹 This is the data for one product. All strategies should be based on this product’s performance. You may suggest bundling or synergies if relevant, but do not describe or analyze other specific products.

{context}

Start your response immediately below. Do not include any additional headers or explanations.
"""


def get_query_prompt(model_choice, user_query, context):
    if model_choice == "On-Premise":
        return f"""
A user has asked:

\"{user_query}\"

Context:
- The product data have sales metrics, product segmentation and descriptions from 5 physical Grand Frais stores located in Lyon.
- These are **offline retail stores** (not online, no e-commerce or delivery).
- Grand Frais is part of Prosol, a French retail company focused on fresh produce, meats, cheeses, and seafood.
- All recommendations must apply strictly to **in-store (offline)** strategies.
- Do NOT suggest anything involving websites, email campaigns, apps, or social media marketing.
- Focus instead on In-store promotions and pricing tactics, Cross-merchandising and shelf placement, in-store signage, product bundling, physical loyalty programs, timing-based offers, local partnerships, and physical customer experience.

Here are the relevant product profiles:

{context}

Start your response immediately below. Do not include any additional headers or explanations.
"""


    elif model_choice == "Gemini API":
        return f"""
You are a seasoned Business Intelligence Strategist specializing in retail product performance. 
You are working with sales and segmentation data from 5 physical Grand Frais stores located in Lyon.

A user has asked:
\"{user_query}\"

Context:
- The product data is from 5 physical Grand Frais stores located in Lyon.
- These are **offline retail stores** (not online, no e-commerce or delivery).
- Grand Frais is part of Prosol, a French retail company focused on fresh produce, meats, cheeses, and seafood.
- All recommendations must apply strictly to **in-store (offline)** strategies.
- Do NOT suggest anything involving websites, email campaigns, apps, or social media marketing.
- Focus instead on in-store promotions, signage, product bundling, layout optimization, local partnerships, and physical customer experience.

Using the product profiles below, generate a concise, executive-level strategy.
Only include:
- Introduction (1-2 sentences)
- 2-3 actionable recommendations
- A brief conclusion

Do not include this prompt or mention the context.
Begin your answer with: ### Final Answer

Here are the relevant product profiles
{context}

Start your response immediately below. Do not include any additional headers or explanations.
"""
    

def enhance_query(user_query):
    system_prompt = """
You are a smart assistant in a business intelligence app that helps analyze product sales and segmentation data for a French retail company.

Business Context:
- Insights are derived exclusively from Grand Frais store data across five cities: Vaulx-en-Velin, Bron, Meyzieu, Décines, and Mions.
- These are physical retail stores. There is no online, delivery, or e-commerce data involved.
- Products include a variety of fresh and packaged items — examples include produce, meat, dairy, seafood, pantry goods, and beverages.
- All insights and strategies must relate to in-store actions (e.g., pricing, promotions, shelf placement, bundling, retention).

Instructions:
1. If the user’s query is unrelated to product performance, customer retention, pricing, segmentation, or strategic business insights, respond with:
   status: unrelated  
   rephrased_query: null

2. If the query is vague but relevant, rewrite it into a **clear and specific business question** about product strategy.
   - You may mention sales, churn, bundling, segmentation, pricing, loyalty, or customer behavior.
   - Do **not** include phrases like "food products" or "offline stores."
   - Do **not** mention specific cities (Vaulx-en-Velin, Bron, Meyzieu, Décines, or Mions) unless the user explicitly mentions a location.
   - The tone should remain analytical, strategic, and business-relevant.

3. If the query is already clear and strategic, return it with only minor refinements to align with the business context (e.g., sales, retention, in-store promotions).

Format your response exactly as:
status: [valid / unclear / unrelated]  
rephrased_query: [query or null]

Examples:

User: churn?  
Response:  
status: unclear  
rephrased_query: Which products across Grand Frais stores show high churn and low repeat purchase rates?

User: What’s selling poorly?  
Response:  
status: unclear  
rephrased_query: Which products are underperforming in sales across Grand Frais stores and may require strategic in-store adjustments?

User: What do people buy repeatedly?  
Response:  
status: unclear  
rephrased_query: Which products have high repeat purchase rates across Grand Frais stores, indicating strong customer retention?

User: Which products need better customer retention strategies?  
Response:  
status: valid  
rephrased_query: Which products need improved in-store strategies to increase customer retention across Grand Frais stores?

User: any bundling ideas?  
Response:  
status: unclear  
rephrased_query: What product bundling opportunities could help improve cross-selling and customer value across Grand Frais stores?

User: Is anything overpriced?  
Response:  
status: unclear  
rephrased_query: Which products may be priced too high compared to competitors, potentially reducing sales at Grand Frais stores?

User: promotions not working?  
Response:  
status: unclear  
rephrased_query: Which products have active in-store promotions but still show weak sales performance across Grand Frais stores?

User: What’s not working in Décines?  
Response:  
status: unclear  
rephrased_query: Which products at the Décines Grand Frais store are underperforming and may require location-specific strategy improvements?

User: tell me a joke  
Response:  
status: unrelated  
rephrased_query: null

User: loyalty?  
Response:  
status: unclear  
rephrased_query: Which products could benefit from being included in in-store loyalty strategies across Grand Frais stores?

User: price drop effect?  
Response:  
status: unclear  
rephrased_query: How did recent in-store price changes impact product performance across Grand Frais stores?

User: top 3 slowest products?  
Response:  
status: unclear  
rephrased_query: Which three products have the slowest sales velocity across Grand Frais stores?

User: Which products have improved lately?  
Response:  
status: valid  
rephrased_query: Which products have shown recent improvements in sales and performance across Grand Frais stores?

User: any city-specific problem?  
Response:  
status: unclear  
rephrased_query: Are there specific cities among Vaulx-en-Velin, Bron, Meyzieu, Décines, and Mions where product performance is significantly lower?
"""
    prompt = f"{system_prompt}\n\nUser Query: {user_query}"
    output = run_gemini_api(prompt).strip()

    result = {"status": None, "rephrased_query": None}
    try:
        for line in output.splitlines():
            if line.lower().startswith("status:"):
                result["status"] = line.split(":", 1)[1].strip().lower()
            elif line.lower().startswith("rephrased_query:"):
                value = line.split(":", 1)[1].strip()
                result["rephrased_query"] = value if value.lower() != "null" else None
    except Exception as e:
        print(f"⚠️ Failed to parse enhanced query: {e}")
        result["status"] = "unrelated"
        result["rephrased_query"] = None

    return result

def extract_clean_output(response, section_markers=None):
    """
    Extract only the relevant portion of the model's output that comes after '### Final Answer'.
    Removes any echoed prompt text, reference strategy, or repeated sections.
    """
    if not response:
        return ""

    # Step 0: Remove any embedded reference strategy blocks
    if "---" in response:
        parts = response.split("---")
        if len(parts) >= 3:
            response = parts[-1].strip()

    # Step 1: Remove everything before ### Final Answer
    start_marker = "### Final Answer"
    start_index = response.find(start_marker)
    if start_index != -1:
        response = response[start_index + len(start_marker):].strip()

    # Step 2: Filter only valid structured strategy content (optional)
    if section_markers:
        response_lines = response.splitlines()
        section_found = False
        final_response = []

        for line in response_lines:
            if any(section.lower() in line.lower() for section in section_markers):
                section_found = True
                final_response.append(line)
            elif section_found:
                final_response.append(line)

        if final_response:
            return "\n".join(final_response).strip()

    # fallback: if markers not found, return everything
    return response.strip()


def replace_strategy_section(full_strategy, suggestion_text):
    """
    Replace only the mentioned strategy section (e.g., Strategy 2) with user's suggestion.
    Assumes user begins their suggestion with the section heading (e.g., "2. Strategy 2: Marketing & Promotion")
    """
    section_match = re.match(r"(\d\.\s+Strategy\s+\d:.*?)\n", suggestion_text)
    if not section_match:
        return full_strategy  # fallback: nothing matched

    section_title = section_match.group(1).strip()

    # Find where this section begins in the original
    pattern = re.compile(rf"{re.escape(section_title)}\n(.*?)(?=\n\d\. Strategy|\n5\. Conclusion|$)", re.DOTALL)
    new_strategy = re.sub(pattern, f"{section_title}\n{suggestion_text.split('\n', 1)[1].strip()}\n", full_strategy)

    return new_strategy


# ===============================
# Load Original Data
# ===============================
@st.cache_data
def load_product_data():
    return joblib.load('./Data/product_insights.joblib')

@st.cache_data
def load_cluster_labels():
    return joblib.load('./Data/label_details.joblib')

product_df = load_product_data()
cluster_labels_df = load_cluster_labels()

# -------------------------------
# Mappings
# -------------------------------
basic_mapping = {
    'Count_of_Purchases': 'Number of Purchases',
    'Total_Quantity': 'Total Quantity',
    'Total_Revenue': 'Total Revenue',
    'Repeat_Purchase_Rate': 'Repeat Purchase Rate',
    'Average_Share_Of_Wallet': 'Average Wallet Share',       
    'Avg_Purchase_Frequency': 'Average Purchase Frequency',  
    'most_used_payment_type': 'Most Used Payment Type',
    'Most_Purchased_Day_Category': 'Most Purchased Days Category',
    'Most_Purchased_Time_Of_Day': 'Most Purchased Time of Day',
}

segmentation_mapping = {
    'sales_performance_segmentation': 'Sales Performance Segmentation',
    'customer_engagement_segmentation': 'Customer Engagement Segmentation',
    'purchase_timing_segmentation': 'Purchase Timing Segmentation',
    'payment_behavior_segmentation': 'Payment Behavior Segmentation',
    'product_segmentation': 'Product Segmentation',
    'geographical_segmentation': 'Geographical Segmentation'
}

numeric_cols = ['Count_of_Purchases', 'Total_Quantity', 'Total_Revenue', 'Repeat_Purchase_Rate','Average_Share_Of_Wallet','Avg_Purchase_Frequency']
text_cols = ['most_used_payment_type', 'Most_Purchased_Day_Category', 'Most_Purchased_Time_Of_Day']
basic_columns = list(basic_mapping.keys())
segmentation_columns = list(segmentation_mapping.keys())


# Get all product names
product_names = product_df.index.astype(str).tolist()

# ===============================
# Download Products by Segmentation Label + Optional Hierarchy Filters
# ===============================

def show_export_products_by_segmentation():
    st.markdown("---")
    st.markdown("#####  Export Products by Segmentation Label")

    # Step 1: Choose Segmentation Type
    selected_segmentation_friendly = st.selectbox(
        "Choose a segmentation type:",
        list(segmentation_mapping.values()),
        key="segmentation_type_dropdown"
    )

    # Step 2: Map back to internal name
    selected_segmentation_col = {
        v: k for k, v in segmentation_mapping.items()
    }[selected_segmentation_friendly]

    # Step 3: Available labels
    available_labels = product_df[selected_segmentation_col].dropna().unique().tolist()
    available_labels.sort()

    selected_label = st.selectbox(
        f"Choose a label from {selected_segmentation_friendly}:",
        available_labels,
        key="segmentation_label_dropdown"
    )

    # Step 4: Optional Description
    label_desc_row = cluster_labels_df[cluster_labels_df["Label"] == selected_label]
    if not label_desc_row.empty:
        desc_text = label_desc_row["Description"].values[0]
        st.info(f"📝 **Description:** {desc_text}")

    # Step 5: Filter by Segmentation
    filtered_products = product_df[product_df[selected_segmentation_col] == selected_label].copy()

    # Step 6: Optional Hierarchy Filtering
    hierarchy_columns = [col for col in product_df.columns if col.startswith("Hierarchy_")]
    with st.expander("🔍 Refine by Product Hierarchy (Optional Filters)", expanded=False):
        hierarchy_filters = {}
        for col in hierarchy_columns:
            options = sorted(product_df[col].dropna().unique().tolist())
            selected_option = st.selectbox(
                f"{col.replace('_', ' ')}:",
                ["(No filter)"] + options,
                key=f"hierarchy_filter_{col}"
            )
            if selected_option != "(No filter)":
                hierarchy_filters[col] = selected_option

    # Step 7: Apply Filters
    for col, val in hierarchy_filters.items():
        filtered_products = filtered_products[filtered_products[col] == val]

    # Step 8: Summary + Export
    st.success(f"{len(filtered_products)} products found matching the selected filters.")

    if not filtered_products.empty:
        csv_data = filtered_products.reset_index().to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download",
            data=csv_data,
            file_name=f"filtered_products_{selected_segmentation_col}_{selected_label}.csv",
            mime="text/csv"
        )


# ===============================
# Product Selection & Details & Strategy Builder Function
# ===============================


def product_strategy_builder():
    st.markdown("---")
    # Initialize product strategy session state variables
    product_state_defaults = {
        "strategy_requested": False,
        "clean_output": None,
        "product_feedback_row": None, # Stores dict of the selected product's RAG row
        "current_product_context": None, # Stores the raw context (rag_input)
        "feedback_choice": "👍",
        "corrected_text": "",
        "feedback_ready": False,
        "context_for_feedback": None, # To store context *actually sent* to LLM (potentially augmented)
        "model_choice_product": "On-Premise", # Default model selection
        "reference_strategy_text": None, # Stores retrieved reference strategy text
        "reference_similarity_score": None, # Stores score of reference strategy
    }

    # Initialize state if not present
    for key, default in product_state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Handle case where product_names list might be empty initially
    if "selected_product" not in st.session_state:
        st.session_state.selected_product = product_names[0] if product_names else None

    # ========================================
    # Product Selection UI
    # ========================================
    st.markdown("<h5>Search Products</h5>", unsafe_allow_html=True)

    def search_function(search_term: str):
        if not product_names: return []
        return [name for name in product_names if search_term.lower() in name.lower()]

    selected_from_search = st_searchbox(
        search_function,
        key="product_searchbox",
        placeholder="Start typing a product name...",
    )

    # Update state if searchbox selects a product
    if selected_from_search and selected_from_search != st.session_state.selected_product:
        st.session_state.selected_product = selected_from_search
        # Reset generation/feedback state when product changes
        st.session_state.strategy_requested = False
        st.session_state.feedback_ready = False
        st.session_state.clean_output = None
        st.session_state.reference_strategy_text = None
        st.session_state.reference_similarity_score = None
        st.rerun()

    st.markdown("##### Browse from Full Product List")
    if not product_names:
        st.warning("No products available to display.")
        return

    # Determine index for selectbox, handle potential errors
    try:
        current_selection = st.session_state.selected_product
        if current_selection not in product_names:
            current_selection = product_names[0]
            st.session_state.selected_product = current_selection
        current_product_index = product_names.index(current_selection)
    except (ValueError, IndexError):
        current_product_index = 0
        if product_names:
             st.session_state.selected_product = product_names[0]
        else:
             st.session_state.selected_product = None


    selected_from_dropdown = st.selectbox(
        "Select a product:",
        product_names,
        index=current_product_index,
        key="product_dropdown_sync"
    )

    # Update state if dropdown selection changes
    if selected_from_dropdown != st.session_state.selected_product:
        st.session_state.selected_product = selected_from_dropdown
        st.session_state.strategy_requested = False
        st.session_state.feedback_ready = False
        st.session_state.clean_output = None
        st.session_state.reference_strategy_text = None
        st.session_state.reference_similarity_score = None
        st.rerun()


    selected_product = st.session_state.selected_product

    # ========================================
    # Display Product Details
    # ========================================
    if selected_product and selected_product in product_df.index:
        try:
            product_details = product_df.loc[selected_product]
            st.markdown(f"<h5>Details for: {selected_product}</h5>", unsafe_allow_html=True)

            # --- Display Metrics ---
            st.markdown("<h6>Key Product Metrics</h6>", unsafe_allow_html=True)
            if numeric_cols and len(numeric_cols) > 0:
                 metric_cols = st.columns(len(numeric_cols))
                 for i, col in enumerate(numeric_cols):
                     friendly_name = basic_mapping.get(col, col)
                     val = product_details.get(col, 'N/A')
                     if isinstance(val, (int, float)):
                         if "rate" in col.lower() or "share" in col.lower(): val_str = f"{val:.2%}"
                         elif isinstance(val, float): val_str = f"{val:.3f}" if abs(val) < 1000 and val != 0 else f"{val:,.0f}"
                         else: val_str = f"{val:,}"
                     else: val_str = str(val)
                     metric_cols[i].metric(friendly_name, val_str)
            else: st.caption("No numeric metrics configured.")

            # --- Display Additional Details ---
            st.markdown("<h6>Additional Details</h6>", unsafe_allow_html=True)
            if text_cols and len(text_cols) > 0:
                 detail_cols = st.columns(len(text_cols))
                 for i, col in enumerate(text_cols):
                     friendly_name = basic_mapping.get(col, col)
                     val = str(product_details.get(col, 'N/A'))
                     detail_cols[i].markdown(f"**{friendly_name}:** {val}")
            else: st.caption("No text details configured.")

            # --- Display Segmentation Overview ---
            st.markdown("<h6>Segmentation Overview</h6>", unsafe_allow_html=True)
            if segmentation_columns and len(segmentation_columns) > 0:
                 for i in range(0, len(segmentation_columns), 3):
                     row_cols = st.columns(min(3, len(segmentation_columns) - i))
                     for j, seg_col in enumerate(segmentation_columns[i:i+3]):
                         product_cluster = product_details.get(seg_col, 'N/A')
                         friendly_seg = segmentation_mapping.get(seg_col, seg_col)
                         with row_cols[j]:
                             st.markdown(f"**{friendly_seg}**")
                             st.markdown(f"{product_cluster}")
            else: st.caption("No segmentation overview configured.")

            # --- Display Segmentation Descriptions (in Expander) ---
            if segmentation_columns and not cluster_labels_df.empty:
                 with st.expander("View Segmentation Descriptions"):
                     displayed_any_desc = False
                     for seg_col in segmentation_columns:
                         product_cluster = product_details.get(seg_col)
                         if product_cluster and not pd.isna(product_cluster):
                             desc_row = cluster_labels_df[cluster_labels_df['Label'] == product_cluster]
                             if not desc_row.empty:
                                 description_text = desc_row['Description'].values[0]
                                 st.markdown(f"**{product_cluster}:** {description_text}")
                                 displayed_any_desc = True
                     if not displayed_any_desc:
                         st.caption("No descriptions available for this product's segmentation labels.")

        except KeyError as e:
            st.error(f"Data Key Error retrieving details for '{selected_product}': {e}")
        except Exception as e:
            st.error(f"Unexpected error displaying product details: {e}")
            import traceback
            traceback.print_exc()

    # ========================================
    # Strategy Generation Trigger
    # ========================================
    st.markdown("---")
    if st.button("Generate Strategy for This Product", key="generate_product_strategy_button"):
        if selected_product and selected_product in rag_df["Product_Name"].values:
            st.session_state.strategy_requested = True
            try:
                row_series = rag_df[rag_df["Product_Name"] == selected_product].iloc[0]
                st.session_state.current_product_context = row_series["rag_input"]
                st.session_state.product_feedback_row = row_series.to_dict()
                # Reset states for new attempt
                st.session_state.feedback_ready = False
                st.session_state.clean_output = None
                st.session_state.reference_strategy_text = None
                st.session_state.reference_similarity_score = None
            except Exception as e:
                 st.error(f"Error preparing for generation: {e}")
                 st.session_state.strategy_requested = False
        else:
            st.error(f"Cannot generate strategy: RAG data missing for '{selected_product}'.")
            st.session_state.strategy_requested = False

    # ========================================
    # Strategy Generation Execution
    # ========================================
    if st.session_state.get("strategy_requested"):

        st.markdown("##### Generate Strategy")
        model_choice = st.selectbox(
            "Choose model:", ["On-Premise", "Gemini API"],
            key="product_model_choice_select",
            index=["On-Premise", "Gemini API"].index(st.session_state.get("model_choice_product", "On-Premise"))
        )
        st.session_state.model_choice_product = model_choice

        if st.button("Confirm and Generate", key="confirm_generate_product_button"):
            with st.spinner("Thinking... Please wait..."):
                context = st.session_state.get("current_product_context")
                row_dict = st.session_state.get("product_feedback_row")

                if not context or not row_dict:
                    st.error("Error: Product context missing. Select product and 'Generate' again.")
                    st.session_state.strategy_requested = False
                    st.stop()

                # --- Embed product context ---
                product_vector = None
                if context and embed_model:
                    try: product_vector = embed_text(context)
                    except Exception as e: st.warning(f"Embedding failed: {e}")

                # --- Search for similar past PRODUCT strategies ---
                similar_entries = []
                st.session_state.reference_strategy_text = None # Clear before search
                st.session_state.reference_similarity_score = None
                if product_vector is not None and product_index is not None:
                    try:
                        similar_entries = search_similar_strategy(
                            query_text_or_vector=product_vector,
                            product_faiss_index=product_index,
                            general_faiss_index=general_index,
                            strategy_type="product",
                            top_k=1, is_vector=True, similarity_threshold=0.7
                        )
                    except Exception as e: st.warning(f"Similarity search error: {e}")

                # --- Augment Context if similar strategy found ---
                augmented_context = context
                if similar_entries:
                    best_match = similar_entries[0]
                    retrieved_past_strategy = best_match.get("corrected_strategy")
                    similarity_score = best_match.get('_similarity_score')
                    if retrieved_past_strategy and isinstance(retrieved_past_strategy, str):
                        # Store reference info in session state for later display
                        st.session_state.reference_strategy_text = retrieved_past_strategy
                        st.session_state.reference_similarity_score = similarity_score
                        # Append reference to the context being sent TO THE LLM
                        augmented_context += (
                            "\n\n---\n📌 **Reference Past Strategy (for inspiration only, do not repeat verbatim):**\n"
                            f"{retrieved_past_strategy.strip()}\n---"
                        )
                        print(f"Using reference product strategy (Score: {similarity_score:.3f})")

                # Store the final context sent to LLM
                st.session_state.context_for_feedback = augmented_context

                # --- Prepare Prompt and Call LLM ---
                try:
                    user_prompt = get_product_prompt(model_choice, augmented_context)
                    # System prompt details instructions for the LLM
                    system_prompt = """
    You are a seasoned Business Intelligence Officer and Product Strategist.

    Your role is to write a **clean, well-structured business strategy** for a retail product, using the data provided.

    **Tone & Style:**
    - Sharp, confident, and executive-level.
    - Use bullet points only if needed for clarity.
    - Avoid over-explaining. Assume the reader understands retail KPIs.
    
    Format your response into exactly 5 sections:
    1. **Introduction** (1-2 lines setting context or objective)
    2. **Strategy 1: Cross-Selling or Synergies**  
    3. **Strategy 2: Marketing & Promotion**  
    4. **Strategy 3: Additional Growth Opportunities**  
    5. **Conclusion** (1-2 lines to close with clarity or next steps)

    Rules:
    - The strategy must be tailored only to the product mentioned. Avoid discussing store-wide or generic strategies.
    - Do NOT summarize the input data.
    - Do NOT describe the context or your reasoning. Go straight to the recommendations.
    - Do NOT summarize the data or echo headings.
    - Do NOT include any other sections like Objectives or Key Insights.
    - Do NOT mention this prompt or that you're an assistant.
    - Only include what belongs in the 5 sections — nothing more.
    - ❗ Do NOT repeat the question, product profiles, or context. Go directly to the strategy starting with '### Final Answer'.

    Your response must begin with:
    ### Final Answer
    """
                    response = None
                    if model_choice == "On-Premise":
                         if model and tokenizer: response = run_llm(system_prompt, user_prompt, model, tokenizer, 0.6, 0.85, 700, 1.2)
                         else: st.error("On-Premise model/tokenizer not loaded.")
                    elif model_choice == "Gemini API": response = run_gemini_api(user_prompt)

                    # --- Process LLM Response ---
                    if response:
                        section_markers = ["Introduction", "Strategy 1", "Strategy 2", "Strategy 3", "Conclusion"]
                        clean_output = extract_clean_output(response, section_markers=section_markers)
                        st.session_state.clean_output = clean_output
                        st.session_state.feedback_ready = True
                        st.session_state.feedback_choice = "👍" # Reset defaults
                        st.session_state.corrected_text = ""
                        st.rerun() # RERUN to display results and feedback form
                    else:
                        st.error("Strategy generation failed: Model returned no response.")
                        st.session_state.feedback_ready = False
                except Exception as e:
                    st.error(f"Error during generation/processing: {e}")
                    import traceback; traceback.print_exc()
                    st.session_state.feedback_ready = False

    # ========================================
    # Feedback Section Display & Handling
    # ========================================
    if st.session_state.get("feedback_ready"):
        clean_output = st.session_state.get("clean_output")
        row_dict = st.session_state.get("product_feedback_row")

        if not clean_output or row_dict is None:
            st.warning("⚠️ Strategy output/data missing. Cannot display feedback.")
        else:
            st.markdown("---")

            # --- Display Reference Strategy Expander (if one was used) ---
            ref_strategy = st.session_state.get("reference_strategy_text")
            ref_score = st.session_state.get("reference_similarity_score")
            if ref_strategy and ref_score is not None:
                 with st.expander("📎 Reference Strategy Used (Similar Product)", expanded=False):
                    st.markdown("This past strategy was used as reference during generation:")
                    score_display = f"{ref_score:.3f}" if isinstance(ref_score, (float, int)) else str(ref_score)
                    st.markdown(f"> _Similarity Score: {score_display}_")
                    st.markdown("---")
                    st.markdown(ref_strategy)
                 st.markdown("---") # Separator

            # --- Display Generated Strategy ---
            st.markdown("#### 📦 Product-Specific Strategy")
            st.markdown(clean_output)

            # --- Display Feedback Form ---
            st.markdown("#### 📣 Feedback on This Strategy")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.session_state.feedback_choice = st.radio(
                    "Helpful?", ["👍", "👎"], 
                    index=["👍", "👎"].index(st.session_state.get("feedback_choice", "👍")),
                    key="product_feedback_choice_radio"
                )
            with col2:
                st.session_state.corrected_text = st.text_area(
                    "🔧 Improvements / Reason",
                    value=st.session_state.get("corrected_text", ""), height=150,
                    key="product_corrected_text_area",
                    help=(
                    "*   👍: Leave blank OR suggest edits (e.g., '2. Strategy 2:...').\n"
                    "*   👎: Leave blank OR optionally explain why.\n"
                    "*   💾 Click 'Save Feedback' to save."
                )
                )

            # --- Save Feedback Button and Logic ---
            if st.button("💾 Save Feedback", key="save_product_feedback_button"):
                user_suggestion = st.session_state.corrected_text.strip()
                feedback_choice = st.session_state.feedback_choice
                final_strategy_to_save, manual_edit, negative_feedback_reason = None, False, None
                original_product_context = row_dict.get("rag_input", "")

                # --- Determine what to save ---
                if feedback_choice == "👍":
                    if user_suggestion:
                        try:
                           modified_strategy = replace_strategy_section(clean_output, user_suggestion)
                           if modified_strategy != clean_output:
                               final_strategy_to_save, manual_edit = modified_strategy, True
                               confirmation_message = "✅ Suggestion applied & feedback saved."
                           else:
                               final_strategy_to_save, manual_edit = clean_output, False
                               confirmation_message = "✅ Feedback saved. Suggestion format error? Original retained."
                               st.warning("Suggestion format error? Check 'X. Section:...'. Original saved.")
                        except Exception as e:
                             st.error(f"Error applying suggestion: {e}")
                             final_strategy_to_save = clean_output
                             confirmation_message = "✅ Feedback saved (Error applying suggestion)."
                    else: # Thumbs up, no suggestion
                        final_strategy_to_save, manual_edit = clean_output, False
                        confirmation_message = "✅ Positive feedback saved."
                elif feedback_choice == "👎":
                    negative_feedback_reason = user_suggestion if user_suggestion else "User marked as not useful (no reason)."
                    confirmation_message = "📝 Negative feedback saved."
                else: st.error("Invalid feedback choice."); st.stop()

                # --- Prepare Feedback Entry ---
                product_vector_for_saving = None
                if original_product_context and embed_model:
                    try: product_vector_for_saving = embed_text(original_product_context).tolist()
                    except Exception as e: st.warning(f"Embedding failed for feedback: {e}")

                if product_vector_for_saving is None:
                    st.error("Vector creation failed. Feedback not saved.")
                else:
                    feedback_entry = {
                        "strategy_type": "product",
                        "product_name": row_dict.get("Product_Name", "Unknown"),
                        "product_vector": product_vector_for_saving,
                        "original_input": original_product_context,
                        "original_strategy": clean_output,
                        "corrected_strategy": final_strategy_to_save,
                        "manual_edit": manual_edit,
                        "feedback": feedback_choice,
                        "negative_feedback_reason": negative_feedback_reason,
                        "timestamp": str(datetime.datetime.now())
                    }
                    # --- Save to Memory ---
                    try:
                        save_to_memory(feedback_entry, strategy_type="product")
                        st.success(confirmation_message)
                        # --- Reset relevant state after saving ---
                        st.session_state.feedback_ready = False
                        st.session_state.strategy_requested = False
                        st.session_state.clean_output = None
                        st.session_state.corrected_text = ""
                        st.session_state.feedback_choice = "👍"
                        st.session_state.reference_strategy_text = None # Clear reference
                        st.session_state.reference_similarity_score = None
                    except Exception as e:
                        st.error(f"CRITICAL Error saving feedback: {e}")

# ===============================
# General Strategy via RAG (Strategy Explorer) Function
# ===============================

def strategy_explorer():
    st.markdown("---")
    st.markdown("<h5>Ask Your Question</h5>", unsafe_allow_html=True)

    # Initialize session state variables for this specific explorer flow
    explorer_state_defaults = {
        "explorer_query_validated": False, # Flag: query analyzed by enhance_query
        "explorer_query_confirmed": False, # Flag: query ready for RAG/generation
        "explorer_enhancement_result": None, # Stores enhance_query output dict
        "explorer_user_query_text": "",      # User's raw input
        "explorer_confirmed_query_text": "", # Query text used for RAG/LLM
        "explorer_rag_context": None,      # RAG results text
        "explorer_retrieved_products": None, # List of dicts of products from RAG
        "explorer_strategy_generated": False,# Flag: LLM response received
        "explorer_general_strategy_output": None, # Clean LLM output
        "explorer_context_for_feedback": None,    # Full context (RAG + reference) sent to LLM
        "explorer_general_feedback_ready": False, # Flag: Show feedback form
        "explorer_general_feedback_choice": "👍", # Stored feedback choice
        "explorer_general_corrected_text": "",    # Stored feedback text
        "explorer_model_choice_general": "On-Premise", # Default model
        "explorer_reference_strategy_text": None, # Store retrieved general reference strategy
        "explorer_reference_similarity_score": None, # Store score for reference
        "explorer_input_key": "explorer_user_query_input_1" # Help reset input box
    }
    # Initialize state if not present
    for key, default in explorer_state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # ========================================
    # Query Input
    # ========================================
    # Use a specific key and update state on change
    st.session_state.explorer_user_query_text = st.text_input(
        "Example: Which products need better customer retention strategies?",
        value=st.session_state.explorer_user_query_text,
        key=st.session_state.explorer_input_key,
    )

    # ========================================
    # Query Processing Trigger & Logic
    # ========================================
    if st.button("Analyze Query", key="analyze_explorer_query_button"):
        query_to_process = st.session_state.explorer_user_query_text.strip()
        if not query_to_process:
            st.warning("Please enter a query.")
        else:
            # Reset states before processing new query
            st.session_state.explorer_query_validated = False
            st.session_state.explorer_query_confirmed = False
            st.session_state.explorer_strategy_generated = False
            st.session_state.explorer_general_feedback_ready = False
            st.session_state.explorer_enhancement_result = None
            st.session_state.explorer_retrieved_products = None # Clear previous RAG results
            st.session_state.explorer_reference_strategy_text = None # Clear previous reference
            st.session_state.explorer_reference_similarity_score = None

            with st.spinner("Analyzing query..."):
                try:
                    enhancement_result = enhance_query(query_to_process)
                    st.session_state.explorer_enhancement_result = enhancement_result

                    status = enhancement_result.get("status")
                    rephrased = enhancement_result.get("rephrased_query")

                    # --- BRANCHING LOGIC BASED ON QUERY STATUS ---
                    if status == "valid" and rephrased:
                        # VALID: Immediately confirm and proceed to generation flow
                        st.success(f"✅ Query confirmed: \"{rephrased}\" Proceeding to generate strategy...")
                        st.session_state.explorer_confirmed_query_text = rephrased
                        st.session_state.explorer_query_confirmed = True
                        st.session_state.explorer_query_validated = False # Skip validation display block
                        st.rerun() # Rerun to trigger the RAG/Generation block
                    elif status == "unclear" and rephrased:
                        # UNCLEAR: Needs user confirmation - set flag to show confirmation UI
                        st.session_state.explorer_query_validated = True
                        # Rerun needed to display the confirmation options
                        st.rerun()
                    elif status == "unrelated":
                        st.error("🚫 Query unrelated to product strategy. Please ask a relevant business question.")
                        st.session_state.explorer_query_validated = False # Reset
                    else: # Error or unexpected status
                        st.error("Query analysis failed or returned an unexpected result. Please try rephrasing.")
                        st.session_state.explorer_query_validated = False # Reset

                except Exception as e:
                    st.error(f"Error during query analysis: {e}")
                    st.session_state.explorer_query_validated = False

    # ========================================
    # Display Query Confirmation UI (Only for UNCLEAR queries)
    # ========================================
    if st.session_state.get("explorer_query_validated") and not st.session_state.get("explorer_query_confirmed"):
        result = st.session_state.get("explorer_enhancement_result")
        # This block should only be reached if status was 'unclear'
        if result and result.get("status") == "unclear" and result.get("rephrased_query"):
            rephrased = result.get("rephrased_query")
            st.markdown("##### ✨ Suggested Query Refinement")
            st.info(f"💬 **{rephrased}**")
            st.caption("Use this refined version for better results?")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Yes, use refined query", key="confirm_refined_explorer_query"):
                    st.session_state.explorer_confirmed_query_text = rephrased
                    st.session_state.explorer_query_confirmed = True
                    st.session_state.explorer_query_validated = False # Done with validation step
                    st.rerun() # Rerun to proceed with confirmed query
            with col2:
                if st.button("❌ No, use my original query", key="use_original_explorer_query"):
                    original_query = st.session_state.explorer_user_query_text.strip()
                    if not original_query:
                         st.warning("Original query was empty.")
                         st.session_state.explorer_query_validated = False # Force re-entry/analysis
                         st.rerun()
                    else:
                        st.session_state.explorer_confirmed_query_text = original_query
                        st.session_state.explorer_query_confirmed = True
                        st.session_state.explorer_query_validated = False 
                        st.rerun()
       

    # ========================================
    # RAG, Generation Preparation, and Trigger
    # ========================================
    # This block runs if query is confirmed BUT strategy hasn't been generated yet
    
    if st.session_state.get("explorer_query_confirmed") and not st.session_state.get("explorer_strategy_generated"):
        confirmed_query = st.session_state.explorer_confirmed_query_text
        # --- Perform RAG using the product index ---
        rag_context = ""
        # Reset retrieved products before RAG
        st.session_state.explorer_retrieved_products = None
        if embed_model and product_index and rag_df is not None and not rag_df.empty:
            with st.spinner("Retrieving relevant product data..."):
                try:
                    query_vec = embed_text(confirmed_query)
                    if query_vec is not None:
                         # Search product index for contexts relevant to the general query
                         scores, indices = product_index.search(query_vec.reshape(1, -1), k=3)
                         valid_indices = [idx for i, idx in enumerate(indices[0]) if idx != -1 and scores[0][i] >= SIMILARITY_THRESHOLD_RAG]

                         if valid_indices:
                             top_chunks = rag_df.iloc[valid_indices]["rag_input"].tolist()
                             rag_context = "\n\n---\n".join(top_chunks) # Join contexts
                             retrieved_products_data = rag_df.iloc[valid_indices][["Product_Name", "rag_input"]].to_dict('records')
                             # Store retrieved products in session state for later display
                             st.session_state.explorer_retrieved_products = retrieved_products_data
                         else:
                             st.warning("⚠️ No highly relevant product profiles found via RAG for context.")
                    else:
                         st.warning("⚠️ Could not embed query for RAG search.")
                except Exception as e:
                    st.error(f"Error during RAG retrieval: {e}")
                    rag_context = "" # Ensure context is empty on error

        # Store RAG context (even if empty) for the LLM call
        st.session_state.explorer_rag_context = rag_context
        # --- Search for similar past GENERAL strategies ---
        # Reset previous reference state
        st.session_state.explorer_reference_strategy_text = None
        st.session_state.explorer_reference_similarity_score = None
        augmented_context_for_llm = rag_context # Start with RAG context

        if confirmed_query and embed_model and general_index:
             with st.spinner("Searching for similar past queries..."):
                 try:
                     query_vector_for_general_search = embed_text(confirmed_query)
                     if query_vector_for_general_search is not None:
                         similar_general_entries = search_similar_strategy(
                            query_vector_for_general_search,
                            product_index, general_index, # Pass both indexes
                            strategy_type="general",      # Specify general search
                            top_k=1, is_vector=True,
                            similarity_threshold=SIMILARITY_THRESHOLD_REFERENCE
                         )

                         if similar_general_entries:
                             best_general_match = similar_general_entries[0]
                             retrieved_past_general_strategy = best_general_match.get("corrected_strategy")
                             similarity_score = best_general_match.get('_similarity_score')

                             if retrieved_past_general_strategy and isinstance(retrieved_past_general_strategy, str):
                                 # Store reference info in session state for later display
                                 st.session_state.explorer_reference_strategy_text = retrieved_past_general_strategy
                                 st.session_state.explorer_reference_similarity_score = similarity_score
                                 # Augment the context FOR THE LLM
                                 augmented_context_for_llm += (
                                     "\n\n---\n📌 **Reference Past General Strategy (for inspiration only, do not repeat verbatim):**\n"
                                     f"{retrieved_past_general_strategy.strip()}\n---"
                                 )
                                 print(f"Using reference general strategy (Score: {similarity_score:.3f})") # Debug
                 except Exception as e:
                     st.warning(f"Error during similarity search for general strategies: {e}")

        # Store the final context (RAG + potential reference) used for generation
        st.session_state.explorer_context_for_feedback = augmented_context_for_llm

        # --- Model Selection & Generate Button ---
        st.markdown("###### Select Model and Generate")
        model_choice = st.selectbox(
            "Choose model:", ["On-Premise", "Gemini API"],
            key="explorer_model_choice_select",
            index=["On-Premise", "Gemini API"].index(st.session_state.get("explorer_model_choice_general", "On-Premise"))
        )
        st.session_state.explorer_model_choice_general = model_choice

        # Button to trigger the actual LLM call
        if st.button("Generate Strategy", key="generate_general_strategy_button"):
            with st.spinner("Thinking... Please wait..."):
                current_user_query = st.session_state.explorer_confirmed_query_text
                final_context_for_llm = st.session_state.explorer_context_for_feedback # Use stored context

                try:
                    # Get the prompt using the confirmed query and augmented context
                    user_prompt = get_query_prompt(model_choice, current_user_query, final_context_for_llm)
                    # Define system prompt (can be constant)
                    system_prompt = """
    You are a Business Intelligence Strategist for offline retail, focused on product performance and segmentation across physical Grand Frais stores.

    Your task:
    - Understand the user's query and adapt your strategic response to match its intent.
    - Use the provided product profiles and segmentation data to support your response.
    - Focus **strictly on offline, in-store strategies** (e.g., shelf placement, physical promotions, pricing, physical loyalty programs, cross-merchandising, in-store signage).

    Format your response in 3 parts:
    1. **Introduction** – Summarize the issue or opportunity based on the user’s query.
    2. **Actionable Recommendations** – Give 2–3 concrete suggestions relevant to the query type (e.g., retention, pricing, bundling).
    3. **Conclusion** – Wrap up with a brief note on next steps or strategy alignment.

    Constraints:
    - DO NOT mention this prompt or the product profiles directly.
    - DO NOT include irrelevant digital strategies (e.g., emails, websites, apps, or social media).
    - DO NOT repeat the user’s question.

    Your output must begin with “### Final Answer”, then include these headers exactly:
    - Introduction
    - Actionable Recommendations
    - Conclusion
    """
                    response = None
                    # Call LLM
                    if model_choice == "On-Premise":
                        if model and tokenizer: response = run_llm(system_prompt, user_prompt, model, tokenizer, 0.6, 0.85, 700, 1.2)
                        else: st.error("On-Premise model/tokenizer not loaded.")
                    elif model_choice == "Gemini API": response = run_gemini_api(user_prompt)

                    # --- Process Response ---
                    if response:
                        section_markers = ["Introduction", "Actionable Recommendations", "Conclusion"]
                        clean_output = extract_clean_output(response, section_markers=section_markers)
                        st.session_state.explorer_general_strategy_output = clean_output
                        st.session_state.explorer_strategy_generated = True # Mark as generated
                        st.session_state.explorer_general_feedback_ready = True # Enable feedback
                        st.session_state.explorer_general_feedback_choice = "👍" # Reset feedback defaults
                        st.session_state.explorer_general_corrected_text = ""
                        st.rerun() # RERUN to display strategy and feedback form
                    else:
                        st.error("Strategy generation failed: Model returned no response.")
                        st.session_state.explorer_strategy_generated = False # Ensure flag is off

                except Exception as e:
                    st.error(f"Error during general strategy generation: {e}")
                    import traceback; traceback.print_exc()
                    st.session_state.explorer_strategy_generated = False

    # ========================================
    # Feedback Section Display & Handling
    # ========================================
    # This block runs only AFTER generation is complete and successful
    if st.session_state.get("explorer_strategy_generated") and st.session_state.get("explorer_general_feedback_ready"):
        clean_output = st.session_state.get("explorer_general_strategy_output")
        context_used = st.session_state.get("explorer_context_for_feedback") # Context used for generation
        query = st.session_state.get("explorer_confirmed_query_text") # The query used
        retrieved_products_list = st.session_state.get("explorer_retrieved_products") # Stored RAG products
        ref_strategy = st.session_state.get("explorer_reference_strategy_text") # Stored reference strategy
        ref_score = st.session_state.get("explorer_reference_similarity_score") # Stored reference score


        if not clean_output or context_used is None or not query:
            st.warning("⚠️ Strategy output/context missing. Cannot display feedback form.")
        else:
            st.markdown("---")

            # --- Display Retrieved Products Expander (if available) ---
            if retrieved_products_list:
                 st.markdown("###### Product Profiles Used for Context")
                 with st.expander("View Retrieved Product Profiles"):
                    for i, product in enumerate(retrieved_products_list):
                        st.markdown(f"**{i+1}. Product:** {product.get('Product_Name', 'N/A')}")
                        st.caption(f"Profile Snippet: {product.get('rag_input', '')[:300]}...")
                        if i < len(retrieved_products_list) - 1: st.markdown("---") # Separator inside expander
                 st.markdown("---") # Separator after expander


            # --- Display Reference Strategy Expander (if available) ---
            if ref_strategy and ref_score is not None:
                 with st.expander("📎 Reference Strategy Used (Similar Past Query)", expanded=False):
                    st.markdown("This past strategy was used as reference during generation:")
                    score_display = f"{ref_score:.3f}" if isinstance(ref_score, (float, int)) else str(ref_score)
                    st.markdown(f"> _Similarity Score: {score_display}_")
                    st.markdown("---")
                    st.markdown(ref_strategy)
                 st.markdown("---") # Separator after expander

            # --- Display Generated Strategy ---
            st.markdown("#### Strategy Suggestion")
            st.markdown(clean_output)

            # --- Display Feedback Form ---
            st.markdown("#### 📣 Feedback")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.session_state.explorer_general_feedback_choice = st.radio(
                    "Helpful?", ["👍", "👎"],
                    index=["👍", "👎"].index(st.session_state.get("explorer_general_feedback_choice", "👍")),
                    key="explorer_general_feedback_choice_radio"
                )
            with col2:
                st.session_state.explorer_general_corrected_text = st.text_area(
                    "🖊️ Better Version / Reason", # Shortened label
                    value=st.session_state.get("explorer_general_corrected_text", ""), height=150,
                    key="explorer_general_corrected_text_area",
                    help=(
                    "*   👍: Leave blank OR provide improved strategy.\n"
                    "*   👎: Leave blank OR optionally explain why.\n"
                    "*   💾 Click 'Save General Feedback' to save."
                )
                )

            # --- Save Feedback Button and Logic ---
            if st.button("💾 Save General Feedback", key="save_general_feedback_button"):
                user_suggestion = st.session_state.explorer_general_corrected_text.strip()
                feedback_choice = st.session_state.explorer_general_feedback_choice
                final_strategy_to_save, manual_edit, negative_feedback_reason = None, False, None

                # --- Determine what to save ---
                if feedback_choice == "👍":
                    if user_suggestion: # User provided improved version
                        final_strategy_to_save, manual_edit = user_suggestion, True
                        confirmation_message = "✅ Suggested strategy saved as corrected version."
                    else: # User liked original
                        final_strategy_to_save, manual_edit = clean_output, False
                        confirmation_message = "✅ Positive feedback saved."
                elif feedback_choice == "👎":
                    negative_feedback_reason = user_suggestion if user_suggestion else "User marked as not useful (no reason)."
                    confirmation_message = "📝 Negative feedback saved."
                else: st.error("Invalid feedback choice."); st.stop()

                # --- Prepare Feedback Entry ---
                # Embed the CONTEXT USED for generation (RAG + Reference)
                vector_for_saving = None
                if context_used and embed_model:
                    try: vector_for_saving = embed_text(context_used).tolist()
                    except Exception as e: st.warning(f"Embedding context failed for feedback: {e}")

                if vector_for_saving is None:
                     st.error("Vector creation failed. Feedback not saved.")
                else:
                    feedback_entry = {
                        "strategy_type": "general",
                        "original_query": query, # The confirmed query used
                        "retrieved_vector": vector_for_saving, # Vector of context used for generation
                        "retrieved_context": context_used, # Text of context used
                        "original_strategy": clean_output, # Raw LLM output
                        "corrected_strategy": final_strategy_to_save, # Improved or original if 👍
                        "manual_edit": manual_edit,
                        "feedback": feedback_choice,
                        "negative_feedback_reason": negative_feedback_reason,
                        "timestamp": str(datetime.datetime.now())
                    }
                    # --- Save to Memory ---
                    try:
                        save_to_memory(feedback_entry, strategy_type="general")
                        st.success(confirmation_message)
                        # --- Crucial: Clear cache for index rebuild ---
                        st.cache_resource.clear()
                        print("Resource cache cleared to update general FAISS index.")
                        # --- Reset Explorer State after saving ---
                        st.session_state.explorer_general_feedback_ready = False
                        st.session_state.explorer_strategy_generated = False
                        st.session_state.explorer_query_confirmed = False
                        st.session_state.explorer_query_validated = False
                        st.session_state.explorer_user_query_text = "" # Clear input text
                        st.session_state.explorer_general_strategy_output = None
                        st.session_state.explorer_general_corrected_text = ""
                        st.session_state.explorer_general_feedback_choice = "👍"
                        st.session_state.explorer_retrieved_products = None # Clear RAG display
                        st.session_state.explorer_reference_strategy_text = None # Clear reference display
                        st.session_state.explorer_reference_similarity_score = None
                        # Update input key to force blank input widget
                        st.session_state.explorer_input_key = f"explorer_user_query_input_{str(time.time())}"
                    except Exception as e:
                        st.error(f"Error saving general feedback: {e}")

# ===============================
# maintenance tools here
# =============================== 
#               
if selected == "Maintenance":
    show_maintenance_tools()
 
# ===============================
# Feedback Viewer Component
# ===============================

if selected  == "Strategy Feedback Vault": 
    show_feedback_viewer()  
    
# ===============================
# Export Products by Segmentation
# ===============================

if selected == "Segment & Export Products":
    show_export_products_by_segmentation()

# ===============================
# Prodcuts insights and strategy builder
# ===============================
    
if selected == "Product Strategy Builder":
    product_strategy_builder()

# ===============================
# strategy explorer
# ===============================

if selected  == "Strategy Explorer":
    strategy_explorer()