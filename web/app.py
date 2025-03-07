import streamlit as st
import pickle
import numpy as np
import pandas as pd
from ast import literal_eval
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.utils import register_keras_serializable

# Define Attention Layer
@register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention = Dense(1, activation='tanh')

    def call(self, inputs):
        weights = self.attention(inputs)  # Compute attention scores
        weights = tf.nn.softmax(weights, axis=1)  # Apply softmax
        return tf.reduce_sum(inputs * weights, axis=1)  # Weighted sum

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config  # Ensures serialization works correctly

# Define Euclidean Distance Function
@register_keras_serializable()
def euclidean_distance(vectors):
    x, y = vectors
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

# Register the custom layer and function in `custom_objects`
custom_objects = {'AttentionLayer': AttentionLayer, 'euclidean_distance': euclidean_distance}

model = load_model("D:\MiniProjectML\ML\92_84\siamese_food_recommendation_model.h5", compile=False)

with open("D:\MiniProjectML\ML\92_84\\tokenizer.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)

max_len = model.input_shape[0][1]


# ✅ โหลด dataset จากไฟล์ CSV
def load_data():
    file_path = "D:\MiniProjectML\dataset\cleaned1_thailand_foods.csv"  # ปรับ path ให้ตรงกับไฟล์จริง
    df = pd.read_csv(file_path)
    df["ingredients"] = df["ingredients"].apply(literal_eval)  # แปลง string -> list
    df["ingredient_text"] = df["ingredients"].apply(lambda ing: " ".join(ing))  # แปลงเป็นข้อความ
    return df

df = load_data()

# ✅ โหลด Tokenizer และสร้าง Sequence
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["ingredient_text"])  # Fit tokenizer กับ dataset จริง
sequences = tokenizer.texts_to_sequences(df["ingredient_text"])
max_len = max(map(len, sequences))  # หาความยาวสูงสุดของ sequence
x_data = pad_sequences(sequences, maxlen=max_len)

# ✅ โหลดโมเดลที่เทรนแล้ว
custom_objects = {'AttentionLayer': AttentionLayer, 'euclidean_distance': euclidean_distance}
siamese_model = load_model("D:\MiniProjectML\ML\92_84\siamese_food_recommendation_model.h5", compile=False)

def recommend_foods(available_ingredients, allergic_ingredients=[], top_n=10):
    """
    Uses the trained Siamese model to find the most similar recipes based on available ingredients.
    
    Parameters:
    - available_ingredients (list): List of available ingredients (e.g., ["rice", "chicken"])
    - allergic_ingredients (list): List of allergic ingredients (e.g., ["peanuts"])
    - top_n (int): Number of recommended menus to return.

    Returns:
    - DataFrame containing the top recommended menus with allergic ingredients removed.
    """

    if not available_ingredients:
        return "Please enter at least one available ingredient!!!"

    # ✅ Convert input ingredients into text and tokenize
    input_text = " ".join(available_ingredients)
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_len)

    # ✅ Remove allergic ingredients from all recipes
    modified_df = df.copy()
    modified_df["ingredients"] = modified_df["ingredients"].apply(
        lambda ing: [item for item in ing if item not in allergic_ingredients]
    )
    
    # ✅ Create modified text for tokenization
    modified_df["ingredient_text"] = modified_df["ingredients"].apply(lambda ing: " ".join(ing))
    
    # ✅ Tokenize and pad the modified ingredients
    modified_sequences = tokenizer.texts_to_sequences(modified_df["ingredient_text"])
    modified_sequences = pad_sequences(modified_sequences, maxlen=max_len)

    # ✅ Use Siamese model to compute similarity
    similarities = siamese_model.predict([np.tile(input_seq, (len(modified_sequences), 1)), modified_sequences]).flatten()

    # ✅ Compute importance scores based on matching ingredients
    importance_scores = np.array([
        sum(1 for ing in available_ingredients if ing in row) for row in modified_df["ingredients"]
    ])
    weighted_similarities = similarities + (0.5 * importance_scores / max(importance_scores, default=1))

    # ✅ Select Top-N most similar menus
    top_indices = weighted_similarities.argsort()[-top_n:][::-1]
    recommended_menus = modified_df.iloc[top_indices][["en_name", "th_name", "ingredients"]]

    return recommended_menus

# sample_available_ingredients = ["rice", "chicken"]
# sample_allergic_ingredients = ["peanuts"]

# recommendations = recommend_foods(sample_available_ingredients, sample_allergic_ingredients, top_n=3)

# print("Recommended Menus:")
# print(recommendations)


# Streamlit UI
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFF3E0;  /* พื้นหลังสีครีม */
        color: #D32F2F
    }
    .stMarkdown, .stTitle, .stHeader {
        color: #3E2723;  /* ตัวอักษรสีน้ำตาลเข้ม */
    }
    .stButton > button {
        background-color: #E53935 !important; /* ปุ่มสีแดง */
        color: white !important;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton > button:hover {
        background-color: #D32F2F !important; /* ปุ่มสีเข้มขึ้นเมื่อ hover */
    }
    /* เปลี่ยนสี label ของ text_input ให้เป็นสีดำ */
    div[data-testid="stTextInput"] label {
        color: #000000 !important;
        font-weight: bold;  /* ทำให้ตัวหนา */
    }
    div[data-testid="stSlider"] label {
        color: #000000 !important;
        font-weight: bold;  /* ทำให้ตัวหนา */
    }

    /* เปลี่ยนสีพื้นหลังของช่อง input เป็นสีดำ */
    .stTextInput > div > div > input {
        background-color: #5D4037  !important;  
        color: #ffffff !important;  
        border-radius: 8px !important;  
        padding: 10px !important;
    }

    /* เปลี่ยนสี placeholder ให้เป็นเทาอ่อน */
    .stTextInput > div > div > input::placeholder {
        color: #bbbbbb !important;
        font-style: italic;
    }

    /* เปลี่ยนสีขอบเวลาคลิกเลือก input */
    .stTextInput > div > div > input:focus {
        border: 2px solid #FF5722 !important; /* เส้นขอบสีส้มเมื่อเลือก */
        outline: none !important;
    }
      /* เปลี่ยนสีพื้นหลังและข้อความของ st.warning() */
    div[data-testid="stAlert"] {
        background-color: #D32F2F !important;  /* สีส้มอ่อน */
        border-radius: 8px !important;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        border: 2px solid #333; /* :white_check_mark: กรอบนอกของตาราง */
        background-color: #FAF3E0; /* :white_check_mark: สีพื้นหลังของตาราง */
    }

    th, td {
        border: 1px solid #FFFFFF; /* :white_check_mark: เส้นแบ่งระหว่างเซลล์ */
        padding: 10px;
        text-align: left;
    }

    th {
        background-color: #FF7043; /* :white_check_mark: สีพื้นหลังของหัวข้อคอลัมน์ */
        color: white;
        font-weight: bold;
    }

    tr:nth-child(even) {
        background-color: #FFF3E0; /* :white_check_mark: สีพื้นหลังแถวคู่ */
    }

    tr:hover {
        background-color: #FFD54F; /* :white_check_mark: เปลี่ยนสีเมื่อ hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)


page = st.sidebar.radio("PAGE", ["Recommended Dishes", "All menu","About"])

if page == "Recommended Dishes":

    st.title("🍽️ Recommended Dishes")

    available_ingredients = st.text_input("What do you want to eat? (comma-separated):")
    allergic_ingredients = st.text_input("What you don't want to eat (comma-separated):")

    top_n = st.slider("Number of recommendations:", 1, 10, 3)


    if st.button("Recommend Food"):
        available_ingredients = [ing.strip().lower() for ing in available_ingredients.split(",") if ing.strip()]
        allergic_ingredients = [ing.strip().lower() for ing in allergic_ingredients.split(",") if ing.strip()]

        recommendations = recommend_foods(available_ingredients, allergic_ingredients, top_n)

        if isinstance(recommendations, str):
            st.warning(recommendations)
        else:
            st.write("Recommended Menus")
            # st.dataframe(recommendations, use_container_width=True, hide_index=True)
            st.markdown(
                recommendations.to_html(index=False, escape=False),
                unsafe_allow_html=True
            )

elif page == "All menu":
    st.title("📜 All menu")
    # st.write("ตารางรวมเมนูทั้งหมดจากฐานข้อมูล")

    # ✅ แสดงผลเป็น HTML Table พร้อมเส้นกรอบ
    st.markdown(
        """
        <style>
        table {
            width: 100%;
            border-collapse: collapse;
            border: 2px solid #333; /* กรอบตาราง */
            background-color: #FAF3E0; /* สีพื้นหลัง */
        }
        th, td {
            border: 1px solid #FFFFFF;
            padding: 10px;
            text-align: left;
        }
        th {
            background-color: #FF7043;
            color: white;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #FFF3E0;
        }
        tr:hover {
            background-color: #FFD54F;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ✅ แสดงตารางทั้งหมดของเมนู
    st.markdown(df.to_html(index=False, escape=False), unsafe_allow_html=True)

elif page == "About":
    st.title("📌 ABOUT THIS PROJECT")
    
    st.markdown(
        """
        ## Welcome to the Food Recommendation Web App! 🍽️
        This application helps you discover food recipes based on your available ingredients.  
        By leveraging **Machine Learning (Siamese Neural Network)**, it calculates recipe similarity  
        and recommends the best matching dishes.

        ---

        ## 🛠️ Technologies Used
        - **Python, TensorFlow, Streamlit, Pandas**
        - **Siamese Neural Network** for intelligent recommendations
        - **Dataset**: Thai food recipes from `cleaned1_thailand_foods.csv`

        ---

        ## 📍 How It Works
        1. **Enter your available ingredients** → The app suggests matching recipes  
        2. **Specify unwanted or allergic ingredients** → The system filters out those recipes  
        3. **Choose the number of recommendations** → Get personalized food suggestions  

        ---

        ## Contact & Resources
        - **GitHub**: [Project Repository](https://github.com/ARen990/miniPROJECT_whatUeat)
        - **Email**: krittimonp28@gmail.com
        - **Youtube**: [Your Profile](https://linkedin.com/in/your-profile)
        """
    )

    # # ✅ Optionally, add an image (like a logo or a sample recommendation)
    # st.image("https://example.com/your-image.jpg", use_column_width=True)
