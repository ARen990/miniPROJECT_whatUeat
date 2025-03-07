# WHAT U EAT? | Siamese Neural Network Food Recommendation App

## Description
This project is a web application utilizing a Siamese Neural Network to recommend suitable dishes based on the ingredients users have available. It can also automatically filter out dishes containing ingredients that users are allergic to or do not wish to consume, helping users decide their meals daily.

## Problem &  Solution
In daily life, many individuals struggle with deciding what to eat, often asking friends for suggestions that may still not be satisfactory or compatible with available ingredients. This project addresses this issue by utilizing a Siamese Neural Network to analyze ingredient similarity, recommend dishes that closely match user preferences, and automatically exclude dishes with unwanted or allergenic ingredients.


## Model Architecture
The Siamese Neural Network model comprises:

- Input Layer (shape: [None, 9]) for ingredient data
- Embedding Layer (output: [None, 9, 128])
- LSTM Layer (output: [None, 9, 64])
- Attention Layer (output: [None, 64])
- Lambda Layer (Euclidean distance calculation)
- Dense Layer (output: [None, 1])

![Screenshot 2025-03-07 130034](https://github.com/user-attachments/assets/e40278fe-47ff-462a-acaa-8dc4270ce5a8)

  I tested this model using `...\ML\92_84\siamese_food_recommendation_model.h5`

## Dataset
The dataset used is Thai food menu data from the file `cleaned1_thailand_foods.csv`, containing menu names (in Thai and English) and ingredients for each dish.


## Installation
```bash
git clone https://github.com/ARen990/miniPROJECT_whatUeat.git
pip install -r requirements.txt
```

### Additional Code Information

For further model modification or retraining:
- `Trainmodelfood.ipynb` (Jupyter Notebook for training and adjusting the model)
- `app.py` (Streamlit script for running the web application)

Ensure TensorFlow and Streamlit are installed:

```bash
pip install tensorflow streamlit numpy pandas
```

## Usage
1. Run the web application using the command:
```bash
streamlit run app.py
```
2. Open a browser and go to `http://localhost:8501`
3. Enter available ingredients (comma-separated), and optionally specify ingredients you dislike or are allergic to.
4. Select the number of recommended dishes.
5. Click to get food recommendations.

## Result
I tested this model using `...\ML\92_84\siamese_food_recommendation_model.h5`
![download](https://github.com/user-attachments/assets/b7293a9e-1e42-4699-a4be-33a25c2d0c23)

When tested on the web
![image](https://github.com/user-attachments/assets/7fcaa85f-c41b-4f09-b3d3-fc37b3f2629d)


## 📬 Contact
- **GitHub:** [ARen990](https://github.com/ARen990)
- **Email:** krittimonp28@gmail.com
- **X:** [Aenijin](https://x.com/Aenijin)

