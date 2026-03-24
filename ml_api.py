from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model ONCE
model = joblib.load("model.pkl")

columns = [
"user_interest_dance",
"user_interest_music",
"user_interest_sports",
"user_interest_tech",
"user_interest_cultural",
"event_category_dance",
"event_category_music",
"event_category_sports",
"event_category_tech",
"event_category_cultural",
"location_chennai"
]

@app.route("/")
def home():
    return "ML API Running"

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json

        user_interests = data.get("user_interests", [])
        location = data.get("location", "chennai")

        interest = user_interests[0] if user_interests else "music"

        row = {
            "user_interest": interest,
            "event_category": interest,
            "location": location
        }

        df = pd.DataFrame([row])
        df = pd.get_dummies(df)

        for col in columns:
            if col not in df:
                df[col] = 0

        df = df[columns]

        prediction = model.predict(df)[0]

        categories = {
            0: "sports",
            1: "music",
            2: "dance",
            3: "tech",
            4: "cultural"
        }

        predicted_category = categories.get(prediction, "tech")

        return jsonify({
            "recommended_categories": [predicted_category]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# IMPORTANT FOR RENDER
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)