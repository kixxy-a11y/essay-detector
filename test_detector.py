import joblib
import os

# 1. Load the "saved brains"
if not os.path.exists('models/detector_model.pkl'):
    print("❌ Error: Train the model first using 'python train.py'")
    exit()

model = joblib.load('models/detector_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

print("\n--- 🤖 AI vs. ✍️ Human Detector ---")
print("Type 'quit' to exit.")

while True:
    user_text = input("\nPaste text to analyze: ")
    
    if user_text.lower() == 'quit':
        break
    
    if not user_text.strip():
        continue

    # 2. Transform the input text
    features = vectorizer.transform([user_text])
    
    # 3. Make a prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    # 4. Show the result
    if prediction == 1:
        print(f"🚩 Result: AI GENERATED (Confidence: {probability[1]:.2%})")
    else:
        print(f"✅ Result: HUMAN WRITTEN (Confidence: {probability[0]:.2%})")