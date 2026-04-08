# AI vs Human Essay Detector

A **web-based essay detector** built with Python and Streamlit that estimates whether a piece of text is AI-generated or human-written.

---

## Description

This app uses a trained AI model to analyze essays or text input and outputs:

- Whether the text is **AI-generated** or **human-written**
- Confidence percentages for both categories

The interface is interactive, browser-based, and fully handled by Streamlit. The app is deployed online using **Replit** and requires no setup for users.

---

## Features

- Web-based interface via Streamlit
- Paste essays or text for analysis
- AI vs Human prediction
- Confidence scores and progress bars
- Simple and interactive design

---

## Live Demo

[Try the app here](YOUR-REPLIT-LINK-HERE)

---

## Built With

- Python
- Streamlit
- Joblib (for model loading)
- Replit (deployment)

---

## How It Works

1. The system loads a pre-trained detector model and vectorizer.
2. Input text is transformed via the vectorizer.
3. Model predicts AI vs Human and outputs probabilities.
4. Streamlit displays results with metrics and progress bars.

⚠️ Note: Model accuracy depends on training data. This is a **learning prototype**.

---

## 💻 How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/kixxy-a11y/essay-detector.git
```
