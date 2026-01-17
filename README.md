# PathWise: AI-Powered Learning Path Generator 🎓

PathWise is an intelligent educational tool designed for MCA (Master of Computer Applications) students. It leverages Machine Learning and Generative AI to identify academic weaknesses and generate personalized 2-week study plans.

## 🚀 Features

-   **Performance Analysis**: Enter your Minor 1, Minor 2, and End Semester marks to get an instant analysis.
-   **Weakness Prediction**: Uses a custom-trained TensorFlow/Keras neural network to predict subjects where you might be struggling.
-   **AI-Generated Study Plans**: Integrates Google's Gemini Pro API to create tailored, week-by-week improvement schedules based on your specific weak areas.
-   **Progress Tracking**: Automatically saves your generated plans to a local database (`user_progress.db`), allowing you to track your learning journey by Roll Number.

## 🛠️ Technology Stack

-   **Frontend**: [Streamlit](https://streamlit.io/)
-   **Machine Learning**: TensorFlow, Keras, Scikit-learn
-   **Generative AI**: Google Gemini API (`gemini-2.5-flash`)
-   **Data Processing**: Pandas, NumPy
-   **Database**: SQLite

## 📂 Project Structure

```text
/
├── app.py              # Main application entry point
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── secrets.toml    # API keys (not shared)
├── models/             # Pre-trained .h5 models and .pkl scalers
├── data/               # Project datasets and SQLite database
├── notebooks/          # Jupyter notebooks for training and EDA
└── assets/             # Project images and resources
```

## ⚙️ Setup & Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/zach-codes932/PathWise.git
    cd PathWise
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Key**
    Create a file named `.streamlit/secrets.toml` in the project root and add your Google Gemini API key:
    ```toml
    GEMINI_API_KEY = "your_api_key_here"
    ```

4.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

## 📊 Models

The project uses two neural network models depending on the available data:
1.  **Minors Only Model**: Predicts weakness based solely on Minor 1 and Minor 2 scores.
2.  **Full Data Model**: Incorporates End Semester scores for a final comprehensive prediction.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and create a pull request for any feature enhancements or bug fixes.