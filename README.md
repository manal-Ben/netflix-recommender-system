# 🎬 Netflix Recommendation System

A content-based movie and TV show recommendation system built with Python and deployed as a Flask web application. Enter any title from the Netflix catalogue and instantly get 10 similar picks, complete with posters fetched live from the OMDb API.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat&logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-TF--IDF-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📖 Description

The system analyses the textual metadata of each title (director, cast, genres, description) to build a **TF-IDF** matrix. Recommendations are ranked by **cosine similarity**, meaning the more textual overlap two titles share, the higher they score. The result is a fast, interpretable, and dependency-light recommender that runs entirely on your local machine.

---

## ✨ Features

- 🔍 **Content-based filtering** — no user history required
- 🧠 **TF-IDF + Cosine Similarity** — lightweight yet effective ML model
- 🖼️ **Live movie posters** — fetched in parallel via the OMDb API
- 🔎 **Case-insensitive search** — partial title matching supported
- ⚡ **Fast inference** — model is built once at startup and cached in memory
- 📱 **Responsive UI** — Netflix-inspired dark interface

---

## 🛠️ Technologies Used

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Web framework | Flask |
| Data processing | Pandas |
| ML / NLP | Scikit-learn (TF-IDF, cosine similarity) |
| Poster API | OMDb API |
| Frontend | HTML5, CSS3 (vanilla) |
| Notebook | Jupyter |

---

## 📁 Project Structure

```
netflix-recommender/
│
├── app.py                          # Flask web application & routes
├── recommendation_system.py        # ML model: TF-IDF + cosine similarity
│
├── templates/
│   └── index.html                  # Jinja2 HTML template
│
├── static/
│   └── style.css                   # Netflix-inspired dark UI
│
├── netflix_titles.csv              # Netflix dataset (source data)
│
└── notebooks/
    └── netflix_recommendation.ipynb  # Exploratory analysis & prototyping
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/netflix-recommender.git
cd netflix-recommender
```

### 2. Create a virtual environment *(recommended)*

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install flask pandas scikit-learn requests
```

> **Note:** No `requirements.txt` yet? Generate one with `pip freeze > requirements.txt` after installing.

---

## ▶️ How to Run

```bash
python app.py
```

Then open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

The model is built automatically on first startup (takes ~5–10 seconds).

---

## 🎯 Example Usage

1. Open the app in your browser.
2. Type a title you enjoy, e.g. **`Stranger Things`**.
3. Click **Get Recommendations**.
4. Browse your 10 personalised picks with posters, genres, and release year.

> Titles are matched case-insensitively. If a title is not found, a clear error message is displayed.

---

## 🔬 Error Analysis

| Error | Cause | Fix |
|---|---|---|
| `File not found` at startup | Dataset CSV is missing or misnamed | Ensure `netflix_titles.csv` is in the project root |
| `Title not found in dataset` | Exact title not in the catalogue | Check spelling or try a different title |
| Poster not showing | OMDb API limit reached or title unknown | Placeholder art is shown automatically |
| `500` on POST when model failed to load | Dataset missing at startup | Fix the dataset path, then restart the server |

---

## 👤 Author

**Your Name**  
[GitHub](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/your-profile)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
## Project Notes
This project demonstrates a simple ML-based recommendation system.
