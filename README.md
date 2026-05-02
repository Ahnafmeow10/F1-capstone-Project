# 🏎️ F1 Race Outcome Predictor
### DATA 4382 Capstone II · Individual Deployment

**Ahnaf Harappa & Tajwar Fahmid** 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://f1-capstone-project-9vmsfzcafpqlx6vda4kjfu.streamlit.app/)

---

## 📁 Folder Structure

```
f1-race-predictor/
├── app_ahnaf.py        # Podium & Winner Predictor (Model B + C)
├── app_tajwar.py       # Points Finish Predictor (Model A)
├── requirements.txt    # Python dependencies
└── data/               # Kaggle F1 CSV files (download separately)
    ├── races.csv
    ├── results.csv
    ├── qualifying.csv
    ├── constructor_standings.csv
    ├── driver_standings.csv
    └── status.csv
```

---

## 📊 Dataset

Download from Kaggle and place all CSV files in a `data/` folder:

👉 [Kaggle — Formula 1 World Championship (Rohan Rao)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)

---

## ▶️ How to Run

**Step 1 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2 — Run Ahnaf's app (Podium & Winner)**
```bash
streamlit run app_ahnaf.py
```

**Step 3 — Run Tajwar's app (Points Finish)**
```bash
streamlit run app_tajwar.py
```

To run both at the same time:
```bash
streamlit run app_ahnaf.py --server.port 8501
streamlit run app_tajwar.py --server.port 8502
```

---

## 📦 Requirements

```
streamlit
scikit-learn
pandas
numpy
plotly
```

---

## 🌐 Live Apps

| App | Description |
|---|---|
| [Ahnaf — Podium & Winner Predictor](https://f1-capstone-project-9vmsfzcafpqlx6vda4kjfu.streamlit.app/) | Models B & C — from top-10 drivers, who podiums and who wins? |
| Tajwar — Points Finish Predictor | Model A — will this driver score points or finish P11+? |

---

*DATA 4382 – Capstone II · Kaggle F1 Dataset (Rohan Rao, 2010–2023)*
