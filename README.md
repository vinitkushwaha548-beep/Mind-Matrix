# 🧠 Mind Matrix

**Measure. Understand. Heal.**

Mind Matrix is a low-cost, AI-assisted biofeedback system for counselling sessions. A small sensor on the client's fingers picks up changes in skin conductance (GSR / EDA), a laptop dashboard shows how their stress and arousal move in real time, and a simple machine learning model turns the signal into an easy-to-read recovery score.

It is a **research and counselling-support prototype**, not a medical device. It is meant to give counsellors one more objective data point alongside their own judgement, never to replace it.

---

## Why this exists

Most counselling today depends on what a client chooses to say and how well they can put their feelings into words. But stress often shows up in the body before we notice it ourselves: sweaty palms, a racing pulse, a sudden spike when a difficult topic comes up.

Tools that measure these signals do exist, but they are usually expensive and built for hospitals and research labs. A small clinic, an independent counsellor or a college wellness centre in India can rarely afford them.

Mind Matrix asks a simple question: *can we build a stripped-down, affordable version that does one job well, which is showing session-by-session arousal and recovery?*

---

## What it does

- **Live arousal graph.** Watch skin conductance (in μS) move during the session.
- **Stress-Arousal Index.** A 0–100 score with a plain label such as "Calm" or "Moderate".
- **Recovery score (TRR).** A Total Recovery/Response score with a trend arrow showing whether the client is improving, steady or declining.
- **Session timer and live status.** Know at a glance whether data is streaming.
- **Session-to-session comparison.** Track how arousal and recovery change across visits.
- **Auto-generated PDF summary.** One short report per session, ready to attach to your notes.
- **Patient ID system.** Sessions are linked to an ID rather than personal details.

---

## How it works

The system follows a seven-step pipeline:

1. **Capture:** a finger GSR/EDA sensor measures sweat-gland activity.
2. **Clean and calibrate:** the signal is smoothed with a moving average, outliers from movement are removed, and a short calibration at the start of each session sets that person's baseline.
3. **Extract features:** 16 physiological features are computed, such as average arousal, peak arousal and recovery trend.
4. **Stream:** an Arduino/ESP32 sends the processed data to a laptop in real time.
5. **Visualise:** a Python/PyQt dashboard shows the live graph and status.
6. **Analyse:** three models work together to estimate the TRR score:
   - **Random Forest** captures non-linear patterns between features.
   - **Ridge Regression** gives a stable, easy-to-interpret baseline.
   - **EWMA** (exponentially weighted moving average) smooths the trend so short spikes don't dominate.
7. **Report:** a concise PDF summary is generated at the end of the session.

---

## What's inside the box

| Part | Role | Approx. cost (₹) |
|---|---|---|
| Arduino / ESP32 microcontroller | Reads and streams sensor data | 700 |
| Finger GSR/EDA sensor | Worn on the index and middle fingers | 600 |
| Cable / sensor interface module | Connects sensor to board | 300 |
| Enclosure and connectors | Portable housing | 500 |
| Wiring and assembly | Miscellaneous | 600 |
| **Total** | | **≈ 2,700** |

Software: Python, PyQt for the dashboard, scikit-learn for the models, and a lightweight PDF library for reports.

---

## Privacy

Session data stays **on the counsellor's own laptop**. Nothing is sent to an external server. Each session is tied to a Patient ID instead of personal details, and generated reports should be handled like any other confidential session notes.

---

## Who it's for

Counsellors and psychologists, college wellness centres, small clinics, neuropsychiatry support settings, medical colleges, research labs, and students who want hands-on access to affordable physiological data.

---

## Honest limitations

We would rather tell you now than have you find out later:

- It has **not** been clinically validated and must not be used to diagnose or make treatment decisions.
- The scoring model was tuned on a **small volunteer dataset**, not a large clinical sample.
- Signal quality drops with heavy movement, poor sensor contact, or extreme heat and humidity.
- It needs a laptop to run the dashboard; there is no standalone or mobile version yet.
- Long-term storage, multi-user accounts and cloud sync are not built yet.

---

## Roadmap

1. ✅ Working prototype (hardware and dashboard)
2. Feedback from counsellors, psychologists and doctors
3. Patent / IP filing
4. Improved hardware: enclosure, signal quality, portability
5. Pilot testing in clinics, wellness centres and colleges
6. Validation and a possible ISO / CDSCO compliance pathway
7. Market launch through institutional and direct channels

---

## 👨‍💻 Author

**Vinit Kushwaha**
B.Tech, Artificial Intelligence and Machine Learning (AIML)
United College of Engineering and Research, Prayagraj

📧 [vinitkushwaha548@gmail.com](mailto:vinitkushwaha548@gmail.com)
🐙 [github.com/vinitkushwaha548-beep](https://github.com/vinitkushwaha548-beep)

---

## 🙏 Acknowledgements

Mind Matrix is based on the original concept, design and project report of **MannPulseAI**, created by **Aditya Kumar Singh** and **Aditya Sharma** (Team Debuggers Squad) at United College of Engineering and Research, Prayagraj, under the guidance of **Dr. Vijay Kumar Dwivedi**. It is adapted and shared here with their permission. Full credit for the original idea and design goes to them.

---

## ⚠️ Disclaimer

Mind Matrix is a research and counselling-support prototype. It is **not** a diagnostic medical device and should always be used as a complement to, never a substitute for, a trained professional's judgement.

---

*If this project interests you, a ⭐ on the repo is always appreciated.*
