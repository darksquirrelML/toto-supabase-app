#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###SUPABASE###

# Toto Prediction Dashboard — Dark Theme, 7-number support, click-to-refresh

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import os
from datetime import datetime
import time

from supabase import create_client


# Load from Streamlit Secrets
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]

# Create Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)



###############################################################
# ---------------------------
# Session state init (TOP)
# ---------------------------
if "lstm_model" not in st.session_state:
    st.session_state.lstm_model = None

if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
#################################################################

# --- Optional ML libs ---
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ---------- Page config ----------
st.set_page_config(page_title="Toto Prediction — Dark Pro", layout="wide", page_icon="🎰")

# ---------- Top header with optional logo ----------
col1, col2 = st.columns([0.12, 0.88])
with col1:
    logo_path = "dark_squirrel.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
    else:
        svg = """
        <svg width='90' height='90' viewBox='0 0 64 64' xmlns='http://www.w3.org/2000/svg'>
          <circle cx='32' cy='32' r='30' fill='#0d0d0d'/>
          <path d='M20 36 C18 28, 26 18, 36 20 C44 22, 46 32, 40 36 C34 40, 30 44, 22 42 Z' fill='#00e5ff' />
        </svg>
        """
        st.markdown(svg, unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='color:#00e5ff; margin-bottom:0'>🎰 Toto Prediction — Dark Pro</h1>", unsafe_allow_html=True)
    st.caption("Trends • Hot/Cold • Machine Learning (LSTM) — Click-to-refresh & PDF export")

st.write("---")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    num_draws = st.slider("Number of past draws to analyze", min_value=20, max_value=2000, value=300, step=10)
    show_animated = st.checkbox("Show animated trend", True)
    train_epochs = st.number_input("LSTM train epochs", min_value=1, max_value=600, value=200)
    batch_size = st.number_input("Batch size", min_value=8, max_value=512, value=64)
    train_ratio = st.slider("Train ratio", 0.5, 0.95, 0.85)
    window_size = st.number_input("LSTM window size", min_value=1, max_value=30, value=10)
    seed = st.number_input("Random seed", value=42)
    st.markdown("---")
    st.markdown("<small style='color:#8d99a6'>Tip: Put your historical CSV file named <code>toto_history_all.csv</code> in this app folder. The CSV must contain columns: Draw No, Draw Date, Winning No (comma-separated 6 numbers), Additional No.</small>", unsafe_allow_html=True)
    st.markdown("<small style='color:#8d99a6'>Click the 'Refresh Data' button to reload without affecting ML predictions.</small>", unsafe_allow_html=True)
#################################################################################
    mc_samples = st.number_input(
        "MC prediction passes",
        min_value=1,
        max_value=200,
        value=20
    )

# --- Disclaimer at the top ---
st.markdown(
    """
    <div style="background-color:#fff3cd; padding:15px; border-left:6px solid #ffc107; border-radius:5px">
    ⚠️ **Disclaimer:** This TOTO app is for **fun and entertainment only**.  
    The predictions or information provided are **not guaranteed**.  
    We are **not responsible** for any decisions or outcomes based on this app.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- Load / refresh data ----------
@st.cache_data(ttl=300)
def load_data_from_supabase(limit=None):
    """
    Load TOTO historical data from Supabase.
    Returns DataFrame in the SAME format as your CSV version.
    """

    query = supabase.table("toto_results") \
        .select("draw_no, draw_date, winning_no, additional_no") \
        .order("draw_no", desc=True)  # newest → oldest 
# .order("draw_no", desc=False)  # oldest → newest
    

    if limit:
        query = query.limit(limit)

    response = query.execute()

    if not response.data:
        return pd.DataFrame()

    df = pd.DataFrame(response.data)

    
    # restore oldest → newest for ML
    df = df.sort_values("draw_no").reset_index(drop=True)

    
    
    # Match your original column names
    df.rename(columns={
        "draw_no": "Draw No",
        "draw_date": "Draw Date",
        "winning_no": "Winning No",
        "additional_no": "Additional No"
    }, inplace=True)

    # Same processing as before
    df['Winning'] = df['Winning No'].apply(
        lambda x: [int(i) for i in str(x).split(',')]
    )

    df['Additional No'] = df['Additional No'].apply(
        lambda x: int(x) if pd.notna(x) else None
    )

    return df.reset_index(drop=True)

# --- Load into session_state ---
if 'df' not in st.session_state or st.button("Refresh Data"):
    st.session_state['df'] = load_data_from_supabase()

df = st.session_state['df']

if df.empty:
    st.error("No data found in Supabase. Please run the scraper first.")
    st.stop()

# --- Safe to use df now ---
st.write(f"**Dataset:** {len(df)} draws loaded — last draw date: {df.iloc[-1]['Draw Date']}")



# ---------- Helper functions ----------
@st.cache_data
def frequency_dataframe(df, last_n):
    recent = df.tail(last_n).reset_index(drop=True)
    all_nums = []
    frames = []

    for i, row in recent.iterrows():
        nums = row['Winning'][:]
        if row['Additional No'] is not None:
            nums.append(row['Additional No'])
        all_nums.extend(nums)
        # Animation frames
        subset = []
        for j in range(i+1):
            s_row = recent.iloc[j]
            s_nums = s_row['Winning'][:]
            if s_row['Additional No'] is not None:
                s_nums.append(s_row['Additional No'])
            subset.extend(s_nums)
        series = pd.Series(subset).value_counts().sort_index()
        for num, cnt in series.items():
            frames.append({'Frame': i, 'Number': num, 'Count': int(cnt)})

    freq = pd.Series(all_nums).value_counts().sort_index()
    freq_df = pd.DataFrame({'Number': freq.index, 'Count': freq.values})
    frames_df = pd.DataFrame(frames)
    return freq_df, frames_df

# ---------- Tabs ----------
# tabs = st.tabs(["Trends", "Hot / Cold Numbers", "Machine Learning Prediction"])
tab = st.radio(
    "Navigation",
    ["Trends", "Hot / Cold Numbers", "Machine Learning Prediction"],
    horizontal=True,
    key="main_tab"
)



# ---------------- Trends Tab ----------------
# with tabs[0]:
if tab == "Trends":
############################################################################################    
    

    st.header("TOTO Trends & Statistics")

    # ============================================
    # LATEST TOTO RESULT (supports 'Winning No' format)
    # ============================================

    latest_date = df["Draw Date"].iloc[-1]

    # Split Winning No row into list of numbers
    latest_main = df["Winning No"].iloc[-1].split(",")

    # Convert strings to int
    latest_main = [int(x) for x in latest_main]

    latest_add = int(df["Additional No"].iloc[-1])

    st.subheader("Latest TOTO Result")
    st.write(f"**Draw Date:** {latest_date}")
    st.write(f"**Main Numbers:** {latest_main}")
    st.write(f"**Additional Number:** {latest_add}")
    st.markdown("---")


    # (Your existing trends code continues here…)

 ##############################################################################################   
    
    
#     st.header("Trends")
    freq_df, frames_df = frequency_dataframe(df, num_draws)
    colA, colB = st.columns([2,1])
    with colA:
        st.subheader("Number Frequency (bar chart)")
        fig = px.bar(freq_df, x='Number', y='Count', title=f'Frequency (last {num_draws} draws)', template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        if show_animated and not frames_df.empty:
            st.subheader("Animated frequency over draws")
            anim_fig = px.bar(frames_df, x='Number', y='Count', color='Number', animation_frame='Frame', range_y=[0, frames_df['Count'].max()+1], template='plotly_dark')
            st.plotly_chart(anim_fig, use_container_width=True)

    with colB:
        st.subheader("Top / Bottom")
        top6 = list(freq_df.sort_values('Count', ascending=False).head(6)['Number'])
        bottom6 = list(freq_df.sort_values('Count', ascending=True).head(6)['Number'])
        st.metric("Top 6 (most frequent)", ', '.join(map(str, top6)))
        st.metric("Bottom 6 (least frequent)", ', '.join(map(str, bottom6)))

# ---------------- Hot / Cold Tab ----------------
# with tabs[1]:
elif tab == "Hot / Cold Numbers":
    st.header("Hot and Cold Numbers")
    window = st.slider("Hot/Cold window (recent draws)", min_value=10, max_value=500, value=100, step=10)
    recent_df = df.tail(window)
    all_recent = []
    for _, r in recent_df.iterrows():
        all_recent.extend(r['Winning'])
        if r['Additional No'] is not None:
            all_recent.append(r['Additional No'])
    recent_counts = pd.Series(all_recent).value_counts()
    overall_counts = pd.Series([n for row in df['Winning'] for n in row] + [row['Additional No'] for _, row in df.iterrows() if row['Additional No'] is not None]).value_counts()
    ratio = (recent_counts / overall_counts).fillna(0).sort_values(ascending=False)
    hot = ratio.head(6)
    cold = overall_counts.sort_values().head(6)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hot Numbers (recent surge)")
        st.write(hot)
    with col2:
        st.subheader("Cold Numbers (least overall)")
        st.write(cold)

# ---------------- Machine Learning Tab ----------------


# with tabs[2]:
elif tab == "Machine Learning Prediction":
    st.header("Machine Learning Prediction — LSTM (7 numbers)")
    st.markdown("Train an LSTM on 7-number draws (6 main + 1 additional). Progress bar + ETA shown during training and prediction.")

    st.write(f"Epochs: {train_epochs}")
    st.write(f"Batch size: {batch_size}")
    st.write(f"Window size: {window_size}")
    st.write(f"Train ratio: {train_ratio}")

    window_in = window_size
    epochs_in = train_epochs
    batch_in = batch_size
    train_ratio_in = train_ratio
    seed_in = seed

# elif tab == "Machine Learning Prediction":
#     st.header("Machine Learning Prediction — LSTM (7 numbers)")
#     st.markdown("Train an LSTM on 7-number draws (6 main + 1 additional). Progress bar + ETA shown during training and prediction.")

    if not TF_AVAILABLE:
        st.warning("TensorFlow is not installed. Install it (`pip install tensorflow`) to use LSTM features.")
    else:
        # --- ML Controls ---
        # cols = st.columns([1,1,1,1])
        # with cols[0]:
        #     batch_in = st.number_input("Batch size", min_value=8, max_value=512, value=batch_size, key="ml_batch")
        # with cols[1]:
        #     train_ratio_in = st.slider("Train ratio", 0.5, 0.95, value=train_ratio, key="ml_train_ratio")
        # with cols[2]:
        #     epochs_in = st.number_input("Epochs", min_value=1, max_value=600, value=train_epochs, key="ml_epochs")
        # with cols[3]:
        #     window_in = st.number_input("Window size", min_value=1, max_value=30, value=window_size, key="ml_window")
        # seed_in = st.number_input("Random seed", value=seed, key="ml_seed")
        # mc_samples = st.number_input("MC passes for prediction", min_value=1, max_value=200, value=20, key="ml_mc")

        # --- Helper function: convert draws to multihot encoding ---
        def draws_to_multihot(df_in):
            X = []
            for _, row in df_in.iterrows():
                v = np.zeros(49, dtype=np.float32)
                for n in row['Winning']:
                    v[int(n)-1] = 1.0
                if 'Additional No' in row and pd.notna(row['Additional No']):
                    v[int(row['Additional No'])-1] = 1.0
                X.append(v)
            return np.array(X)

        data_X = draws_to_multihot(df)
        if len(data_X) <= window_in:
            st.error("Not enough draws to build sequences. Reduce window size or add more data.")
        else:
            sequences = []
            targets = []
            for i in range(len(data_X) - window_in):
                sequences.append(data_X[i:i+window_in])
                targets.append(data_X[i+window_in])
            sequences = np.array(sequences)
            targets = np.array(targets)
            st.write(f"Prepared {len(sequences)} sequences (window={window_in}) — features=49")

            model_path = "lstm_model.h5"

            def build_model(window_size, features=49):
                tf.random.set_seed(int(seed_in))
                model = keras.Sequential([
                    layers.Input(shape=(window_size, features)),
                    layers.LSTM(128, return_sequences=False),
                    layers.Dropout(0.2),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(features, activation='sigmoid')
                ])
                model.compile(optimizer='adam', loss='binary_crossentropy')
                return model

            model = None
            if os.path.exists(model_path):
                st.info("Saved model found on disk")
                if st.button("Load saved model"):
                    with st.spinner("Loading model..."):
                        model = keras.models.load_model(model_path)
                    st.success("Model loaded")

            # ---- Train LSTM ----
            if st.button("Train LSTM model", key="train_lstm"):

                total_epochs = int(train_epochs)
                batchsz = int(batch_size)
                val_split = 1.0 - float(train_ratio)
                window_in = int(window_size)
                seed_in = int(seed)

                tf.random.set_seed(int(seed_in))

                # model = build_model(window_in)
                # total_epochs = int(epochs_in)
                # batchsz = int(batch_in)
                # val_split = 1.0 - float(train_ratio_in)

                # CREATE MODEL ONLY IF NOT EXISTS
                if st.session_state.lstm_model is None:
                    st.session_state.lstm_model = build_model(window_in)

                model = st.session_state.lstm_model  # ✅ always use session model

                progress = st.progress(0)
                status = st.empty()
                loss_chart = st.empty()
                start_time = time.time()
                history_logs = {"loss": [], "val_loss": []}

                for ep in range(total_epochs):
                    ep_start = time.time()
                    hist = model.fit(sequences, targets,
                                     epochs=1,
                                     batch_size=batchsz,
                                     validation_split=val_split,
                                     verbose=0)
                    loss = hist.history.get('loss', [None])[-1]
                    val_loss = hist.history.get('val_loss', [None])[-1]
                    history_logs['loss'].append(loss)
                    history_logs['val_loss'].append(val_loss)

                    percent = int(((ep + 1) / total_epochs) * 100)
                    progress.progress(percent)
                    elapsed = time.time() - start_time
                    avg_per_epoch = elapsed / (ep + 1)
                    remaining = avg_per_epoch * (total_epochs - (ep + 1))
                    status.text(f"Epoch {ep+1}/{total_epochs} — loss: {loss:.4f} val_loss: {val_loss:.4f} — ETA: {remaining:.1f}s")

                    loss_chart.line_chart({
                        "loss": history_logs['loss'],
                        "val_loss": history_logs['val_loss']
                    })

                model.save(model_path)
                progress.progress(100)
                status.text(f"Training completed in {time.time() - start_time:.1f}s — model saved")
                st.success("Model training finished and saved")
########################################################################################################################
            # ---------------- Prediction with MC forward passes + progress ----------------
            st.markdown("### Predict next draw (LSTM)")

            # Select how many recent draws to prioritize
            last_n_for_priority = st.number_input(
                "Number of recent draws to prioritize",
                min_value=5,
                max_value=50,
                value=10,
                step=1,
                key="ml_recent_draws"
            )

            if st.button("Predict next draw (LSTM)"):
                if model is None:
                    if os.path.exists(model_path):
                        with st.spinner("Loading saved model..."):
                            model = keras.models.load_model(model_path)
                    else:
                        st.error("No trained model available. Train or load a model first.")
                        model = None

                if model is not None:
                    # Prepare input sequence
                    last_seq = data_X[-window_in:]
                    inp = last_seq.reshape((1, window_in, 49)).astype(np.float32)

                    # Monte-Carlo passes for averaged probabilities
                    mc = int(mc_samples)
                    probs_accum = np.zeros(49, dtype=np.float64)

                    prog = st.progress(0)
                    status_p = st.empty()
                    t0 = time.time()

                    for i in range(mc):
                        pred = model(inp, training=True).numpy().reshape(-1)
                        probs_accum += pred
                        prog.progress(int(((i+1)/mc)*100))
                        elapsed = time.time() - t0
                        avg = elapsed / (i+1)
                        remaining = avg * (mc - (i+1))
                        status_p.text(f"MC pass {i+1}/{mc} — ETA: {remaining:.1f}s")

                    avg_probs = probs_accum / mc

                    # Last N draws numbers for prioritization
                    recent_draws = df.tail(last_n_for_priority)
                    recent_numbers = set()
                    for _, row in recent_draws.iterrows():
                        recent_numbers.update(row['Winning'])
                        if row['Additional No'] is not None:
                            recent_numbers.add(row['Additional No'])

                    # Sort all numbers by probability descending
                    all_probs_sorted = [(i+1, avg_probs[i]) for i in range(49)]
                    all_probs_sorted.sort(key=lambda x: x[1], reverse=True)

                    # Pick top 7 numbers, prioritize numbers in recent draws
                    top7_probs = []
                    for num, prob in all_probs_sorted:
                        if num in recent_numbers:
                            top7_probs.append((num, prob))
                        if len(top7_probs) == 7:
                            break
                    # Fill to 7 if fewer than 7
                    if len(top7_probs) < 7:
                        for num, prob in all_probs_sorted:
                            if num not in [x[0] for x in top7_probs]:
                                top7_probs.append((num, prob))
                            if len(top7_probs) == 7:
                                break

                    # Table with indication if number is from recent draws
                    table_data = []
                    for num, prob in top7_probs:
                        table_data.append({
                            'Number': num,
                            'Prob': prob,
                            'Recent (last N draws)': 'Yes' if num in recent_numbers else 'No'
                        })

                    top7_idx = [x['Number'] for x in table_data]

                    status_p.text("Prediction done.")
                    prog.progress(100)

                    st.success(f"Predicted numbers (6 main + 1 additional): {top7_idx}")
                    st.table(pd.DataFrame(table_data).set_index('Number'))

          
######################################################################################################################
                    if REPORTLAB_AVAILABLE:
                        try:
                            pdf_bytes = io.BytesIO()
                            c = canvas.Canvas(pdf_bytes, pagesize=letter)
                            text = c.beginText(40, 700)
                            text.setFont("Helvetica", 14)
                            text.textLine("Toto Prediction — LSTM")
                            text.textLine(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                            text.textLine("")
                            text.textLine("Predicted numbers (6 main + additional):")
                            text.textLine(', '.join(map(str, list(top7_idx))))
                            c.drawText(text)
                            c.showPage()
                            c.save()
                            pdf_bytes.seek(0)
                            st.download_button("Download prediction as PDF", data=pdf_bytes, file_name="toto_prediction.pdf", mime='application/pdf')
                        except TypeError as e:
                            st.warning("PDF export temporarily unavailable due to Python/ReportLab version mismatch.")
                    else:
                        st.info("Install reportlab to enable PDF export: pip install reportlab")


                    # ---------------- Last 10 Draws & Comparison ----------------
                    st.markdown("### Last 10 Draws — Reference & Comparison")
#                     last10 = df.tail(10).reset_index(drop=True)
                    last10 = df.tail(last_n_for_priority).reset_index(drop=True)

                    last10_table = []
                    for i, row in last10.iterrows():
                        numbers = row['Winning'][:]
                        if row['Additional No'] is not None:
                            numbers.append(row['Additional No'])
                        last10_table.append({
                            "Draw No": row['Draw No'],
                            "Draw Date": row['Draw Date'],
                            "Numbers": ', '.join(map(str, numbers))
                        })
                    st.table(pd.DataFrame(last10_table))

                    st.markdown("### Comparison of Predicted Numbers with Last 10 Draws")
                    pred_set = set(top7_idx)
                    comparison_table = []
                    for i, row in last10.iterrows():
                        draw_set = set(row['Winning'][:])
                        if row['Additional No'] is not None:
                            draw_set.add(row['Additional No'])
                        match_count = len(pred_set & draw_set)
                        comparison_table.append({
                            "Draw No": row['Draw No'],
                            "Draw Date": row['Draw Date'],
                            "Matching Numbers": match_count
                        })
                    st.table(pd.DataFrame(comparison_table))

pass


