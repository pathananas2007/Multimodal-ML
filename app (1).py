# ==============================================
# Advanced Multimodal AI System (Upgraded Version)
# ==============================================

import gradio as gr
from transformers import pipeline
import torch

print("Initializing Advanced AI models...")

device = 0 if torch.cuda.is_available() else -1

# -------------------------------
# 🔥 Stronger Models
# -------------------------------

text_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    device=device
)

image_pipeline = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224",
    device=device
)


audio_pipeline = pipeline(
    "automatic-speech-recognition",
    model="facebook/wav2vec2-base-960h",
    device=device
)

print("All models loaded successfully.")


# ==================================================
# 🚀 Advanced Fusion Logic with Weighted Scoring
# ==================================================

def multimodal_analyze(text, image, audio):
    print("Inference started")

    text_label = None
    text_conf = 0
    image_label = None
    image_conf = 0
    transcription = None
    audio_label = None
    audio_conf = 0

    text_result_display = "No text provided."
    image_result_display = "No image provided."
    audio_result_display = "No audio provided."

    # ---------------- TEXT ----------------
    if text and text.strip():
        res = text_pipeline(text)[0]
        raw_label = res["label"]
        text_conf = round(res["score"] * 100, 2)

        label_map = {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "POSITIVE"
        }

        text_label = label_map.get(raw_label, raw_label)

        text_result_display = f"""
## 📝 Text Sentiment
**Prediction:** {text_label}
**Confidence:** {text_conf}%
"""

    # ---------------- IMAGE ----------------
    if image is not None:
        try:
            class_res = image_pipeline(image)

            image_result_display = "## 🖼 Image Classification\n\n"

            for r in class_res[:3]:
                label = r["label"]
                conf = round(r["score"] * 100, 2)
                image_result_display += f"- **{label}** ({conf}%)\n"

            image_label = class_res[0]["label"]
            image_conf = round(class_res[0]["score"] * 100, 2)

        except Exception as e:
            image_result_display = f"Image processing error: {str(e)}"

    # ---------------- AUDIO ----------------
    if audio is not None:
        res = audio_pipeline(audio)
        transcription = res["text"]

        audio_sent = text_pipeline(transcription)[0]
        raw_audio_label = audio_sent["label"]
        audio_conf = round(audio_sent["score"] * 100, 2)

        label_map = {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "POSITIVE"
        }

        audio_label = label_map.get(raw_audio_label, raw_audio_label)

        audio_result_display = f"""
## 🎙 Audio Intelligence
**Transcription:**  
"{transcription}"
**Detected Tone:** {audio_label} 
({audio_conf}%)
"""

    # ---------------- FUSION ----------------
    reasoning_lines = []

    if text_label:
        reasoning_lines.append(
            f"Text suggests a {text_label.lower()} tone ({text_conf}%)."
        )

    if image_label:
        reasoning_lines.append(
            f"Image classified as {image_label} ({image_conf}%)."
        )

    if audio_label:
        reasoning_lines.append(
            f"Spoken content carries a {audio_label.lower()} tone ({audio_conf}%)."
        )

    fusion_score = 0

    if text_label == "POSITIVE":
        fusion_score += text_conf * 0.4
    elif text_label == "NEGATIVE":
        fusion_score -= text_conf * 0.4

    if audio_label == "POSITIVE":
        fusion_score += audio_conf * 0.3
    elif audio_label == "NEGATIVE":
        fusion_score -= audio_conf * 0.3

    if image_conf:
        fusion_score += image_conf * 0.2

    if fusion_score > 60:
        alignment_message = "All modalities align toward a strong positive contextual interpretation."
    elif fusion_score < 0:
        alignment_message = "Signals indicate potential negative contextual alignment."
    else:
        alignment_message = "Signals are mixed or neutral across modalities."

    fusion_summary = f"""
<div style="padding:20px;border-radius:16px;
background:linear-gradient(135deg,#0f172a,#1e293b);
border:1px solid #1f2a44;">

<h2>🔎 Multimodal Intelligence Summary</h2>
{"<br>".join(reasoning_lines)}
<hr>
<h3>📊 Fusion Score</h3>
<h1 style="color:#14b8a6;">{round(fusion_score,2)}</h1>
<hr>
<h3>🧠 Contextual Interpretation</h3>
<p>{alignment_message}</p>

</div>
"""

    print("Inference finished")

    return fusion_summary, text_result_display, image_result_display, audio_result_display

# ------------------------------------------------
# Custom Dark Styling (UNCHANGED)
custom_css = """
/* =====================================================
   FULL DARK BACKGROUND
===================================================== */
html, body {
    background-color: #0b1220 !important;
}
.gradio-container {
    background-color: #0b1220 !important;
}

/* =====================================================
   DARK CARD BLOCKS
===================================================== */
.gradio-container .block {
    background-color: #162033 !important;
    border: 1px solid #1f2a44 !important;
    border-radius: 14px !important;
}

/* =====================================================
   INPUT FIELDS
===================================================== */
textarea, input, select {
    background-color: #1e293b !important;
    border: 1px solid #2a3a55 !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

/* =====================================================
   TEXT COLORS
===================================================== */
.gradio-container * {
    color: #e2e8f0 !important;
}
h1, h2, h3 {
    color: #ffffff !important;
}

/* =====================================================
   MAIN BUTTON STYLE
===================================================== */
button {
    background: linear-gradient(135deg, #14b8a6, #8b5cf6) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    color: white !important;
    border: none !important;
}
button:hover {
    filter: brightness(1.1);
}

/* Remove disabled fade */
button[disabled] {
    opacity: 1 !important;
    background: linear-gradient(135deg, #14b8a6, #8b5cf6) !important;
    color: white !important;
}

/* =====================================================
   UPLOAD AREA FIX (NO WHITE BOX)
===================================================== */

/* Remove white wrapper behind upload header */
.gr-image > div:first-child,
.gr-audio > div:first-child {
    background: transparent !important;
    border: none !important;
}

/* Remove internal pale backgrounds */
.gr-image .wrap,
.gr-audio .wrap {
    background: transparent !important;
}

/* Style upload header button */
.gr-image button,
.gr-audio button {
    background: linear-gradient(135deg, #14b8a6, #8b5cf6) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    opacity: 1 !important;
    box-shadow: none !important;
}/* ======================================
   REMOVE DEFAULT UPLOAD HEADER BUTTON
====================================== */
.gr-image label,
.gr-audio label {
    display: none !important;
}

.gr-image button,
.gr-audio button {
    display: none !important;
}

/* Gradient drop zone */
.gr-image > div,
.gr-audio > div {
    background: linear-gradient(135deg, #14b8a6, #8b5cf6) !important;
    border-radius: 14px !important;
}

/* Upload preview container */
.gr-file-preview,
.gradio-container div[class*="file"],
.gradio-container div[class*="upload"],
.gradio-container div[class*="progress"] {
    background-color: #162033 !important;
    color: #e2e8f0 !important;
}

/* =====================================================
   SETTINGS MODAL DARK FIX
===================================================== */
.gr-modal,
.gradio-container .modal,
.gradio-container .dialog,
div[class*="overlay"],
div[class*="modal"],
div[class*="dialog"] {
    background-color: #162033 !important;
    color: #e2e8f0 !important;
}
.gr-modal * {
    background-color: #162033 !important;
    color: #e2e8f0 !important;
}

/* =====================================================
   SCROLLBAR DARK
===================================================== */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #0b1220;
}
::-webkit-scrollbar-thumb {
    background: #2a3a55;
    border-radius: 10px;
}
#fusion_box {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #1f2a44;
}

#text_box, #image_box, #audio_box {
    background-color: #162033;
    padding: 16px;
    border-radius: 14px;
    border: 1px solid #1f2a44;
    margin-top: 12px;
}
button {
    transition: all 0.3s ease;
}
button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(20,184,166,0.4);
}
"""
with gr.Blocks(css=custom_css) as demo:

    gr.Markdown("# 🤖 Advanced Multimodal AI System")
    gr.Markdown("Analyze text, images, and audio using Transformer-based AI models.")
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(label="📝 Enter Text")
            gr.Markdown("## 🖼️ Upload Image")
            image_input = gr.Image(type="pil", show_label=False)

            gr.Markdown("## 🎵🎙️ Upload Audio")
            audio_input = gr.Audio(type="filepath", show_label=False)
            analyze_btn = gr.Button("🚀 Run Analysis")

        with gr.Column(scale=1):
            fusion_output = gr.Markdown(elem_id="fusion_box")
            text_output = gr.Markdown(elem_id="text_box")
            image_output = gr.Markdown(elem_id="image_box")
            audio_output = gr.Markdown(elem_id="audio_box")

    analyze_btn.click(
    multimodal_analyze,
    inputs=[text_input, image_input, audio_input],
    outputs=[fusion_output, text_output, image_output, audio_output]
)

if __name__ == "__main__":
    demo.launch()
    # temp
    print("Inference finished")