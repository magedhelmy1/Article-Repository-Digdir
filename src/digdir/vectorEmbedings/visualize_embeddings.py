import streamlit as st
from datetime import datetime
import subprocess
import socket
import os
import time
import psutil
from digdir.vectorEmbedings.utils import TextEmbeddingsHandler
from digdir.vectorEmbedings.app_sample_data import (
    embedding_test_items,
    model_choices,
)
from streamlit_tensorboard import st_tensorboard
from dotenv import load_dotenv

load_dotenv()
local_dev_mode = os.getenv("LOCAL")


# Utility Functions
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def launch_tensorboard(logdir):
    port = find_free_port()
    path = os.path.join(logdir, "embedding-visualization")
    tb_process = subprocess.Popen(
        ["tensorboard", "--logdir", path, "--port", str(port)]
    )
    return port, tb_process


def stop_tensorboard():
    if "tb_process" in st.session_state:
        process = psutil.Process(st.session_state["tb_process"].pid)
        for child in process.children(recursive=True):
            child.kill()
        process.kill()
        del st.session_state["tb_process"]
        st.success("TensorBoard has been stopped.")


st.markdown(
    """
    <a href="https://github.com/magedhelmy1/Article-Repository-Digdir" target="_blank">
        <button style="
            background-color: #008CBA; /* Blue */
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
            border: 2px solid #004C8C; /* Darker blue */
            transition-duration: 0.4s;
            box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24), 0 17px 50px 0 rgba(0,0,0,0.19);
        ">
            View Code on GitHub
        </button>
    </a>
    """,
    unsafe_allow_html=True,
)
# UI Setup
st.header("Embeddings Visualization")
hf_token_input = st.text_input(
    label="Hugging Face Token",
    value="hf_XXXXX",
    help="Please enter your Hugging Face API token. [How to get a token?](https://huggingface.co/docs/hub/security-tokens)",
)


model_id_input = st.selectbox(
    "Model ID",
    options=model_choices,
    index=model_choices.index("NbAiLab/nb-sbert-base"),
    help="Select a model ID for embeddings.",
)
default_text_for_embedding = "\n".join(embedding_test_items)

user_texts = st.text_area(
    "Texts for Embedding",
    height=450,
    help="write each word or sentence on a new line. Example: ski ferie familie",
    value=default_text_for_embedding,
)

# Replace direct button instantiation with placeholders
visualize_button_placeholder = st.empty()

if visualize_button_placeholder.button("Visualize Embeddings"):
    visualize_button_placeholder.empty()  # Clear the button

    if not hf_token_input or not model_id_input or user_texts.strip() == "":
        st.error("Please fill in all fields.")
    else:
        with st.spinner("Calculating embeddings..."):
            # Initialize the embeddings handler with user inputs
            text_handler = TextEmbeddingsHandler(
                hf_token=hf_token_input, model_id=model_id_input
            )

            # Split user input text for processing
            text_chunks = user_texts.strip().split("\n")

            # Unique directory for TensorBoard logs based on the current timestamp
            unique_dir = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.abspath(os.path.join("logs", unique_dir))

            # Process embeddings and setup visualization
            embeddings_df = text_handler.analyze_text_embeddings(text_chunks)
            text_handler.visualize_embeddings_with_tensorboard(
                embeddings_df, save_location=log_dir
            )

            if local_dev_mode:
                print("Running in local dev mode")
                # Launch TensorBoard and temporarily store the process in session_state
                port, tb_process = launch_tensorboard(log_dir)
                time.sleep(5)
                st.session_state["tb_process"] = tb_process

                # Generating URL and displaying success message
                tb_url = f"http://localhost:{port}?darkMode=true#projector"
                st.success(
                    f"TensorBoard is running. [Click here to view]({tb_url}) it in a new tab."
                )
            else:
                path = os.path.join(log_dir, "embedding-visualization")
                st_tensorboard(logdir=path, port=6006, width=1000)

            # Pause to ensure TensorBoard has time to initialize


if "tb_process" in st.session_state:
    stop_button_placeholder = st.empty()
    if stop_button_placeholder.button("Stop TensorBoard"):
        stop_tensorboard()
        stop_button_placeholder.empty()
