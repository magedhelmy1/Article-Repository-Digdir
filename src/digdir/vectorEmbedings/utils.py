import numpy as np
import pandas as pd
import requests
import os
from dotenv import load_dotenv
import tensorflow as tf
from tensorboard.plugins import projector


class EmbeddingAPIError(Exception):
    """Exception raised for errors in the embedding API call.

    Attributes:
        message -- explanation of the error
        status_code -- HTTP status code returned by the API call
    """

    def __init__(self, message, status_code=None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

    def __str__(self):
        if self.status_code:
            return f"EmbeddingAPIError: {self.message} Status Code: {self.status_code}"
        else:
            return f"EmbeddingAPIError: {self.message}"


class TextEmbeddingsHandler:
    BASE_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/"

    def __init__(self, hf_token: str, model_id: str):
        if not hf_token:
            raise ValueError("Hugging Face token is required")
        self.hf_token = hf_token
        self.model_id = model_id
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}
        self.api_url = f"{self.BASE_URL}{self.model_id}"

    def _get_embeddings(self, text_chunk: str) -> list:
        """
        Use the Hugging Face API to calculate embeddings for provided text chunks.

        Args:
            text_chunk (str): The sentence or text snippet for which embeddings are required.

        Returns:
            list: A list containing the embedding dimensions.

        Raises:
            Exception: Raises an exception if the API call is unsuccessful.
        """
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": text_chunk, "options": {"wait_for_model": True}},
        )

        if response.status_code == 200:
            embedding = response.json()
            return embedding
        else:
            raise EmbeddingAPIError(
                f"Failed to get embeddings, API status code: {response.status_code}"
            )

    def from_text_to_embeddings(self, text_chunks: list) -> pd.DataFrame:
        """
        Translate sentences into vector embeddings

        Attributes:
            - text_chunks (list): list of example strings

        Returns:
            - embeddings_df (DataFrame): data frame with the columns "text_chunk" and "embeddings"
        """

        # split the embeddings column into individuell columns for each vector dimension
        embeddings_list = []
        for chunk in text_chunks:
            embedding = self._get_embeddings(
                chunk
            )  # This should return the embedding as a list
            embeddings_list.append(embedding)
        embeddings_df = pd.DataFrame(
            {
                "text_chunk": text_chunks,
                "embeddings": embeddings_list,  # Make sure embeddings are stored here
            }
        )
        return embeddings_df

    def visualize_embeddings_with_tensorboard(
        self, embeddings_df: pd.DataFrame, save_location: str = "./logs"
    ):

        # Ensuring the save location exists
        log_dir = os.path.join(save_location, "embedding-visualization")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Save metadata (labels)
        metadata_file = os.path.join(log_dir, "metadata.tsv")
        with open(metadata_file, "w") as metadata:
            for text in embeddings_df["text_chunk"]:
                metadata.write(f"{text}\n")

        embeddings_for_tensorboard = np.array(embeddings_df["embeddings"].tolist())

        weights = tf.Variable(embeddings_for_tensorboard)
        checkpoint = tf.train.Checkpoint(embedding=weights)
        checkpoint_path = checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

        config = projector.ProjectorConfig()
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding_config.metadata_path = "metadata.tsv"

        projector_config_path = os.path.join(log_dir, "projector_config.pbtxt")
        with open(projector_config_path, "w") as config_file:
            config_file.write(str(config))

        projector.visualize_embeddings(log_dir, config)

    def analyze_text_embeddings(
        self,
        information=[
            "våpen",
            "ørner",
            "lillestrøm",
            "ski",
            "ferie",
            "familie",
            "beard in mailbox.",
            "Burgers with fries are tasty.",
            "Moustache is cool.",
            "Norge er et vakkert land.",
        ],
    ):
        embeddings_df = self.from_text_to_embeddings(information)

        return embeddings_df


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    model_id = "sentence-transformers/bert-base-nli-mean-tokens"
    text_handler = TextEmbeddingsHandler(hf_token=hf_token, model_id=model_id)
    embeddings_df = text_handler.analyze_text_embeddings()
    text_handler.visualize_embeddings_with_tensorboard(
        embeddings_df, save_location="./logs"
    )
