import numpy as np
from numpy.linalg import norm
import pandas as pd
import requests
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv


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
    DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, hf_token: str, model_id: str = DEFAULT_MODEL_ID):
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
        embeddings = [self._get_embeddings(chunk) for chunk in text_chunks]
        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df["text_chunk"] = text_chunks

        return embeddings_df

    def create_pca_plot(self, embeddings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs principal component analysis (PCA) to reduce the dimensions to 2
        and visualizes them in a scatter plot.

        Parameters:
            embeddings_df (pd.DataFrame): DataFrame with the columns "text_chunk" and "embeddings".

        Returns:
            pd.DataFrame: DataFrame with the 2 most relevant Principal Components.
        """
        # Perform PCA with 2 components
        pca = PCA(n_components=2)

        # Apply principal component analysis to the embeddings table
        df_reduced = pca.fit_transform(embeddings_df[embeddings_df.columns[1:-1]])

        # Create a new DataFrame with reduced dimensions
        df_reduced = pd.DataFrame(df_reduced, columns=["PC1", "PC2"])

        # Create and save scatter plot
        self.create_scatter_plot(df_reduced, embeddings_df)

        return df_reduced

    def create_scatter_plot(
        self,
        df_reduced: pd.DataFrame,
        embeddings_df: pd.DataFrame,
        user_query_point: np.ndarray = None,
        user_query_label: str = None,
        save_location: str = None,
    ) -> None:
        """
        Creates and displays a scatter plot visualizing the result of a PCA reduction, highlighting
        the relationship between embedded text chunks and optionally a specific user query.

        This method plots each PCA-reduced point, labels them with their corresponding text chunk,
        and, if provided, visually distinguishes a user query point in green. Additionally, it
        calculates the distances from the user query to all other points, drawing dashed lines
        to the closest three, emphasizing the query's nearest neighbors in the plot.

        Parameters:
        - df_reduced (pd.DataFrame): A DataFrame containing the PCA-reduced dimensions of the text embeddings,
        specifically, columns labeled "PC1" and "PC2" for the first two principal components.
        - embeddings_df (pd.DataFrame): The original DataFrame of text embeddings before PCA reduction.
        This DataFrame should contain at least a column "text_chunk" with the text corresponding
        to each point in `df_reduced`.
        - user_query_point (np.ndarray, optional): The coordinates of the user query in the PCA-reduced
        space, expressed as a numpy array `[x, y]`. This point will be highlighted and connected to its
        nearest neighbors if provided.
        - user_query_label (str, optional): The label for the user query point to be displayed on the plot.
        Ideally, this is the text of the user query itself. This parameter is required if `user_query_point`
        is not None.
        - save_location (str, optional): The location where the plot will be saved.

        Returns:
        - None: This method does not return a value but displays the scatter plot and saves it as an image file.
        """
        if not os.path.isdir(os.path.dirname(save_location)):
            raise FileNotFoundError("Save location directory does not exist.")

        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            df_reduced["PC1"], df_reduced["PC2"], label="Information Provided"
        )

        # Optionally add labels to each point for clarity
        for i, label in enumerate(embeddings_df["text_chunk"].to_list()):
            plt.text(
                df_reduced["PC1"][i] + 0.003,
                df_reduced["PC2"][i] + 0.003,
                label,
                fontsize=9,
            )

        # Plot user query point if provided
        if user_query_point is not None and user_query_label is not None:
            distances = df_reduced.apply(
                lambda x: norm(user_query_point - np.array([x["PC1"], x["PC2"]])),
                axis=1,
            )
            closest_points_indices = distances.nsmallest(
                3
            ).index  # Updated to get only the closest 3 points

            plt.scatter(
                user_query_point[0],
                user_query_point[1],
                color="green",
                label="User Query",
                zorder=5,
            )
            plt.text(
                user_query_point[0] + 0.003,
                user_query_point[1] + 0.003,
                user_query_label,
                fontsize=9,
                color="green",
            )

        plt.xlabel("Embedding Dimension 1")
        plt.ylabel("Embedding Dimension 2")
        plt.title("Semantic Text Embedding Visualization")

        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)

        if user_query_point is not None:
            # Calculate distances to the user query point and sort them
            distances = df_reduced.apply(
                lambda x: norm(user_query_point - np.array([x["PC1"], x["PC2"]])),
                axis=1,
            )
            closest_points_indices = distances.nsmallest(3).index

            # Draw dashed lines to the closest 3 points
            for index in closest_points_indices:
                if index < len(df_reduced):
                    plt.plot(
                        [user_query_point[0], df_reduced.loc[index, "PC1"]],
                        [user_query_point[1], df_reduced.loc[index, "PC2"]],
                        "k--",
                        linewidth=0.5,
                    )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        absolute_image_path = os.path.abspath(save_location)

        plt.savefig(absolute_image_path, format="png", bbox_inches="tight")
        print(f"Scatter plot saved at: {absolute_image_path}")

        plt.show()

    def calculate_cosine_similarity(
        self, text_chunk: str, embeddings_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate the cosine similarity between the query sentence and every other sentence.

        Parameters:
            text_chunk (str): The text snippet we want to use to look for similar entries in our database (embeddings_df).
            embeddings_df (pd.DataFrame): DataFrame with the columns "text_chunk" and "embeddings".

        Returns:
            pd.DataFrame: The DataFrame including a "cos_sim" column for cosine similarities, sorted in descending order.
        """

        # use the _get_embeddings function the retrieve the embeddings for the text chunk
        sentence_embedding = self._get_embeddings(text_chunk)

        # combine all dimensions of the vector embeddings to one array
        embeddings_df["embeddings_array"] = embeddings_df.apply(
            lambda row: row.values[:-1], axis=1
        )

        # create a list to store the calculated cosine similarity
        cos_sim = []

        for _, row in embeddings_df.iterrows():
            A = row.embeddings_array
            B = sentence_embedding

            cosine_similarity = np.dot(A, B) / (norm(A) * norm(B))
            cos_sim.append(cosine_similarity)

        embeddings_df["cos_sim"] = cos_sim
        embeddings_df.sort_values(by=["cos_sim"], ascending=False)

        return embeddings_df

    def plot_with_user_query(
        self, text_chunks: list, user_query: str, save_location: str
    ):
        """
        Combine the PCA plotting of embeddings with a user query.

        Parameters:
            text_chunks (list): List of text chunks for generating embeddings.
            user_query (str): User query to be plotted distinctly.
            save_location (str): The location where the plot will be saved.

        """

        # Generate embeddings for the provided text chunks
        embeddings_df = self.from_text_to_embeddings(text_chunks + [user_query])

        # Perform PCA with 2 components
        pca = PCA(n_components=2)

        # We skip the last element (user_query) when applying PCA to the embeddings
        df_embeddings = embeddings_df[embeddings_df.columns[:-2]]
        pca_result = pca.fit_transform(df_embeddings)

        # Separate the PCA results for information and user_query
        df_reduced = pd.DataFrame(pca_result[:-1], columns=["PC1", "PC2"])
        user_query_point = pca_result[-1]

        # Create a new DataFrame for information without the user query
        df_reduced_info = pd.DataFrame(df_reduced, columns=["PC1", "PC2"])
        embeddings_df = embeddings_df.drop(
            embeddings_df.index[-1]
        )  # Drop the user query row

        # In plot_with_user_query method, modify the create_scatter_plot call like this:
        self.create_scatter_plot(
            df_reduced_info,
            embeddings_df,
            user_query_point=user_query_point,
            user_query_label=user_query,
            save_location=save_location,
        )

    def analyze_text_embeddings(
        self,
        user_query="The blue is blue.",
        information=[
            "The sky is blue.",
            "The grass is green.",
            "The sun is shining.",
            "I love chocolate.",
            "Pizza is delicious.",
            "Coding is fun.",
            "Roses are red.",
            "Violets are blue.",
            "Water is essential for life.",
            "The moon orbits the Earth.",
        ],
        save_location="./principal_component_plot.png",
    ):
        embeddings_df = self.from_text_to_embeddings(information)
        embeddings_cosine_df = self.calculate_cosine_similarity(
            user_query, embeddings_df
        )
        self.plot_with_user_query(information, user_query, save_location=save_location)
        similarity_ranked_df = embeddings_cosine_df[
            ["text_chunk", "cos_sim"]
        ].sort_values(by="cos_sim", ascending=False)
        user_query_row = pd.DataFrame(
            [{"text_chunk": f"(user query) {user_query}", "cos_sim": 1.00}]
        )
        final_df = pd.concat([user_query_row, similarity_ranked_df], ignore_index=True)
        return final_df


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    text_handler = TextEmbeddingsHandler(hf_token=hf_token)
    results_df = text_handler.analyze_text_embeddings()
    print(results_df)
