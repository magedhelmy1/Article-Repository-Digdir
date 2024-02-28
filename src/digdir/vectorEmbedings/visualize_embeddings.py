import numpy as np
from numpy.linalg import norm
import pandas as pd
import requests
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import plotly.express as px


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
    DEFAULT_MODEL_ID = "sentence-transformers/bert-base-nli-mean-tokens"

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

    def create_pca_plot(
        self, embeddings_df: pd.DataFrame, n_components=3
    ) -> pd.DataFrame:
        """
        Performs principal component analysis (PCA) to reduce the dimensions to 3 (default)
        and visualizes them in a scatter plot.

        Parameters:
            embeddings_df (pd.DataFrame): DataFrame with the columns "text_chunk" and "embeddings".
            n_components (int): Number of components for PCA. Default is 3.

        Returns:
            pd.DataFrame: DataFrame with the 3 most relevant Principal Components.
        """
        # Perform PCA with n_components
        pca = PCA(n_components=n_components)

        # Apply principal component analysis to the embeddings table
        pca_result = pca.fit_transform(embeddings_df[embeddings_df.columns[:-1]])

        # Create a new DataFrame with reduced dimensions
        pca_columns = [f"PC{i+1}" for i in range(n_components)]
        df_reduced = pd.DataFrame(pca_result, columns=pca_columns)
        return df_reduced

    def create_scatter_plot_3d(
        self,
        df_reduced: pd.DataFrame,
        embeddings_df: pd.DataFrame,
        user_query_point: np.ndarray = None,
        user_query_label: str = None,
        save_location: str = None,
    ) -> None:
        """
        Creates and displays a 3D scatter plot visualizing the result of a PCA reduction, highlighting
        the relationship between embedded text chunks and optionally a specific user query.

        This method plots each PCA-reduced point in a 3D space, labels them with their corresponding text chunk,
        and, if provided, visually distinguishes a user query point in a different color. Optionally, it
        can also display dashed lines connecting the user query to its nearest neighbors.

        Parameters:
        - df_reduced (pd.DataFrame): A DataFrame containing the PCA-reduced dimensions of the text embeddings.
        - embeddings_df (pd.DataFrame): The original DataFrame of text embeddings before PCA reduction.
        - user_query_point (np.ndarray, optional): The PCA-reduced coordinates of the user query.
        - user_query_label (str, optional): The label for the user query point to be displayed on the plot.
        - save_location (str, optional): The location where the plot will be saved.

        Returns:
        - None: This method does not return a value but displays the 3D scatter plot.
        """
        # Create a DataFrame for plotly
        plot_df = df_reduced.copy()
        plot_df["text_chunk"] = embeddings_df["text_chunk"]

        # Create the 3D scatter plot using plotly.express
        fig = px.scatter_3d(
            plot_df, x="PC1", y="PC2", z="PC3", text="text_chunk", color="text_chunk"
        )

        # Set labels and title
        fig.update_layout(
            scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3")
        )
        fig.update_layout(title="3D PCA Scatter Plot of Text Embeddings")

        # Display the plot
        if save_location:
            # Change file extension to .html if necessary
            if not save_location.endswith(".html"):
                save_location += ".html"
            fig.write_html(save_location)  # Save the figure as an HTML file
            print(f"3D scatter plot saved at: {save_location}")
        fig.show()

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
        if save_location and not os.path.isdir(os.path.dirname(save_location)):
            raise FileNotFoundError("Save location directory does not exist.")

        plt.figure(figsize=(10, 7))
        plt.scatter(
            df_reduced["PC1"],
            df_reduced["PC2"],
            label="Information Provided",
            alpha=0.7,
        )

        texts = []
        # Add labels to each point for clarity, storing in the texts list for later adjustment
        for i, label in enumerate(embeddings_df["text_chunk"].to_list()):
            texts.append(
                plt.text(
                    df_reduced["PC1"][i] + 0.01,
                    df_reduced["PC2"][i] + 0.01,
                    label,
                    fontsize=9,
                )
            )

        if user_query_point is not None and user_query_label is not None:
            plt.scatter(
                user_query_point[0],
                user_query_point[1],
                color="green",
                label="User Query",
                zorder=5,
            )
            # Adding user query label to texts list
            texts.append(
                plt.text(
                    user_query_point[0] + 0.01,
                    user_query_point[1] + 0.01,
                    user_query_label,
                    fontsize=9,
                    color="green",
                )
            )

            # Calculate distances only if the user query point is provided
            distances = df_reduced.apply(
                lambda x: norm(user_query_point - np.array([x["PC1"], x["PC2"]])),
                axis=1,
            )
            closest_points_indices = distances.nsmallest(
                3
            ).index  # Get only the closest 3 points

            # Draw dashed lines to the closest 3 points
            for index in closest_points_indices:
                plt.plot(
                    [user_query_point[0], df_reduced.iloc[index]["PC1"]],
                    [user_query_point[1], df_reduced.iloc[index]["PC2"]],
                    "k--",
                    linewidth=0.5,
                )

        # Optimizing label positions to minimize overlap
        # adjust_text(texts, arrowprops=dict(arrowstyle="->", color="w"))

        plt.xlabel("Embedding Dimension 1")
        plt.ylabel("Embedding Dimension 2")
        plt.title("Semantic Text Embedding Visualization")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_location:
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

        # Perform PCA with 3 components
        pca = PCA(n_components=3)

        # Apply PCA to the embeddings, excluding the last element (user_query)
        df_embeddings = embeddings_df.iloc[:-1, :-1]  # Adjust indexing if necessary
        pca_result = pca.fit_transform(df_embeddings)

        # Separate the PCA results for information and user_query
        df_reduced = pd.DataFrame(pca_result, columns=["PC1", "PC2", "PC3"])
        user_query_point = pca_result[-1]  # This assumes the user query is the last row

        # Create and display the 3D scatter plot
        self.create_scatter_plot_3d(
            df_reduced,
            embeddings_df.iloc[:-1],  # Exclude the user query from the plot data
            user_query_point=user_query_point[:3],  # Ensure this is a 3D point
            user_query_label=user_query,
            save_location=save_location,
        )

    def analyze_text_embeddings(
        self,
        user_query="army",
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
