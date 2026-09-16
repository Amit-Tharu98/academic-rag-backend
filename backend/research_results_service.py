from pathlib import Path

import pandas as pd


RESULTS_FOLDER = Path("results")

RETRIEVAL_SUMMARY_FILE = (
    RESULTS_FOLDER
    / "retrieval_summary.csv"
)

GENERATION_SUMMARY_FILE = (
    RESULTS_FOLDER
    / "generation_summary.csv"
)

RETRIEVAL_RESULTS_FILE = (
    RESULTS_FOLDER
    / "retrieval_results.csv"
)

GENERATION_RESULTS_FILE = (
    RESULTS_FOLDER
    / "generation_results.csv"
)


class ResearchResultsService:
    """
    Load and prepare dissertation experiment results
    for the frontend research dashboard.
    """

    def _load_csv(
        self,
        file_path: Path,
    ) -> pd.DataFrame:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Results file not found: "
                f"{file_path}"
            )

        dataframe = pd.read_csv(
            file_path
        )

        if dataframe.empty:
            raise ValueError(
                f"Results file is empty: "
                f"{file_path}"
            )

        return dataframe

    
    # Retrieval summary
    

    def get_retrieval_summary(
        self,
    ) -> list[dict]:
        """
        Return average retrieval metrics
        for each embedding model.
        """

        dataframe = self._load_csv(
            RETRIEVAL_SUMMARY_FILE
        )

        
        # Handle model stored as CSV index
        

        if (
            "model"
            not in dataframe.columns
        ):

            first_column = (
                dataframe.columns[0]
            )

            dataframe = (
                dataframe.rename(
                    columns={
                        first_column:
                            "model"
                    }
                )
            )

        records = (
            dataframe
            .fillna(0)
            .to_dict(
                orient="records"
            )
        )

        return records

    
    # Generation summary
    

    def get_generation_summary(
        self,
    ) -> list[dict]:
        """
        Return average generation metrics
        for each embedding model.
        """

        dataframe = self._load_csv(
            GENERATION_SUMMARY_FILE
        )

        if (
            "model"
            not in dataframe.columns
        ):

            first_column = (
                dataframe.columns[0]
            )

            dataframe = (
                dataframe.rename(
                    columns={
                        first_column:
                            "model"
                    }
                )
            )

        records = (
            dataframe
            .fillna(0)
            .to_dict(
                orient="records"
            )
        )

        return records

    
    # Detailed retrieval results
    

    def get_retrieval_results(
        self,
    ) -> list[dict]:
        """
        Return detailed per-question retrieval results.
        """

        dataframe = self._load_csv(
            RETRIEVAL_RESULTS_FILE
        )

        return (
            dataframe
            .fillna("")
            .to_dict(
                orient="records"
            )
        )

    
    # Detailed generation results
    

    def get_generation_results(
        self,
    ) -> list[dict]:
        """
        Return detailed per-question generation results.
        """

        dataframe = self._load_csv(
            GENERATION_RESULTS_FILE
        )

        return (
            dataframe
            .fillna("")
            .to_dict(
                orient="records"
            )
        )

    
    # Research overview
    

    def get_overview(
        self,
    ) -> dict:
        """
        Return high-level research findings.
        """

        retrieval = (
            self.get_retrieval_summary()
        )

        generation = (
            self.get_generation_summary()
        )

        if not retrieval:
            raise ValueError(
                "No retrieval results found."
            )

        if not generation:
            raise ValueError(
                "No generation results found."
            )

        
        # Best retrieval models
        

        best_top5 = max(
            retrieval,
            key=lambda item:
                float(
                    item.get(
                        "top5_accuracy",
                        0,
                    )
                ),
        )

        best_mrr = max(
            retrieval,
            key=lambda item:
                float(
                    item.get(
                        "reciprocal_rank",
                        0,
                    )
                ),
        )

        fastest_retrieval = min(
            retrieval,
            key=lambda item:
                float(
                    item.get(
                        "retrieval_time",
                        float("inf"),
                    )
                ),
        )

        
        # Best generation models
        

        best_rouge1 = max(
            generation,
            key=lambda item:
                float(
                    item.get(
                        "rouge1",
                        0,
                    )
                ),
        )

        best_rouge2 = max(
            generation,
            key=lambda item:
                float(
                    item.get(
                        "rouge2",
                        0,
                    )
                ),
        )

        best_rougel = max(
            generation,
            key=lambda item:
                float(
                    item.get(
                        "rougeL",
                        0,
                    )
                ),
        )

        return {
            "corpus": {
                "documents": 30,
                "pages": 414,
                "chunks": 2762,
                "evaluation_questions": 90,
                "top_k": 5,
            },

            "embedding_models": [
                "sentence_transformer",
                "bge",
                "openai",
            ],

            "retrieval_findings": {
                "best_top5_accuracy": {
                    "model":
                        best_top5["model"],

                    "value":
                        float(
                            best_top5[
                                "top5_accuracy"
                            ]
                        ),
                },

                "best_mrr": {
                    "model":
                        best_mrr["model"],

                    "value":
                        float(
                            best_mrr[
                                "reciprocal_rank"
                            ]
                        ),
                },

                "fastest_model": {
                    "model":
                        fastest_retrieval[
                            "model"
                        ],

                    "retrieval_time":
                        float(
                            fastest_retrieval[
                                "retrieval_time"
                            ]
                        ),
                },
            },

            "generation_findings": {
                "best_rouge1": {
                    "model":
                        best_rouge1["model"],

                    "value":
                        float(
                            best_rouge1[
                                "rouge1"
                            ]
                        ),
                },

                "best_rouge2": {
                    "model":
                        best_rouge2["model"],

                    "value":
                        float(
                            best_rouge2[
                                "rouge2"
                            ]
                        ),
                },

                "best_rougeL": {
                    "model":
                        best_rougel["model"],

                    "value":
                        float(
                            best_rougel[
                                "rougeL"
                            ]
                        ),
                },
            },

            "interpretation": (
                "OpenAI achieved the strongest overall "
                "retrieval and generation effectiveness, "
                "while BGE achieved closely comparable "
                "generation performance with lower retrieval "
                "latency. Sentence Transformer was the "
                "fastest model but produced lower retrieval "
                "effectiveness in the controlled evaluation."
            ),
        }