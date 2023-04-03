"""Modified llama-hub example for github_repo"""

import argparse
import pickle
import os
import logging
from llama_index import (
    GPTSimpleVectorIndex,
    ServiceContext,
    LLMPredictor,
    download_loader,
)
from llama_index.logger.base import LlamaLogger
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingMode
from llama_index.prompts.chat_prompts import CHAT_REFINE_PROMPT

from langchain.chat_models import ChatOpenAI


assert (
    os.getenv("OPENAI_API_KEY") is not None
), "Please set the OPENAI_API_KEY environment variable."
assert (
    os.getenv("GITHUB_TOKEN") is not None
), "Please set the GITHUB_TOKEN environment variable."

# This is a way to test loaders on different forks/branches.
# LLAMA_HUB_CONTENTS_URL = "https://raw.githubusercontent.com/claysauruswrecks/llama-hub/bugfix/github-repo-splitter"  # noqa: E501
# LOADER_HUB_PATH = "/loader_hub"
# LOADER_HUB_URL = LLAMA_HUB_CONTENTS_URL + LOADER_HUB_PATH

download_loader(
    "GithubRepositoryReader",
    # loader_hub_url=LOADER_HUB_URL,
    # refresh_cache=True,
)

from llama_index.readers.llamahub_modules.github_repo import (  # noqa: E402
    GithubClient,
    GithubRepositoryReader,
)

REPOS = {
    "jerryjliu/llama_index@1b739e1fcd525f73af4a7131dd52c7750e9ca247": dict(
        filter_directories=(
            ["docs", "examples", "gpt_index", "tests"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
        filter_file_extensions=(
            [".py", ".md"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
    ),
    "emptycrown/llama-hub@8312da4ee8fcaf2cbbf5315a2ab8f170d102d081": dict(
        filter_directories=(
            ["loader_hub", "tests"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
        filter_file_extensions=(
            [".py", ".md"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
    ),
}

PICKLE_DOCS_DIR = os.path.join(
    os.path.join(os.path.join(os.path.dirname(__file__), "../"), "data"),
    "pickled_docs",
)
# Create the directory if it does not exist
if not os.path.exists(PICKLE_DOCS_DIR):
    os.makedirs(PICKLE_DOCS_DIR)


# MODEL_NAME = "gpt-3.5-turbo"
MODEL_NAME = "gpt-4"

CHUNK_SIZE_LIMIT = 512
MAX_TOKENS = None  # Set to None to use model's maximum

EMBED_MODEL = OpenAIEmbedding(mode=OpenAIEmbeddingMode.SIMILARITY_MODE)

LLM_PREDICTOR = LLMPredictor(
    llm=ChatOpenAI(
        temperature=0.0, model_name=MODEL_NAME, max_tokens=MAX_TOKENS
    )
)


def load_pickle(filename):
    """Load the pickled embeddings"""
    with open(os.path.join(PICKLE_DOCS_DIR, filename), "rb") as f:
        return pickle.load(f)


def save_pickle(obj, filename):
    """Save the pickled embeddings"""
    with open(os.path.join(PICKLE_DOCS_DIR, filename), "wb") as f:
        pickle.dump(obj, f)


def main():
    """Run the trap."""
    g_docs = {}

    for repo in REPOS.keys():
        repo_owner, repo_name_at_sha = repo.split("/")
        repo_name, commit_sha = repo_name_at_sha.split("@")
        docs_filename = f"{repo_owner}-{repo_name}-{commit_sha}-docs.pkl"
        docs_filepath = os.path.join(PICKLE_DOCS_DIR, docs_filename)

        if os.path.exists(docs_filepath):
            g_docs[repo] = load_pickle(docs_filename)

        if not g_docs.get(repo):
            github_client = GithubClient(os.getenv("GITHUB_TOKEN"))
            loader = GithubRepositoryReader(
                github_client,
                owner=repo_owner,
                repo=repo_name,
                filter_directories=REPOS[repo]["filter_directories"],
                filter_file_extensions=REPOS[repo]["filter_file_extensions"],
                verbose=True,
                concurrent_requests=10,
            )

            embedded_docs = loader.load_data(commit_sha=commit_sha)
            g_docs[repo] = embedded_docs

            save_pickle(embedded_docs, docs_filename)

    # NOTE: set a chunk size limit to < 1024 tokens
    service_context = ServiceContext.from_defaults(
        llm_predictor=LLM_PREDICTOR,
        embed_model=EMBED_MODEL,
        llama_logger=LlamaLogger(),
        chunk_size_limit=512,
    )

    # Collapse all the docs into a single list
    docs = []
    for repo in g_docs.keys():
        docs.extend(g_docs[repo])

    index = GPTSimpleVectorIndex.from_documents(
        documents=docs, service_context=service_context
    )

    # Ask for CLI input in a loop
    while True:
        print("QUERY:")
        query = input()
        answer = index.query(query, refine_template=CHAT_REFINE_PROMPT)
        print(f"ANSWER: {answer}")


# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--verbose",
    action="store_const",
    dest="loglevel",
    const=logging.INFO,
    help="Set to True to enable verbose logging.",
    required=False,
)
parser.add_argument(
    "--debug",
    action="store_const",
    dest="loglevel",
    const=logging.DEBUG,
    default=logging.WARNING,
    help="Set to True to enable debug logging.",
)
args = parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=args.loglevel)
    main()
