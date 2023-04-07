"""Modified llama-hub example for github_repo"""

import argparse
import logging
import os
import pickle
import qdrant_client

from llama_index import GPTQdrantIndex

from langchain.chat_models import ChatOpenAI
from llama_index import (
    GPTSimpleVectorIndex,
    LLMPredictor,
    ServiceContext,
    download_loader,
)

# from llama_index.logger.base import LlamaLogger
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingMode
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.prompts.chat_prompts import CHAT_REFINE_PROMPT

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

# TODO: Modify github loader to support exclude list of filenames and unblock .ipynb  # noqa: E501
REPOS = {
    # NOTE: Use this to find long line filetypes to avoid: `find . -type f -exec sh -c 'awk "BEGIN { max = 0 } { if (length > max) max = length } END { printf \"%s:%d\n\", FILENAME, max }" "{}"' \; | sort -t: -k2 -nr`  # noqa: E501
    "jerryjliu/llama_index@1b739e1fcd525f73af4a7131dd52c7750e9ca247": dict(
        filter_directories=(
            ["docs", "examples", "gpt_index", "tests"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
        filter_file_extensions=(
            [
                ".bat",
                ".md",
                # ".ipynb",
                ".py",
                ".rst",
                ".sh",
            ],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
    ),
    "emptycrown/llama-hub@8312da4ee8fcaf2cbbf5315a2ab8f170d102d081": dict(
        filter_directories=(
            ["loader_hub", "tests"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
        filter_file_extensions=(
            [".py", ".md", ".txt"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
    ),
    "hwchase17/langchain@d85f57ef9cbbbd5e512e064fb81c531b28c6591c": dict(
        filter_directories=(
            ["docs", "langchain", "tests"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
        filter_file_extensions=(
            [
                ".bat",
                ".md",
                # ".ipynb",
                ".py",
                ".rst",
                ".sh",
            ],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
    ),
    "qdrant/qdrant@f978dbbcb157f6ff243452b7c2d9356916fdaadc": dict(
        filter_directories=(
            [
                "benches",
                "config",
                "docs",
                "lib",
                "openapi",
                "src",
                "tests",
                "tools",
            ],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
        filter_file_extensions=(
            [
                ".bat",
                ".js",
                ".md",
                # ".ipynb",
                ".py",
                ".rs",
                ".rst",
                ".sh",
                ".toml",
                "Dockerfile",
            ],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
    ),
    "qdrant/docs@777a161bab8da8d286ba33b26256026190e18247": dict(
        filter_directories=(
            ["cloud", "qdrant/v1.1.x"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
        filter_file_extensions=(
            [".md", "README.md"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
    ),
}

# MODEL_NAME = "gpt-3.5-turbo"
MODEL_NAME = "gpt-4"

CHUNK_SIZE_LIMIT = 512
CHUNK_OVERLAP = 200  # default
MAX_TOKENS = None  # Set to None to use model's maximum, default

EMBED_MODEL = OpenAIEmbedding(mode=OpenAIEmbeddingMode.SIMILARITY_MODE)

LLM_PREDICTOR = LLMPredictor(
    llm=ChatOpenAI(
        temperature=0.1, model_name=MODEL_NAME, max_tokens=MAX_TOKENS
    )
)

PICKLE_DOCS_DIR = os.path.join(
    os.path.join(os.path.join(os.path.dirname(__file__), "../"), "data"),
    "pickled_docs",
)

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

QDRANT_INDEX_DOCS_DIR = os.path.join(
    os.path.join(os.path.join(os.path.dirname(__file__), "../"), "data"),
    "qdrant_index_docs",
)

QDRANT_INDEX_DOCS_FILE = os.path.join(
    QDRANT_INDEX_DOCS_DIR, "qdrant_index_docs.json"
)


# Create the directories if they do not exist
if not os.path.exists(PICKLE_DOCS_DIR):
    os.makedirs(PICKLE_DOCS_DIR)

if not os.path.exists(QDRANT_INDEX_DOCS_DIR):
    os.makedirs(QDRANT_INDEX_DOCS_DIR)


def load_pickle(filename):
    """Load the pickled embeddings"""
    with open(os.path.join(PICKLE_DOCS_DIR, filename), "rb") as f:
        logging.debug(f"Loading pickled embeddings from {filename}")
        return pickle.load(f)


def save_pickle(obj, filename):
    """Save the pickled embeddings"""
    with open(os.path.join(PICKLE_DOCS_DIR, filename), "wb") as f:
        logging.debug(f"Saving pickled embeddings to {filename}")
        pickle.dump(obj, f)


def main(args):
    """Run the trap."""
    g_docs = {}

    for repo in REPOS.keys():
        logging.debug(f"Processing {repo}")
        repo_owner, repo_name_at_sha = repo.split("/")
        repo_name, commit_sha = repo_name_at_sha.split("@")
        docs_filename = f"{repo_owner}-{repo_name}-{commit_sha}-docs.pkl"
        docs_filepath = os.path.join(PICKLE_DOCS_DIR, docs_filename)

        if os.path.exists(docs_filepath):
            logging.debug(f"Path exists: {docs_filepath}")
            g_docs[repo] = load_pickle(docs_filename)

        if not g_docs.get(repo):
            github_client = GithubClient(os.getenv("GITHUB_TOKEN"))
            loader = GithubRepositoryReader(
                github_client,
                owner=repo_owner,
                repo=repo_name,
                filter_directories=REPOS[repo]["filter_directories"],
                filter_file_extensions=REPOS[repo]["filter_file_extensions"],
                verbose=args.debug,
                concurrent_requests=10,
            )

            embedded_docs = loader.load_data(commit_sha=commit_sha)
            g_docs[repo] = embedded_docs

            save_pickle(embedded_docs, docs_filename)

    # NOTE: set a chunk size limit to < 1024 tokens
    service_context = ServiceContext.from_defaults(
        llm_predictor=LLM_PREDICTOR,
        embed_model=EMBED_MODEL,
        node_parser=SimpleNodeParser(
            text_splitter=TokenTextSplitter(
                separator=" ",
                chunk_size=CHUNK_SIZE_LIMIT,
                chunk_overlap=CHUNK_OVERLAP,
                backup_separators=[
                    "\n",
                    "\n\n",
                    "\r\n",
                    "\r",
                    "\t",
                    "\\",
                    "\f",
                    "//",
                    "+",
                    "=",
                    ",",
                    ".",
                    "a",
                    "e",  # TODO: Figure out why lol
                ],
            )
        ),
        # llama_logger=LlamaLogger(),  # TODO: ?
    )

    # Collapse all the docs into a single list
    logging.debug("Collapsing all the docs into a single list")
    docs = []
    for repo in g_docs.keys():
        docs.extend(g_docs[repo])

    # index = GPTSimpleVectorIndex.from_documents(
    #     documents=docs, service_context=service_context
    # )
    q_client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Check if QDRANT index already exists
    if os.path.exists(QDRANT_INDEX_DOCS_FILE):
        logging.debug(f"QDRANT index already exists: {QDRANT_INDEX_DOCS_FILE}")
        index = GPTQdrantIndex.load_from_disk(
            QDRANT_INDEX_DOCS_FILE, client=q_client
        )
    else:
        index = GPTQdrantIndex.from_documents(
            documents=docs,
            client=q_client,
            service_context=service_context,
            collection_name="docs",
        )

    # Check if QDRANT index already exists
    if not os.path.exists(QDRANT_INDEX_DOCS_FILE):
        logging.debug(f"QDRANT index already exists: {QDRANT_INDEX_DOCS_FILE}")
        index.save_to_disk(QDRANT_INDEX_DOCS_FILE)

    # Ask for CLI input in a loop
    while True:
        print("QUERY:")
        query = input()
        answer = index.query(query, refine_template=CHAT_REFINE_PROMPT)
        print(f"ANSWER: {answer}")
        if args.pdb:
            import pdb

            pdb.set_trace()


# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Enable debug logging.",
)
parser.add_argument(
    "--pdb",
    action="store_true",
    help="Invoke PDB after each query.",
)
args = parser.parse_args()

if __name__ == "__main__":
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    main(args)
