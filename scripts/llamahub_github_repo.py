import pickle
import os
import logging
from llama_index import GPTSimpleVectorIndex, PromptHelper, ServiceContext, LLMPredictor
from langchain import OpenAI

# Set maximum input size
max_input_size = 1000
# Set number of output tokens
num_output = 256
# Set maximum chunk overlap
max_chunk_overlap = 20

prompt_helper = PromptHelper(
    max_input_size,
    num_output,
    max_chunk_overlap,
)

# Define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003"))

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, prompt_helper=prompt_helper
)

assert (
    os.getenv("OPENAI_API_KEY") is not None
), "Please set the OPENAI_API_KEY environment variable."

from llama_index import download_loader

logging.basicConfig(level=logging.DEBUG)

# LLAMA_HUB_CONTENTS_URL = "https://raw.githubusercontent.com/claysauruswrecks/llama-hub/bugfix/github-repo-splitter"
# LOADER_HUB_PATH = "/loader_hub"
# LOADER_HUB_URL = LLAMA_HUB_CONTENTS_URL + LOADER_HUB_PATH

# download_loader(
#     "GithubRepositoryReader", loader_hub_url=LOADER_HUB_URL, refresh_cache=True
# )

download_loader("GithubRepositoryReader")

from llama_index.readers.llamahub_modules.github_repo import (
    GithubClient,
    GithubRepositoryReader,
)

docs = None

if os.path.exists("docs.pkl"):
    with open("docs.pkl", "rb") as f:
        docs = pickle.load(f)

if docs is None:
    github_client = GithubClient(os.getenv("GITHUB_TOKEN"))
    loader = GithubRepositoryReader(
        github_client,
        owner="jerryjliu",
        repo="llama_index",
        filter_directories=(
            ["gpt_index", "docs"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
        filter_file_extensions=([".py"], GithubRepositoryReader.FilterType.INCLUDE),
        verbose=True,
        concurrent_requests=10,
    )

    docs = loader.load_data(commit_sha="1b739e1fcd525f73af4a7131dd52c7750e9ca247")

    with open("docs.pkl", "wb") as f:
        pickle.dump(docs, f)

index = GPTSimpleVectorIndex.from_documents(docs, service_context=service_context)

index.query("Explain each LlamaIndex class?")
