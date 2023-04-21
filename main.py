from langchain.llms import LlamaCpp
from llama_index import LLMPredictor, ServiceContext, PromptHelper, SimpleDirectoryReader, SimpleWebPageReader, \
    GPTListIndex, download_loader
from pathlib import Path
import argparse

# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

home = Path.home()
parser = argparse.ArgumentParser()
default_llm = "dalai/alpaca/models/7B/ggml-model-q4_0.bin"
parser.add_argument('--save', type=str, required=False)
parser.add_argument('--prompt', type=str, required=False)
parser.add_argument('--path', type=str, required=False)
parser.add_argument('--url', type=str, required=False)
parser.add_argument('--git', type=str, required=False)
parser.add_argument('--model', type=str, required=False)
args = parser.parse_args()
if args.model is None:
    model_path = f"{home}/{default_llm}".format(home, default_llm)
else:
    model_path = args.model_path

if __name__ == "__main__":
    llm_predictor = LLMPredictor(
        llm=LlamaCpp(
                model_path=model_path, 
                n_ctx=2048, 
                use_mlock=True, 
                top_k=10000, 
                max_tokens=300, 
                n_parts=-1, 
                temperature=0.8, 
                top_p=0.40,
                last_n_tokens_size=100,
                n_threads=8,
                f16_kv=True
            )
        )
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    # Load data from list of urls
    if args.url:
        with open(args.url, "r") as urls:
            lines = urls.readlines()
            documents = SimpleWebPageReader(html_to_text=True).load_data(lines)

    # Load git repo by path
    if args.git:
        GPTRepoReader = download_loader("GPTRepoReader")
        loader = GPTRepoReader()
        documents = loader.load_data(repo_path=args.git)
    
    # Load directory by path
    if args.path:
        documents = SimpleDirectoryReader(args.path).load_data()
    
    # Persist index
    index = GPTListIndex.from_documents(documents, service_context=service_context)
    if args.save:
        index.save_to_disk(args.save)
    
    # Prompt index
    if args.prompt:
        response = index.query(args.prompt)
        print(response)   

