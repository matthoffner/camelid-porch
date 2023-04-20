import subprocess
from llama_index import LLMPredictor, ServiceContext, PromptHelper, SimpleDirectoryReader, SimpleWebPageReader, \
    GPTListIndex, download_loader
from pathlib import Path
import asyncio
from typing import List, Optional, Mapping, Any
from langchain.llms.base import Generation, LLMResult, BaseLLM
from pydantic import BaseModel
import threading
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
parser.add_argument('--save', type=str, required=False)
parser.add_argument('--prompt', type=str, required=False)
parser.add_argument('--path', type=str, required=False)
parser.add_argument('--url', type=str, required=False)
parser.add_argument('--git', type=str, required=False)
parser.add_argument('--model-main-path', type=str, required=False)
parser.add_argument('--model-path', type=str, required=False)
args = parser.parse_args()
if args.model_main_path is None:
    model_main_path = f"{home}/dalai/alpaca/main".format(home)
else:
    model_main_path = args.model_main_path
if args.model_path is None:
    model_path = f"{home}/dalai/alpaca/models/7B/ggml-model-q4_0.bin".format(home)
else:
    model_path = args.model_path

def remove_matching_end(a, b):
    min_length = min(len(a), len(b))

    for i in range(min_length, 0, -1):
        if a[-i:] == b[:i]:
            return b[i:]

    return b


async def load_model(
        prompt: str = "The sky is blue because",
        n_predict: int = 300,
        temp: float = 0.8,
        top_k: int = 10000,
        top_p: float = 0.40,
        repeat_last_n: int = 100,
        repeat_penalty: float = 1.2,
        chunk_size: int = 4,  # Define a chunk size (in bytes) for streaming the output bit by bit
):
    args = (
        model_main_path,
        "--model",
        "" + model_path,
        "--prompt",
        prompt,
        "--n_predict",
        str(n_predict),
        "--temp",
        str(temp),
        "--top_k",
        str(top_k),
        "--top_p",
        str(top_p),
        "--repeat_last_n",
        str(repeat_last_n),
        "--repeat_penalty",
        str(repeat_penalty),
        "--threads",
        "8",
    )
    print(args)
    procLlama = await asyncio.create_subprocess_exec(
        *args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    answer = ""

    while True:
        chunk = await procLlama.stdout.read(chunk_size)
        if not chunk:
            return_code = await procLlama.wait()

            if return_code != 0:
                error_output = await procLlama.stderr.read()
                raise ValueError(error_output.decode("utf-8"))
            else:
                return

        chunk = chunk.decode("utf-8")
        print(chunk, end="", flush=True)
        answer += chunk

        if prompt in answer:
            yield remove_matching_end(prompt, chunk)


class Camelid(BaseLLM, BaseModel):
    async def _agenerate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        response = ""
        async for token in load_model(prompt=prompts[0]):
            response += token
            self.callback_manager.on_llm_new_token(token, verbose=True)

        generations = [[Generation(text=response)]]
        return LLMResult(generations=generations)

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        result = None

        def run_coroutine_in_new_loop():
            nonlocal result
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(self._agenerate(prompts, stop))
            finally:
                new_loop.close()

        result_thread = threading.Thread(target=run_coroutine_in_new_loop)
        result_thread.start()
        result_thread.join()

        return result

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        result = self._generate([prompt], stop)
        return result.generations[0][0].text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}

    @property
    def _llm_type(self) -> str:
        return "llama"


if __name__ == "__main__":
    llm_predictor = LLMPredictor(llm=Camelid())
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

