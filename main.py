import subprocess
from llama_index import PromptHelper, SimpleDirectoryReader
from llama_index import LLMPredictor, ServiceContext
from pathlib import Path

# define prompt helper
# set maximum input size
max_input_size = 2048
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

from llama_index import GPTListIndex, SimpleWebPageReader


def remove_matching_end(a, b):
    min_length = min(len(a), len(b))

    for i in range(min_length, 0, -1):
        if a[-i:] == b[:i]:
            return b[i:]

    return b

## todo make these args?
size = 30
modelName = f"{size}B/ggml-model-q4_0.bin".format(size)
camelid = "llama"
home = Path.home()
model = f"{home}/dalai/{camelid}/models/{modelName}".format(home, camelid, modelName)
modelExecutePath = f"{home}/dalai/{camelid}/main".format(home, camelid)


async def load_model(
        model: str = model,
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
        modelExecutePath,
        "--model",
        "" + model,
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


import asyncio
from typing import List, Optional, Mapping, Any
from langchain.llms.base import Generation, LLMResult, BaseLLM
from pydantic import BaseModel
import threading


class Llama(BaseLLM, BaseModel):
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
    llm_predictor = LLMPredictor(llm=Llama())


    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    # Load the your data
    #documents = SimpleWebPageReader(html_to_text=True).load_data(["https://github.com/jerryjliu/llama_index"])
    documents = SimpleDirectoryReader('./data').load_data()
    # documents = SimpleJsonReader()

    index = GPTListIndex.from_documents(documents, service_context=service_context)

    # Query and print response
    response = index.query("")
    print(response)