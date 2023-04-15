# 🐪 camelid-porch 🦙 

![](./camelid.jpeg)

Python scripts for using custom LLMs with Langchain Alpaca or Llama model using [llama-index](https://github.com/jerryjliu/llama_index) and [langchain](https://github.com/hwchase17/langchain) and [BabyAGI]().

## Setup

### Install models from [Dalai](https://github.com/cocktailpeanut/dalai):

```
npx dalai llama install 7B
```
  
#### Set model size in script (optional)

### Install requirements:

```
pip install -r requirements.txt
```

### Run it:
```
python main.py --path ./data --prompt "Tell me if an API key required to use camelid-porch"
```

```
Answer: No it is not necessary
```

## Commands

### Index a list of urls from a file:

```
python main.py --urls urls.txt
```

### Index a list of files from a directory:

```
python main.py --path ./data
```

### Prompt after indexing:

```
python main.py --path ./data --prompt "Where was this alpaca born?"
```

### Save to index.json after indexing:

```
python main.py --save index.json
```


### Chain prompts 
From https://github.com/ai8hyf/babyagi/commit/6fcd528a92c80846dbb351f2b7babdd50c38709d

```
python babyagi.py --prompt "Plan me a 1 week trip to Hawaii"
```

## Related links
* [Gist adding Python LLM Class](https://gist.github.com/lukestanley/6517823485f88a40a09979c1a19561ce_)
* [Langchain issue [#1777]](https://github.com/hwchase17/langchain/issues/1777)
* [llama confusion](https://github.com/yoheinakajima/babyagi/issues/130)