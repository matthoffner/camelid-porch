# 🐪 camelid-porch 🦙 

![](./camelid.jpeg)

A Python script to load various data sources into a custom Alpaca or Llama model using [llama-index](https://github.com/jerryjliu/llama_index) and [langchain](https://github.com/hwchase17/langchain).

## List of urls using SimpleWebPageReader

```
python main.py --urls urls.txt
```

## Directory using SimpleDirectoryReader

```
python main.py --path ./data
```

## Prompt after indexing

```
python main.py --prompt "Where was this alpaca born?"
```

## Save to index.json

```
python main.py --save index.json
```


* [Gist adding Python LLM Class](https://gist.github.com/lukestanley/6517823485f88a40a09979c1a19561ce_)
* [Langchain issue [#1777]](https://github.com/hwchase17/langchain/issues/1777)
* [Dalai](https://github.com/cocktailpeanut/dalai)
