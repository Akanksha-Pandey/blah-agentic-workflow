# blah-workflow

Setting up local environment
```
python3 -m pip install --upgrade pip
python3 -m venv .venv
source .venv/bin/activate
```

Install Ollama

```
brew install ollama
ollama serve
```

Install models on Ollama
```
ollama pull Qwen2.5:7B-Instruct
ollama pull qwen3:8b  
ollama pull llama3.1:8b  

ollama list
ollama run qwen3:8b
```

Interact with LLM
```
Hello, who are you?
What can you do?
Do you have any tools?
```

