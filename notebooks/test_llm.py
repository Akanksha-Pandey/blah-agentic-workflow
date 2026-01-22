from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="qwen3:8b")

response = llm.invoke("Say hello in one sentence.")
print(response)

assert "hello" in response.lower()