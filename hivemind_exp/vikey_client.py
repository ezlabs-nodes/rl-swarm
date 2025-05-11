import requests
import logging

class VikeyAdapter:
    def __init__(self, endpoint: str, model_id: str):
        self.endpoint = endpoint
        self.model_id = model_id
        self.session = requests.Session()
        self.session.timeout = 60

    def generate(self, prompt: str, **kwargs) -> str:
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_length", 256),
            "temperature": kwargs.get("temperature", 0.7)
        }
        try:
            response = self.session.post(
                f"{self.endpoint}/completions",
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["text"]
        except Exception as e:
            logging.error(f"Vikey error: {str(e)}")
            raise
