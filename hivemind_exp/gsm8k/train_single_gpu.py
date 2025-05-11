import logging
import requests
from typing import Optional, Dict, Any

# Needs to be before trl!
from hivemind_exp.runner.grpo_runner import GRPOArguments, GRPORunner

import colorlog
from trl import GRPOConfig, ModelConfig, TrlParser

from hivemind_exp.chain_utils import (
    ModalSwarmCoordinator,
    WalletSwarmCoordinator,
    setup_web3,
)
from hivemind_exp.gsm8k.generate_prompts import get_stage1_samples as gsm8k_stage1_samples
from hivemind_exp.dapo.generate_prompts import get_stage1_samples as dapo_stage1_samples
from hivemind_exp.runner.gensyn.testnet_grpo_runner import (
    TestnetGRPOArguments,
    TestnetGRPORunner,
)

class VikeyModelAdapter:
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
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9)
        }
        
        try:
            response = self.session.post(
                f"{self.endpoint}/completions",
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["text"]
        except requests.exceptions.RequestException as e:
            logging.error(f"Vikey API request failed: {str(e)}")
            raise
        except (KeyError, IndexError) as e:
            logging.error("Invalid response format from Vikey")
            raise ValueError("Invalid Vikey response") from e

def main():
    # Setup logging.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(green)s%(levelname)s:%(name)s:%(message)s")
    )
    root_logger.addHandler(handler)

    parser = TrlParser((ModelConfig, GRPOArguments, TestnetGRPOArguments, GRPOConfig))  # type: ignore
    model_args, grpo_args, testnet_args, training_args = parser.parse_args_and_config()

    # Initialize runner with Vikey adapter
    contract_address = testnet_args.contract_address
    if org_id := testnet_args.modal_org_id:
        assert contract_address, "Contract address must be set!"
        runner = TestnetGRPORunner(
            ModalSwarmCoordinator(setup_web3(), contract_address, org_id)
        )
    elif priv_key := testnet_args.wallet_private_key:
        assert contract_address, "Contract address must be set!"
        runner = TestnetGRPORunner(
            WalletSwarmCoordinator(setup_web3(), contract_address, priv_key)
        )
    else:
        runner = GRPORunner()
        # Inject Vikey adapter for local inference
        runner.model = VikeyModelAdapter(
            endpoint="http://152.42.189.162:14444/v1",
            model_id="qwen2.5-0.5b-instruct"
        )

    # Start training
    game = grpo_args.game
    try:
        match game:
            case "gsm8k":
                runner.run(model_args, grpo_args, training_args, gsm8k_stage1_samples)
            case "dapo":
                runner.run(model_args, grpo_args, training_args, dapo_stage1_samples)
            case _:
                raise ValueError(f"Unknown game: {game}")
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
