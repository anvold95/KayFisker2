"""
Replicate Predictor for Kay Fisker Mistral+LoRA model
=====================================================
Denne filen definerer hvordan modellen kjører på Replicate.
"""

import torch
from cog import BasePredictor, Input
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login
import os

# Dine modell-IDs
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
LORA_ADAPTER = "anvold/fisker-lora-clean"

class Predictor(BasePredictor):
    def setup(self):
        """
        Lastes én gang når modellen starter på Replicate.
        Merger LoRA med base-modellen for raskere inference.
        """
        # Login til HuggingFace hvis token er satt
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        print("Laster tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Laster base-modell...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        print("Laster og merger LoRA-adapter...")
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
        self.model = model.merge_and_unload()
        self.model.eval()

        print("Modell klar!")

    def predict(
        self,
        prompt: str = Input(description="Full prompt inkludert system message og kilder"),
        max_new_tokens: int = Input(description="Maks antall tokens å generere", default=400, ge=50, le=1000),
        temperature: float = Input(description="Sampling temperature", default=0.40, ge=0.1, le=1.5),
        top_p: float = Input(description="Top-p sampling", default=0.88, ge=0.1, le=1.0),
        top_k: int = Input(description="Top-k sampling", default=45, ge=1, le=100),
        repetition_penalty: float = Input(description="Repetition penalty", default=1.25, ge=1.0, le=2.0),
    ) -> str:
        """
        Genererer tekst basert på prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Returner kun den genererte delen (etter prompten)
        if prompt in generated:
            generated = generated[len(prompt):].strip()

        return generated
