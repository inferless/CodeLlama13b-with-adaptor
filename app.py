import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

class InferlessPythonModel:
    def initialize(self):
        model_id = "codellama/CodeLlama-13b-Instruct-hf"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        config = PeftConfig.from_pretrained("G-ML-Hyly/stg-cli13b-t6-cdp-ca.mt.him.cln.inter-b4s1e1-20231219-1611")
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model = PeftModel.from_pretrained(self.model, "G-ML-Hyly/stg-cli13b-t6-cdp-ca.mt.him.cln.inter-b4s1e1-20231219-1611")
        self.model.to("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-13b-Instruct-hf")
    
    def infer(self, inputs):
        prompt = inputs["prompt"]
        batch = self.tokenizer(prompt, return_tensors='pt').to("cuda:0")
        output_tokens = self.model.generate(**batch, max_new_tokens=50)
        output = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return {"generated_output": output}

    def finalize(self,args):
        self.model = None
        self.tokenizer = None
