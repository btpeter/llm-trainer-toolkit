import torch
from peft import PeftModel
from transformers import(
    AutoModelForCausalLM, 
    AutoTokenizer
)


def apply_lora(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
 
    print(f"Loading the LoRA adapter from {lora_path}")
 
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    )
 
    print("Applying the LoRA")
    model = lora_model.merge_and_unload()
 
    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    base_model_path = "/data2/model_file/pretrained-checkpoint/Baichuan2-7B-Chat"
    output_path = "/data2/home/yangbt/projects/llm-chat-finetune/merge_models/baichuan2-zhuiwen-multiple-v4"
    lora_path = "/data2/home/yangbt/projects/llm-chat-finetune/baichuan2-chat-zhuiwen-multiple-v4/checkpoint-1200"

    apply_lora(base_model_path, output_path, lora_path)
