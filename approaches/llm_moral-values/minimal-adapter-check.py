

# put these *before* importing torch/transformers in a real script
import os; os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig, BitsAndBytesConfig

VALUE_DEFINITIONS = {
    "Self-direction: thought":      "Freedom to cultivate one’s own ideas and abilities",
    "Self-direction: action":       "Freedom to determine one’s own actions",
    "Stimulation":                  "Excitement, novelty, and change",
    "Hedonism":                     "Pleasure and sensuous gratification",
    "Achievement":                  "Success according to social standards",
    "Power: dominance":             "Power through exercising control over people",
    "Power: resources":             "Power through control of material and social resources",
    "Face":                         "Maintaining one’s public image and avoiding humiliation",
    "Security: personal":           "Safety in one’s immediate environment",
    "Security: societal":           "Safety and stability in the wider society",
    "Tradition":                    "Maintaining and preserving cultural, family, or religious traditions",
    "Conformity: rules":            "Compliance with rules, laws, and formal obligations",
    "Conformity: interpersonal":    "Avoidance of upsetting or harming other people",
    "Humility":                     "Recognising one’s insignificance in the larger scheme of things",
    "Benevolence: caring":          "Devotion to the welfare of in-group members",
    "Benevolence: dependability":   "Being a reliable and trustworthy member of the in-group",
    "Universalism: concern":        "Commitment to equality, justice, and protection for all people",
    "Universalism: nature":         "Preservation of the natural environment",
    "Universalism: tolerance":      "Acceptance and understanding of those who are different from oneself",
}

# Build one multiline definition block – used only in the “definition” prompt
DEFS_BLOCK = "\n".join(
    f"- **{k}**: {v}" for k, v in VALUE_DEFINITIONS.items()
)

adapters = "output/qlora-hier-gemma2-gate/qlora_output/lora_adapters"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,   # if this fails on your setup, retry float32
)

peft = AutoPeftModelForCausalLM.from_pretrained(
    adapters,
    device_map="auto",              # let Accelerate shard/offload
    quantization_config=bnb_cfg,    # <-- critical: load base in 4-bit
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    attn_implementation="eager",
)

tok = AutoTokenizer.from_pretrained(adapters)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token_id

# IMPORTANT: with device_map="auto", keep inputs on CPU and let Accelerate move them
p = (
    "### Value definitions\n"
    f"{DEFS_BLOCK}\n\n"
    "### Task\nIdentify which of the above values the SENTENCE relates to. "
    "Return **only** a JSON array of the matching value names.\n\n"
    "SENTENCE: Spain’s minister pleads for ‘millions and millions’ of immigrants\n"
    "OUTPUT: {"
)
ids = tok(p, return_tensors="pt")   # <-- stay on CPU

# Stop once we see the closing brace
from transformers import StoppingCriteria, StoppingCriteriaList
close_obj_ids = tok.encode("}", add_special_tokens=False)
class StopAtClosingBrace(StoppingCriteria):
    def __init__(self, pat): self.pat = pat
    def __call__(self, input_ids, scores, **kw):
        seq = input_ids[0].tolist(); L = len(self.pat)
        return L and len(seq) >= L and seq[-L:] == self.pat

gcfg = GenerationConfig(max_new_tokens=64, do_sample=False, use_cache=True,
                        pad_token_id=tok.eos_token_id, eos_token_id=None)

with torch.inference_mode():
    out = peft.generate(**ids, generation_config=gcfg,
                        stopping_criteria=StoppingCriteriaList([StopAtClosingBrace(close_obj_ids)]))

print(tok.decode(out[0], skip_special_tokens=True))