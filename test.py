import transformers
import json
from tqdm import tqdm
import torch

jf = json.load(open("/c1/kangsan/AI612/ehrsql-2024/data/mimic_iv/valid/data.json"))
model = transformers.AutoModelForCausalLM.from_pretrained("/c1/kangsan/AI612/SQL-LLaMA2/output", device_map="auto")
tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/c1/kangsan/AI612/SQL-LLaMA2/output",
        padding_side="right",
        use_fast=False,
    )
model.eval()
output = {}

for data in tqdm(jf['data']):
    id = data['id']
    question = data['question']
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:"
    input = tokenizer(prompt, return_tensors='pt').to("cuda")

    with torch.no_grad():
        preds = model.generate(input.input_ids, max_length=1024)
    preds = tokenizer.batch_decode(preds)[0]
    output[id] = preds.split('### Response:')[-1].replace('</s>', '')
    
with open('valid_prediction.json', 'w') as f:
    json.dump(output, f)
    
'''
Tell me the name of the organism that was detected in the last urine test of patient 10027602 on their first hospital visit?
SELECT microbiologyevents.org_name FROM microbiologyevents WHERE microbiologyevents.hadm_id IN ( SELECT admissions.hadm_id FROM admissions WHERE admissions.subject_id = 10027602 AND admissions.dischtime IS NOT NULL ORDER BY admissions.admittime ASC LIMIT 1 ) AND microbiologyevents.spec_type_desc = 'urine' AND microbiologyevents.org_name IS NOT NULL ORDER BY microbiologyevents.charttime DESC LIMIT 1"
SELECT microbiologyevents.org_name FROM microbiologyevents WHERE microbiologyevents.hadm_id IN ( SELECT admissions.hadm_id FROM admissions WHERE admissions.subject_id = 10027602 AND admissions.dischtime IS NOT NULL ORDER BY admissions.admittime ASC LIMIT 1 ) AND microbiologyevents.spec_type_desc = 'urine' AND microbiologyevents.org_name IS NOT NULL ORDER BY microbiologyevents.charttime DESC LIMIT 1
SELECT microbiologyevents.org_name FROM microbiologyevents WHERE microbiologyevents.hadm_id IN ( SELECT admissions.hadm_id FROM admissions WHERE admissions.subject_id = 10027602 AND admissions.dischtime IS NOT NULL ORDER BY admissions.admittime ASC LIMIT 1 ) AND microbiologyevents.spec_type_desc = 'urine' AND microbiologyevents.org_name IS NOT NULL ORDER BY microbiologyevents.charttime DESC LIMIT 1
'''