from llama_cpp import Llama
import json
from tqdm import tqdm
from multiprocessing import Process

jf = json.load(open("/c1/kangsan/AI612/ehrsql-2024/data/mimic_iv/test/data.json"))
llm = Llama(model_path="/c1/kangsan/AI612/SQL-LLaMA2/models_hf/gguf-model-f32.gguf")
datas = jf['data'][:50]
output = {}

def run(inputs):
    for data in tqdm(inputs):
        id = data['id']
        question = data['question']
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:"
        
        pred = llm(prompt, max_tokens=1024, stop=["Output"], echo=True)
        output['id'] = pred['choices'][0]['text'].split('### Response:')[-1]

# processes = []
# pnum = 3
# d = len(datas) // pnum
# for i in range(pnum):
#     p = Process(target=run, args=(datas[i*d:(i+1)*d],))
#     processes.append(p)
#     p.start()
# p = Process(target=run, args=(datas[pnum*d:],))
# processes.append(p)
# p.start()

# for p in processes:
#     p.join()

run(datas)

with open('test_pred.json', 'w') as f:
    json.dump(output, f)
    
# "prompt_no_input": (
#         "Below is an instruction that describes a task. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Response:"
#     ),

'''
Tell me the name of the organism that was detected in the last urine test of patient 10027602 on their first hospital visit?
SELECT microbiologyevents.org_name FROM microbiologyevents WHERE microbiologyevents.hadm_id IN ( SELECT admissions.hadm_id FROM admissions WHERE admissions.subject_id = 10027602 AND admissions.dischtime IS NOT NULL ORDER BY admissions.admittime ASC LIMIT 1 ) AND microbiologyevents.spec_type_desc = 'urine' AND microbiologyevents.org_name IS NOT NULL ORDER BY microbiologyevents.charttime DESC LIMIT 1"
'''