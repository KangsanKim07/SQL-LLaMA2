import openai
import json
openai.api_key = None

datas = json.load(open('/c1/kangsan/AI612/ehrsql-2024/data/mimic_iv/valid/data.json'))['data']
result = {}
for data in datas:
    id = data['id']
    question = data['question']
    completion = openai.Completion.create(
    model="ft:babbage-002:personal::97J0Xki7",
    prompt=f"{question} ->",
    max_tokens=1,
    n=1,
    stop=None,
    temperature=0.7
    )
    
    answer = completion.choices[0]['text'].strip()
    result[id] = answer

with open('validset_verify.json', 'w') as f:
    json.dump(result, f)