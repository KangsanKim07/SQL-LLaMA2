import os
import re
import json

tables = ['patients', 'admissions', 'd_icd_diagnoses', 'd_icd_procedures', 'd_labitems', 'd_items', 'diagnoses_icd', 'procedures_icd', 'labevents', 
          'prescriptions', 'cost', 'chartevents', 'inputevents', 'outputevents', 'microbiologyevents', 'icustays', 'transfers']

jf = json.load(open('valid_prediction.json'))
gpt = json.load(open('/c1/kangsan/AI612/SQL-LLaMA2/chatgpt/validset_verify.json'))

new_jf = {}
for key in jf.keys():
    value = jf[key]
    matches = matches = re.findall(r'FROM\s+([a-zA-Z_0-9]+)', value)
    if any(item not in tables for item in matches) or gpt[key] == 'no':
        print(key)
        print([x for x in matches if x not in tables])
        new_jf[key] = "null"
    else:
        new_jf[key] = value

with open('valid_prediction_filtered.json', 'w') as f:
    json.dump(new_jf, f)