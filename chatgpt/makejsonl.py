import json

datas = json.load(open("/c1/kangsan/AI612/ehrsql-2024/data/mimic_iv/train/data.json"))['data']
labels = json.load(open("/c1/kangsan/AI612/ehrsql-2024/data/mimic_iv/train/label.json"))

results = []

for data in datas:
    id = data['id']
    question = data['question']
    answer = "yes" if labels[id].strip() != 'null' else "no"
    query = {"prompt": question, "completion": answer}
    results.append(query)

with open("valid.jsonl" , encoding= "utf-8",mode="w") as file: 
	for i in results: file.write(json.dumps(i) + "\n")