import json

datas = json.load(open('/c1/kangsan/AI612/ehrsql-2024/data/mimic_iv/train/data.json'))
datas = datas['data']
labels = json.load(open('/c1/kangsan/AI612/ehrsql-2024/data/mimic_iv/train/label.json'))

output = []

for data in datas:
    id = data['id']
    val = {}
    val['instruction'] = data['question'].strip()
    val['output'] = labels[id].strip()
    # if val['output'] == 'null':
    #     continue
    output.append(val)
    
with open('ehrsql24_train_withnull.json', 'w') as f:
    json.dump(output, f)