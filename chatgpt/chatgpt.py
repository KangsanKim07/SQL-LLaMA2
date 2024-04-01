from openai import OpenAI
import json
from tqdm import tqdm

jf = json.load(open('/c1/kangsan/AI612/ehrsql-2024/data/mimic_iv/test/data.json'))['data']

OPENAI_API_KEY = None

client = OpenAI(api_key=OPENAI_API_KEY)
results = {}
for data in tqdm(jf):
    id = data['id']
    # question = data['question']
    question = "When does patient 8016's influenza quarantine end?"
    model = "gpt-4"
    messages = [{
        "role": "system",
        "content": "You are an accurate prompt verifier for a medical text-to-SQL model in a hospital. We have csv files in the hospital database, and their file names and column names of each table are as follows. \
            {admissions.csv: [subject_id,hadm_id,admittime,dischtime,admission_type,admission_location,discharge_location,insurance,language,marital_status,age], chartevents.csv: [subject_id,hadm_id,stay_id,charttime,itemid,valuenum,valueuom], \
            cost.csv: [subject_id,hadm_id,event_type,event_id,chargetime,cost], d_icd_diagnoses.csv: [icd_code,long_title], d_icd_procedures.csv: [icd_code,long_title], d_items.csv: [itemid,label,abbreviation,linksto], d_labitems.csv: [itemid,label], \
            diagnoses_icd.csv: [subject_id,hadm_id,icd_code,charttime], icustays.csv: [subject_id,hadm_id,stay_id,first_careunit,last_careunit,intime,outtime], inputevents.csv: [subject_id,hadm_id,stay_id,starttime,itemid,amount], \
            labevents.csv: [subject_id,hadm_id,itemid,charttime,valuenum,valueuom], microbiologyevents.csv: [subject_id,hadm_id,charttime,spec_type_desc,test_name,org_name], outputevents.csv: [subject_id,hadm_id,stay_id,charttime,itemid,value], patients.csv: [subject_id,gender,dob,dod], \
            prescriptions.csv: [subject_id,hadm_id,starttime,stoptime,drug,dose_val_rx,dose_unit_rx,route], procedures_icd.csv: [subject_id,hadm_id,icd_code,charttime], transfers.csv: [subject_id,hadm_id,transfer_id,eventtype,careunit,intime,outtime]}.\n \
            Please refer this instruction of those tables for your understanding. Tables are linked by identifiers which usually have the suffix id, subject_id refers to a unique patient, hadm_id refers to a unique admission to the hospital. Charted events such as notes, laboratory tests, and fluid balance are stored in a series of 'events' tables. For example the OUTPUTEVENTS table contains all measurements related to output for a given patient, while the LABEVENTS table contains laboratory test results for a patient. \
            Tables prefixed with 'd_' are dictionary tables and provide definitions for identifiers. For example, every row of CHARTEVENTS is associated with a single ITEMID which represents the concept measured, but it does not contain the actual name of the measurement. By joining CHARTEVENTS and D_ITEMS on ITEMID, it is possible to identify the concept represented by a given ITEMID. \
            Broadly speaking, five tables are used to define and track patient stays: ADMISSIONS; PATIENTS; ICUSTAYS; and TRANSFERS. Another five tables are dictionaries for cross-referencing codes against their respective definitions: D_ICD_DIAGNOSES; D_ICD_PROCEDURES; D_ITEMS; and D_LABITEMS. The remaining tables contain data associated with patient care, such as physiological measurements, caregiver observations, and billing information. \
            In some cases it would be possible to merge tables—for example, D_ICD_PROCEDURES contains detail relating to procedures and could be combined.\n \
            Your task is to verify a question from user that the question can be answered using SQL only with given dataset. If the answer of the question can be figured out within the given dataset, print yes. If the question can't be answered since the lack of information in the dataset, print no. You should also give some explanation why you think like that. \
            For example, if the question is 'What are the ways to take methimazole?', then we can find the answer by this SQL query(SELECT DISTINCT prescriptions.route FROM prescriptions WHERE prescriptions.drug = 'methimazole'). So the answer is yes. \
            But if the question is 'What is the emergency contact number for patient 273?', since the dataset does not contain patient's emergency contact information, the answer should be no. \n\
            EXAMPLES : ('Tell me the minimum respiratory rate in patient 10021118 in the first ICU visit.', 'yes'), ('What is the percentile of 4.46 for a red blood cells test given the same age as patient 10026406 on the first hospital stay?', 'yes'), ('How many patients were diagnosed with hyperlipidemia, unspecified during the same hospital encounter after the previous diagnosis with presence of cardiac pacemaker throughout this year?', 'yes'), ('Tell me the method of disinfection in a quarantine?', 'no'), ('How many artificial tear ointment prescription instances occurred?', 'yes'), ('When was patient 10019172 prescribed a medication via the po route the last time in 11/2100?', 'yes'), ('What is the number of operating rooms available at this time?', 'no'), ('Tell me the top four most common diagnoses for patients aged 30s.', 'yes'), ('Tell me the maximum total hospital cost that involves accidental poisoning from other specified plants in 2100.', 'yes'), ('Tell me the average o2 saturation pulseoxymetry of patient 10021938 on last month/14.', 'yes'), ('What was the name of the organism that was found in the last microbiology test on patient 10007795's abscess?', 'yes'), ('Does patient 65759 have a record of hypotension nec in his / her family?', 'no'), ('How many days have passed since the last time patient 10021487 received a lab test of calcium, urine during the current hospital encounter?', 'yes'), ('Did patient 10021938 receive any surgery on their last hospital encounter?', 'yes'), ('Since 04/04/2100, when did patient 10018501 have their last input?', 'yes'), ('When was the first time patient 10038999 had a o2 saturation pulseoxymetry that was greater than 96.0 since 02/02/2100?', 'yes'), ('What is the cost of introduction of nutritional substance into upper gi, via natural or artificial opening?', 'yes'), ('For the first time in 08/this year, how much scopolamine patch was prescribed to patient 10016742?', 'yes'), ('When was the first time patient 10004733, on this month/25, had the minimum heart rate?', 'yes'), ('I need the total volume of input for patient 10038999 on 06/03/2100.', 'yes'), ('When did patient 10038933 receive sputum microbiology test the first time since 03/2100?', 'yes'), ('When was patient 10005348 last blood culture microbiologic test since 07/2100?', 'yes'), ('When did patient 10031757 last get swab's microbiology test during their last hospital encounter?', 'yes'), ('Tell me the sum of the input volume for patient 10013049 since 05/12/2100.', 'yes'), ('What are the top four most common diagnoses that patients were diagnosed with within 2 months after receiving partial hepatectomy's this year?', 'yes'), ('Has patient 49176 ever experienced a side effect during a blood transfusion?', 'no'), ('How many pseudoephedrine prescription instances took place during the previous year?', 'yes'), ('How many cases involved the prescription of sulfameth/trimethoprim ss this year?', 'yes'), ('What percentile is the value of 30.0 in calculated total co2 among patients of the same age as patient 10038999 on their first hospital encounter?', 'yes'), ('Tell me the minimum total cost of a hospital that involved dysphagia, unspecified since 2100.', 'yes')\n \
            Remember, the answer should be yes or no. If the answer is yes, make SQL query for that. If the answer is no, explain it."
    }, {
        "role": "user",
        "content": question
    }]

    response = client.chat.completions.create(model=model, messages=messages)
    answer = response.choices[0].message.content
    results[id] = answer
    print(answer)
    break

# with open('gpt4_verify_0t100.json', 'w') as f:
#     json.dump(results, f) 