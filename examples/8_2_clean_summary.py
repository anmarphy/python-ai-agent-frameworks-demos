import json

# Read the JSON file
with open('results/extracted_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

if isinstance(data, str):
    data = json.loads(data)

# Extract patient name and date
patient_name = data.get('patient')
date = data.get('date')

# Find exam results that are not in range
not_in_range = [
    f"- {exam['exam']}: Value {exam['value']} (Range: {exam['ranges']})"
    for exam in data.get('result', [])
    if exam.get('value_in_range', '').lower() == 'false'
]

print("• Patient name:", patient_name)
print("• Date:", date)
print("• Exam results not in range:")
if not_in_range:
    for item in not_in_range:
        print(item)
else:
    print("- All results are within range.")

