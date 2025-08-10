import random
import json
import os
import pandas as pd

list_model = os.listdir("../mlruns/models")
df = pd.DataFrame(list_model, columns=['model_name'])

# Assume the dataframe has a column named 'value' containing strings like the example
column_name = df.columns[0]  # Take the first column as the target if unspecified

# Fixed ky_du_lieu for all rows (could be dynamic if needed)
ky_du_lieu = "2025-07-10"

# Function to parse a value string into components
def parse_value(val):
    parts = val.split("_")
    ma_don_vi = parts[0]
    ma_bao_cao = parts[1]
    ma_tieu_chi = "_".join(parts[2:-1])
    fn_key = parts[-1]
    fn_value = round(random.uniform(1, 100), 2)
    return ma_don_vi, ma_bao_cao, ma_tieu_chi, fn_key, fn_value

# Dictionary to group by ma_don_vi and ma_bao_cao
grouped_data = {}

for val in df[column_name]:
    ma_don_vi, ma_bao_cao, ma_tieu_chi, fn_key, fn_value = parse_value(val)

    key = (ma_don_vi, ma_bao_cao)
    if key not in grouped_data:
        grouped_data[key] = {}
    if ma_tieu_chi not in grouped_data[key]:
        grouped_data[key][ma_tieu_chi] = {}
    grouped_data[key][ma_tieu_chi][fn_key] = fn_value

# Create JSON structure
result_json_list = []
for (ma_don_vi, ma_bao_cao), data_dict in grouped_data.items():
    data_list = []
    for ma_tieu_chi, fn_dict in data_dict.items():
        item = {"ma_tieu_chi": ma_tieu_chi}
        item.update(fn_dict)
        data_list.append(item)

    result_json = {
        "ma_don_vi": ma_don_vi,
        "ma_bao_cao": ma_bao_cao,
        "ky_du_lieu": ky_du_lieu,
        "data": data_list
    }
    result_json_list.append(result_json)

# Save the result to a JSON file
output_path = 'parsed_data.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result_json_list, f, ensure_ascii=False, indent=2)

output_path
