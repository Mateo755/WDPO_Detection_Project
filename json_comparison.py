import json

def compare_json_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    mismatch_count = 0

    for key in data1:
        if key in data2:
            list1 = list(data1[key].values())
            list2 = list(data2[key].values())

            if list1 != list2:
                mismatch_count += 1
                print(f"Mismatch in {key}: Result:{list1} vs Reference:{list2}")

    return mismatch_count

if __name__ == "__main__":
    file_path_template = "data/train.json"
    file_path_result = "results.json"

    mismatches = compare_json_files(file_path_result, file_path_template)

    print(f"Number of mismatches: {mismatches}")