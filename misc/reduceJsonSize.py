import json


def split_in_files_codesearchnet(json_data, amount):
    step = len(json_data) // amount
    pos = 0
    for i in range(amount - 1):
        pos += step
    # last one
    with open('train_codesearchnet_7_short.json'.format(amount), 'w') as file:
        json.dump(json_data[pos:], file)


f = open("../data/train_codesearchnet_7.json")
json_data = json.load(f)
split_in_files_codesearchnet(json_data, 12)
