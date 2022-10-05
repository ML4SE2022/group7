import os
import json
import time


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class DupesRemover:
    def __init__(self):
        pass

    def convert_file_to_json_object(self, file_location):
        print(BColors.HEADER + "converting \"" + str(file_location) + "\" to json object")

        data_file = open(file_location)

        data_file.seek(0, os.SEEK_END)
        file_size = data_file.tell()
        data_file.seek(0)
        if file_size >= 100000000:
            print(BColors.WARNING + "WARNING: \"" + str(
                file_location) + "\" is a big file, so it may take a while" + BColors.ENDC)

        time_start = time.time()
        json_data = json.load(data_file)
        time_end = time.time()
        print(BColors.ENDC + "converted file in " + str(round(time_end - time_start, 2)) + " seconds")

        return json_data

    def remove_dupes_in_json_data(self, json_data):
        print(BColors.HEADER + "removing duplicates")
        hash_table = dict()
        duplicates_count = 0
        unique_count = 0

        unique_entries = []

        for i in range(len(json_data)):
            if json_data[i]["doc"] in hash_table.keys():
                duplicates_count += 1
            else:
                hash_table[json_data[i]["doc"]] = 1
                unique_entries.append(json_data[i])
                unique_count += 1

        print(BColors.ENDC + str(duplicates_count) + " duplicates removed")
        print(BColors.ENDC + str(unique_count) + " unique entries left")
        print(BColors.ENDC + "Fraction removed is " + str(duplicates_count / (duplicates_count + unique_count)))

        return unique_entries

    def print_data_to_file(self, data, file_name):
        with open(file_name, "w") as outfile:
            json.dump(data, outfile)

    def remove_dupes(self, file_name, file_destination):
        json_data = self.convert_file_to_json_object(file_name)
        unique_entries = self.remove_dupes_in_json_data(json_data)
        self.print_data_to_file(unique_entries, file_destination)


dupes_remover = DupesRemover()
# json_data = dupes_remover.remove_dupes("./test_webquery.json", "./test_webquery_unique.json")
dupes_remover.remove_dupes("./train_codesearchnet_7.json", "./train_codesearchnet_7_unique.json")
