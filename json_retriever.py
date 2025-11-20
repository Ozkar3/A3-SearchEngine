import json

def main():
    with open("inverted_index.json") as json_file:
        index_data = json.load(json_file)
        print(index_data["cristina"])

if __name__ == "__main__":
    main()