import os

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    directory_path = os.getcwd() + "/images"
    for file_name in os.listdir(path=directory_path):
        new_file_name = file_name.split(".")[1].lower()
        os.rename(directory_path + "/" + file_name, directory_path + "/" + new_file_name)
        print("Renamed " + file_name + " to " + new_file_name)
