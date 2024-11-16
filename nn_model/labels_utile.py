import os

# Variable to store the computed label
cached_labels_file = os.path.join(os.path.dirname(__file__), "cached_labels.txt")
cached_labels = None
# cached_labels_file = "cached_labels.txt"
# cached_labels = None

def get_labels():
    global cached_labels

    if cached_labels is None:
        cached_labels = load_labels()
        if cached_labels is None:
            raise Exception("cached_labels is None!")

    return cached_labels

def set_labels(labels):
    global cached_labels

    cached_labels = labels
    # print("cached_labels=", cached_labels)

    save_labels()
    
    # Print the updated labels
    # print("Updated cached_labels:", cached_labels)
    # Return the updated labels
    return cached_labels

# def set_labels(labels):
#     global cached_labels

#     cached_labels = labels
#     save_labels()

def save_labels():
    global cached_labels

    with open(cached_labels_file, "w") as file:
        file.write("\n".join(cached_labels))

def load_labels():
    if os.path.exists(cached_labels_file):
        with open(cached_labels_file, "r") as file:
            return file.read().splitlines()
    else:
        return None

