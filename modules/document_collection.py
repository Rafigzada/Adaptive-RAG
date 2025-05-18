






import os

def load_documents(source='default', folder_path=None):
    """
    Function to load documents. The source can be either 'default', a folder path, or a file path.

    Args:
    - source (str): The source from which to load the documents.
                    - 'default' loads a predefined set of documents.
                    - A folder path loads documents from all text files in the folder.
                    - A file path loads documents from a single file.

    Returns:
    - List of documents (str): List containing documents.
    """

    documents = []

    if source == 'default':
        # Default documents collection
        documents = [
            "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary.",
            "Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\".",
            "Next to the Main Building is the Basilica of the Sacred Heart.",
            "Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection.",
            "It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858.",
            "At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary."
        ]

    elif folder_path:
        # Load documents from all text files in a folder
        if not os.path.isdir(folder_path):
            raise ValueError(f"The specified folder path {folder_path} does not exist.")

        for filename in os.listdir(folder_path):
            # Only process .txt files
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents.append(file.read().strip())

    elif os.path.isfile(source):
        # Load documents from a single file
        with open(source, 'r', encoding='utf-8') as file:
            documents = [line.strip() for line in file.readlines()]

    else:
        raise ValueError(f"Invalid source: {source}. Provide a valid folder path or file path, or use 'default'.")

    print(f"Loaded {len(documents)} documents")
    return documents


documents = load_documents()
print(documents)







