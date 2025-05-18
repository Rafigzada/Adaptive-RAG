import os

def generate_folder_structure(root_dir):
    """
    Recursively generate the folder structure of the project.
    
    Args:
    - root_dir (str): The root directory of the project to scan.
    
    Returns:
    - str: A string representation of the folder structure.
    """
    folder_structure = []

    # Walk through the directory and list the files and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Calculate the depth of the current directory
        depth = dirpath.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * depth
        folder_structure.append(f"{indent}{os.path.basename(dirpath)}/")

        # Add files at the current level
        for filename in filenames:
            folder_structure.append(f"{indent}    {filename}")

    return "\n".join(folder_structure)

# Specify the root directory of your project
root_dir = os.getcwd()  # Get current working directory (you can specify the root project path)

# Generate the folder structure
folder_structure = generate_folder_structure(root_dir)

# Print the folder structure
print(folder_structure)

# Optionally, write the folder structure to a file
with open("folder_structure.txt", "w") as file:
    file.write(folder_structure)

print("\nFolder structure has been saved to folder_structure.txt")
