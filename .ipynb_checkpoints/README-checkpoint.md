# ARAG Project

## Overview

ARAG is a project that implements a Retrieval-Augmented Generation (RAG) pipeline using document embeddings, retrieval systems, context integration, and text generation. The system is based on a modular architecture that includes several components to handle document processing, embeddings, retrieval, and generation of answers using models like GPT-4.

This repository includes:

- Jupyter notebooks for exploration and experimentation.
- Python scripts for various components of the RAG pipeline.
- Scripts for setup and deployment.

## Requirements

To run the project locally or on a teammate's machine, you need to set up the environment with all the necessary dependencies.

### 1. Install Anaconda Navigator

First, install **Anaconda Navigator** if it's not already installed. Anaconda Navigator is a graphical user interface (GUI) for managing conda environments and packages. You can download it from [here](https://www.anaconda.com/products/distribution).

### 2. Create a New Environment

Once Anaconda Navigator is installed:

1. Launch **Anaconda Navigator**.
2. In the Anaconda Navigator interface, navigate to the **Environments** tab.
3. Click the **Create** button to create a new environment.
4. Choose the Python version (preferably Python 3.x) and click **Create**.

### 3. Install JupyterLab in the New Environment

After creating the environment:

1. Select your newly created environment from the **Environments** tab.
2. Click the **Open Terminal** button in the environment window.
3. In the terminal, run the following command to install JupyterLab:
   ```bash
   conda install -c conda-forge jupyterlab

### 4. Launch JupyterLab

Once JupyterLab is installed:

1. In the terminal, run the following command to launch JupyterLab:

   ```bash
   jupyter lab
   ```
2. JupyterLab will open in your web browser.

### 5. Install Git

Git is required to clone the project repository. To install Git in the newly created environment:

1. Open the terminal in Anaconda Navigator.
2. Run the following command to install Git:

   ```bash
   conda install git
   ```

### 6. Clone the Project Repository

Once Git is installed:

1. Open the terminal in Anaconda Navigator.
2. Navigate to the directory where you want to clone the repository.
3. Run the following command to clone the project:

   ```bash
   git clone https://your-repository-url.git
   cd ARAG
   ```

### 7. Set Up the Environment from `environment.yaml`

You can set up the environment from the `environment.yaml` file or update the newly created environment with it.

1. In the terminal, while in the project folder (`ARAG`), run the following command to create the environment from the `environment.yaml` file:

   ```bash
   conda env create -f environment.yaml
   ```
2. If you want to **update** the environment instead of creating a new one, use:

   ```bash
   conda env update -f environment.yaml --prune
   ```

### 8. Create a Kernel for the New Environment

To use your newly created environment in Jupyter Notebooks, you'll need to create a Jupyter kernel associated with it.

1. First, activate the environment:

   ```bash
   conda activate rag_env
   ```
2. Then install the `ipykernel` package:

   ```bash
   conda install ipykernel
   ```
3. Finally, create the Jupyter kernel:

   ```bash
   python -m ipykernel install --user --name rag_env --display-name "Python (rag_env)"
   ```

After running this command, you will be able to select the `rag_env` kernel when opening notebooks in JupyterLab.

### 9. Activate the Environment in the Terminal

Whenever you need to work in the environment, you can activate it in the terminal using the following command:

```bash
conda activate rag_env
```

Once activated, you can install any additional dependencies or run Python scripts inside this environment.

---

### 10. Install Required Python Packages

After setting up the environment, you need to install the project dependencies. You can install the necessary packages by running:

```bash
pip install -r requirements.txt
```

Or if you want to install them via Conda, you can use the `environment.yaml` file:

```bash
conda env update -f environment.yaml
```

### 11. Running JupyterLab

After installing the necessary dependencies, you can start JupyterLab with:

```bash
jupyter lab
```

### 12. Running the Notebooks

* Open the notebooks in the `notebooks/` directory in JupyterLab.
* Start with `1_document_processing.ipynb` to preprocess the documents and follow the pipeline through to `6_rag_pipeline.ipynb` for the complete RAG pipeline execution.

## Files in the Project

* `pyproject.toml`: Contains the project's metadata and dependencies managed by Poetry.
* `environment.yaml`: Conda environment setup for the project.
* `requirements.txt`: Basic dependencies for the project.
* `install.txt`: A guide for project installation and setup.
* `deployment.txt`: Deployment instructions for setting up the project in production environments.
* `generate_folder_structure.py`: A script to generate the folder structure of your project.
* `tests/`: Unit tests for the project.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```