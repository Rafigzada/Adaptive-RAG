# ARAG Project

## Overview

ARAG is a project that implements a Retrieval-Augmented Generation (RAG) pipeline using document embeddings, retrieval systems, context integration, and text generation. The system is based on a modular architecture that includes several components to handle document processing, embeddings, retrieval, and generation of answers using models like GPT-4.

This repository includes:

- Jupyter notebooks for exploration and experimentation.
- Python scripts for various components of the RAG pipeline.
- Scripts for setup and deployment.

## Requirements

To run the project locally, you need to set up the environment with all the necessary dependencies.

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
   git clone https://git.cs.uni-paderborn.de/ssahoo/advanced-rag.git
   cd advanced-rag
   ```

### Step 7: Set Up the `.env` File for API Key Management

To securely manage your API key (e.g., for Gemini), use a `.env` file in your project directory.

#### Steps to Create and Use `.env`

1. **Create a `.env` file** in your project root:

   ```bash
   touch .env
    ```

2. **Add your API key** to the `.env` file (replace with your actual key):

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Install the required library, if you haven’t already** to load `.env` files:

   ```bash
   pip install python-dotenv
   ```

4. **Prevent `.env` from being committed to Git** by adding it to `.gitignore`:

   ```bash
   echo ".env" >> .gitignore
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

This ensures that any packages you install or scripts you run will use the correct Python environment.


### 10. Install Required Python Packages

Once the environment is activated, install the required Python packages by running:

```bash
pip install -r requirements.txt
```

> ⚠️ **Important:** Make sure you have activated the environment (`rag_env`) before running this command.
> Installing packages **without activation** may install them globally or in the wrong environment, which can lead to conflicts or import errors.

You can confirm the active environment by running:

```bash
which python
```

It should return a path like:

```
/path/to/anaconda3/envs/rag_env/bin/python
```

This confirms you're installing into the correct environment.


### Step 11: Display Hidden Files like `.env` and `.gitignore` in JupyterLab

By default, hidden files and folders (those starting with a dot, e.g., `.env`, `.gitignore`) are not visible in JupyterLab. You can enable this by following these steps:

---

#### Steps to Enable Viewing Hidden Files in JupyterLab

1. **Check Jupyter configuration paths:**

   Open your terminal and run:

   ```bash
   jupyter --paths
    ````

You'll see output like this:

```
config:
    /Users/your-username/.jupyter
    /usr/local/etc/jupyter
    /etc/jupyter
data:
    /Users/your-username/Library/Jupyter
    /usr/local/share/jupyter
    /usr/share/jupyter
runtime:
    /Users/username/Library/Jupyter/runtime
    ...
```

Take note of the `config` path — particularly the one in your home directory (e.g., `/Users/your-username/.jupyter`).

2. **Generate a Jupyter server config file** (if it doesn't exist):

   ```bash
   jupyter server --generate-config
   ```

   This creates a file called:

   ```bash
   ~/.jupyter/jupyter_server_config.py
   ```

3. **Edit the config file to allow hidden files:**

   Open `jupyter_server_config.py` in a text editor and find the following line:

   ```python
   # c.ContentsManager.allow_hidden = False
   ```

   Uncomment it and change the value to `True`:

   ```python
   c.ContentsManager.allow_hidden = True
   ```

   Save and close the file.

4. **Restart JupyterLab**, and open the file browser.

5. **Enable hidden file view in JupyterLab UI**:

   * Go to the **menu bar**: `View → Show Hidden Files`
   * You should now see `.env`, `.gitignore`, and any other hidden files in the file browser


### 12. Running JupyterLab

After installing the necessary dependencies, you can start JupyterLab with:

```bash
jupyter lab
```

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```