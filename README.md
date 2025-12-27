# Pothole Detection Project

This project trains a YOLOv5 model to detect potholes and provides a FastAPI service for inference.

## How to Run on a New Machine (Windows)

### 1. Prerequisites

*   Make sure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
*   Make sure you have Git installed. You can download it from [git-scm.com](https://git-scm.com/downloads).

### 2. Setup

1.  **Copy the Project:**
    Copy the entire project folder (the `harish` directory) to the Windows machine.

2.  **Open the Command Prompt:**
    Open the Command Prompt (or PowerShell) and navigate into the copied project folder.
    ```bash
    cd path/to/your/project
    ```

3.  **Install Dependencies:**
    Run the following commands to install the necessary Python packages:
    ```bash
    pip install -r yolov5/requirements.txt
    pip install fastapi uvicorn python-multipart onnxruntime
    ```

### 3. Running the Application

1.  **Start the Server:**
    Run the `app.py` script to start the FastAPI server.
    ```bash
    python app.py
    ```
    You should see a message indicating that the server is running on `http://127.0.0.1:8000`.

2.  **Test the Server:**
    Open a new Command Prompt window, navigate to the project folder again, and run the `measure_inference.py` script to send a test image to the server.
    ```bash
    python measure_inference.py
    ```
    You should see the inference time and a JSON response with the detected potholes.

### 4. Project Structure

*   `pothole.v18i.yolov5pytorch/`: The dataset used for training.
*   `yolov5/`: The YOLOv5 repository.
*   `app.py`: The FastAPI server for inference.
*   `measure_inference.py`: A script to test the server.
*   `split_dataset.py`: The script used to split the dataset.
*   `detail.txt`: The original task description.
