
import requests
import time

# Define the API endpoint
url = 'http://10.2.29.175:8000/predict/'

# Define the image path
image_path = 'images.jpeg'

# Read the image file
with open(image_path, 'rb') as f:
    files = {'file': f}
    
    # Measure the time taken for the request
    start_time = time.time()
    response = requests.post(url, files=files)
    end_time = time.time()

# Calculate and print the time taken
inference_time = end_time - start_time
print(f'Inference time: {inference_time:.4f} seconds')

# Print the response
print('Response:')
print(response.json())
