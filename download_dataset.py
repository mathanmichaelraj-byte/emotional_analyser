import kagglehub

# Download latest version
path = kagglehub.dataset_download("shivamb/go-emotions-google-emotions-dataset")

print("Path to dataset files:", path)