import os
import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

DATASET_PATH = "dataset/Weather_Dataset"
FEATURE_FILE = "features_output/features.npy"
PATH_FILE    = "features_output/image_paths.npy"


# image loader
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.resize(img, (224, 224))


# ORB feature extraction
def extract_orb(image):
    orb  = cv2.ORB_create(nfeatures=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None:
        return np.zeros(32, dtype=np.float32)

    return descriptors.astype(np.float32).mean(axis=0)


# SIFT feature extraction
def extract_sift(image):
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None:
        return np.zeros(128, dtype=np.float32)

    return descriptors.astype(np.float32).mean(axis=0)


# color histogram (BGR + HSV)
def extract_color_hist(image):
    # BGR histogram
    hist_bgr = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_bgr = cv2.normalize(hist_bgr, hist_bgr).flatten().astype(np.float32)

    # HSV histogram
    hsv      = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten().astype(np.float32)

    return np.concatenate([hist_bgr, hist_hsv])  # 512 + 512 = 1024-dim


# feature pipeline
def extract_features(image):
    orb_features   = extract_orb(image)
    sift_features  = extract_sift(image)
    color_features = extract_color_hist(image)

    return np.concatenate([orb_features, sift_features, color_features])


# process single image
def process_image(path):
    img = load_image(path)

    if img is None:
        return None

    return {
        "path": path,
        "features": extract_features(img)
    }


# collect images
def collect_images(dataset_path):
    image_paths = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))

    return image_paths


# build parallel database
def build_feature_database():
    start = time.time()

    print("Scanning dataset.")
    image_paths = collect_images(DATASET_PATH)
    print("Total images:", len(image_paths))

    print("Starting parallel feature extraction.")

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for r in tqdm(executor.map(process_image, image_paths), total=len(image_paths)):
            if r is not None:
                results.append(r)

    features = np.array([r["features"] for r in results], dtype=np.float32)
    paths    = [r["path"] for r in results]

    os.makedirs("features_output", exist_ok=True)
    np.save(FEATURE_FILE, features)
    np.save(PATH_FILE, paths)

    end = time.time()

    print("Feature extraction complete")
    print("Feature matrix shape:", features.shape)
    print("Execution time:", round(end - start, 2), "seconds")


# cosine similarity
def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return np.dot(a, b) / denom


# search similar images
def search_similar(query_image, top_k=5):
    features = np.load(FEATURE_FILE)
    paths    = np.load(PATH_FILE, allow_pickle=True)

    img = load_image(query_image)
    if img is None:
        print("Could not load query image:", query_image)
        return

    query_features = extract_features(img)

    similarities = []

    for i, f in enumerate(features):
        if paths[i] == query_image:
            continue
        sim = cosine_similarity(query_features, f)
        similarities.append((sim, paths[i]))

    similarities.sort(key=lambda x: x[0], reverse=True)

    print("\nTop", top_k, "similar images:\n")

    for score, path in similarities[:top_k]:
        print(round(score, 4), path)


def main():
    print("1: Build feature database")
    print("2: Search similar images")

    choice = input("Select option: ")

    if choice == "1":
        build_feature_database()

    elif choice == "2":
        query = input("Enter query image path: ")

        start = time.time()
        search_similar(query)
        end = time.time()

        print("\nSearch time:", round(end - start, 4), "seconds")

    else:
        print("Invalid option")


if __name__ == "__main__":
    main()