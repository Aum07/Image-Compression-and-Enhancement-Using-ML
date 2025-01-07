import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed
import io

# Parallelize the computation of the closest centroids
def compute_closest_centroids_parallel(X, centroids):
    def compute_distance(i):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        return np.argmin(distances)

    idx = Parallel(n_jobs=-1)(delayed(compute_distance)(i) for i in range(X.shape[0]))
    return np.array(idx)

# K-Means Function with parallelization for the closest centroids computation
def run_kMeans(X, K, max_iters=10, batch_size=100):
    kmeans = MiniBatchKMeans(n_clusters=K, max_iter=max_iters, batch_size=batch_size, n_init=3)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_

    # Compute closest centroids using parallelization
    idx = compute_closest_centroids_parallel(X, centroids)

    return centroids, idx

# Streamlit UI
st.title('K-Means Image Compression')

# Space for uploading the image
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

# Explanation text
st.write("""
**Explanation:**
- **Number of Colors (K):** This controls the number of distinct colors used to represent the image. A lower K will result in a more compressed image with fewer color details, while a higher K will retain more colors and details but may lead to a larger file size.
- **Max Iterations:** Sets the maximum number of iterations for the K-Means algorithm to converge. More iterations may improve the result but also increase the processing time.
""")

if uploaded_file:
    original_img = np.array(Image.open(uploaded_file))  # No resizing

    # Calculate the total unique colors in the image
    reshaped_img = original_img.reshape(-1, 3)
    unique_colors = len(np.unique(reshaped_img, axis=0))

    # Set the maximum value for K based on unique colors
    max_k = min(unique_colors, 1000)  # Cap to 1000 for practicality

    # Create a row for manual input of K and Max Iterations
    col1, col2 = st.columns(2)

    with col1:
        K = st.number_input(
            f"Enter the number of colors (K) [Max: {max_k}]:",
            min_value=2,
            max_value=max_k,
            value=min(10, max_k)
        )

    with col2:
        max_iters = st.number_input("Enter max iterations:", min_value=5, max_value=50, value=10)

    # Create two columns for side-by-side image display
    col3, col4 = st.columns(2)

    with col3:
        st.image(original_img, caption="Original Image", use_container_width=True)

    with col4:
        with st.spinner("Running K-Means... Please wait."):
            # Optional downsampling for speed
            downsampling_factor = 2  # Adjust based on the speed/quality tradeoff
            small_img = original_img[::downsampling_factor, ::downsampling_factor]
            X_img = np.reshape(small_img, (-1, 3))

            # Run K-Means with optimizations
            batch_size = 500  # Adjust batch size for speed
            centroids, idx = run_kMeans(X_img, K, max_iters, batch_size)

            # Reshape back to the original image dimensions
            X_recovered = np.reshape(centroids[idx], small_img.shape)

            # Normalize the reconstructed image to [0, 255] and convert to uint8
            X_recovered = np.clip(X_recovered, 0, 255).astype(np.uint8)

            # Show the output image
            st.image(X_recovered, caption=f"Compressed Image with {K} Colors", use_container_width=True)

            # Prepare the image for download
            buffered = io.BytesIO()
            output_image = Image.fromarray(X_recovered)
            output_image.save(buffered, format="PNG")
            buffered.seek(0)

            st.download_button(
                label="Download Compressed Image",
                data=buffered,
                file_name="compressed_image.png",
                mime="image/png",
                use_container_width=True
            )
