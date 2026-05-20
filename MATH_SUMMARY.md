# Mathematics in the Image Search System

Based on the codebase, here is a summary of all the mathematical operations and concepts used, broken down by their function:

## 1. Vector Normalization (L2 Norm)
Before being saved to the index or used for querying, every embedding vector (whether it's from a color histogram or CLIP) undergoes **L2 normalization**.
This process scales the vector so that its length (magnitude) is exactly 1, turning it into a "unit vector." 

**Formula:**
For a vector $\mathbf{v} = [v_1, v_2, \dots, v_n]$, the L2 norm is $||\mathbf{v}||_2 = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2}$.
The normalized vector is:
$$ \hat{\mathbf{v}} = \frac{\mathbf{v}}{||\mathbf{v}||_2} $$
*(Found in `src/embeddings.py`)*

## 2. Distance and Similarity Metrics (Cosine)
To find similar images, the system calculates the distance between the query image's vector and the vectors in the database. The system uses **Cosine Distance** via `scikit-learn`'s `NearestNeighbors`.

**Formula:**
Cosine distance measures the angle between two vectors. Since your vectors are already L2-normalized, the cosine distance is simply $1$ minus the dot product of the vectors:
$$ D_{cosine}(\mathbf{u}, \mathbf{v}) = 1 - (\mathbf{u} \cdot \mathbf{v}) $$

To convert this distance back into a **Similarity Score** (where 1.0 is a perfect match and 0.0 is completely dissimilar), the system subtracts the distance from 1:
$$ Similarity = 1.0 - D_{cosine}(\mathbf{u}, \mathbf{v}) $$
*(Found in `src/search.py`)*


## 3. 3D Color Histograms
When using the "histogram" embedding method, the math involves creating a **3-Dimensional Joint Histogram** of the RGB color space. 
* The image's RGB pixel values are mapped from the range `[0, 255]` to the continuous range `[0.0, 1.0]`.
* The 3D color space is divided into bins (8 bins per color channel, resulting in $8 \times 8 \times 8 = 512$ total bins).
* The algorithm counts how many pixels fall into each specific 3D bin (e.g., how many pixels have a specific combination of red, green, and blue).
*(Found in `src/embeddings.py`)*

## 4. Utility Math
* **Ceiling function**: Used during indexing to calculate the total number of batches. `math.ceil(total_images / batch_size)` ensures that any remainder images are processed in a final, smaller batch. *(Found in `src/indexer.py`)*
