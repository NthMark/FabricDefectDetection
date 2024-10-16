import cv2
import numpy as np
import math
import cupy as cp
K = 64
C = 1
def create_patches(image, patch_size=7):
    # Create sliding windows of the given patch size
    patches = cp.lib.stride_tricks.sliding_window_view(image, (patch_size, patch_size, image.shape[2])).astype(cp.float32)
    return patches
def color_distance_vectorized(src, r, c,patch_size):
    # Calculate color distances from the pixel at (r, c) to all other pixels in the image
    c1 = src[r, c].astype(cp.float32)
    # Calculate half the patch size
    half_patch = patch_size // 2

    # Extract the 7x7 patch centered at (r, c)
    start_row = max(0, r - half_patch)
    end_row = min(src.shape[0], r + half_patch + 1)
    start_col = max(0, c - half_patch)
    end_col = min(src.shape[1], c + half_patch + 1)

    patch = src[start_row:end_row, start_col:end_col].astype(cp.float32)
    patch=patch.reshape(-1,3)
    neighbor_patches=create_patches(src,patch_size)
    neighbor_patches=neighbor_patches.reshape(-1,neighbor_patches.shape[2],neighbor_patches.shape[3],neighbor_patches.shape[4],neighbor_patches.shape[5])
    neighbor_patches=neighbor_patches.reshape(neighbor_patches.shape[0],neighbor_patches.shape[1],neighbor_patches.shape[2]*neighbor_patches.shape[3],neighbor_patches.shape[4])
    # all_pixels = src.reshape(-1, 3).astype(cp.float32)
    dc = (neighbor_patches - patch) / 255.0
    color_distances = cp.linalg.norm(dc, axis=(1,2))
    color_distances = np.linalg.norm(color_distances,axis=1)
    return color_distances

def distance_vectorized(src, r, c, rows, cols,patch_size):
    # Calculate color distance using the vectorized function
    color_distances = color_distance_vectorized(src, r, c,patch_size)
    # Calculate spatial distances from the pixel (r, c) to all other pixels
    row_indices, col_indices = cp.indices((rows, cols))
    larger_dimen=rows if rows >cols else cols
    row_indices=row_indices[int(patch_size//2):-int(patch_size//2),int(patch_size//2):-int(patch_size//2)]
    col_indices=col_indices[int(patch_size//2):-int(patch_size//2),int(patch_size//2):-int(patch_size//2)]
    # print(len(row_indices))
    # print(f'why:{row_indices.flatten()[int(patch_size//2):-int(patch_size//2)]}')
    dRow = (row_indices.flatten() - r) / larger_dimen
    dCol = (col_indices.flatten() - c) / larger_dimen
    xy_distances = cp.sqrt(dRow**2 + dCol**2)
    # Combine color and spatial distances
    combined_distances = color_distances / (1 + C * xy_distances)
    return combined_distances

def salient(src, r, c, rows, cols,patch_size):
    # Get the vectorized distance array
    distances = distance_vectorized(src, r, c, rows, cols,patch_size)
    # Sort the distances and select the smallest K
    smallest_diffs = cp.partition(distances, K)[:K]
    sum_diff = cp.sum(smallest_diffs)

    # Calculate the saliency value
    return 1 - math.exp(-sum_diff / K)
def saliencyMatrix(src, u,patch_size):
    src = src.copy()
    ###Cupy
    ###
    rows, cols, _ = src.shape
    for row in range(rows):
        for col in range(cols):
            n, l, a, b = 0, 0.0, 0.0, 0.0  # Initialize l, a, b as floats to prevent overflow
            for r in range(max(row - u, 0), min(row + u + 1, rows)):
                for c in range(max(col - u, 0), min(col + u + 1, cols)):
                    n += 1
                    l += float(src[r, c][0])  # Cast to float to avoid overflow
                    a += float(src[r, c][1])
                    b += float(src[r, c][2])
            # Store the average back in the src matrix, converting to integers
            src[row, col] = (int(l / n), int(a / n), int(b / n))
    # tg = np.zeros((rows, cols), dtype=np.uint8)
    # mid = np.zeros((rows, cols), dtype=np.float64)
    src = cp.asarray(src)
    tg = cp.zeros((rows, cols), dtype=cp.uint8)
    mid = cp.zeros((rows, cols), dtype=cp.float32)
    _max = 0.0
    print(f'Generating at u={u}')
    for row in range(patch_size//2,rows-patch_size//2):
        print(f"{row}\n")
        for col in range(patch_size//2,cols-patch_size//2):
            value = salient(src, row, col, rows, cols,patch_size)
            mid[row, col] = value
            if value > _max:
                _max = value
    print("Done salient \n")
    # for row in range(rows):
    #     for col in range(cols):
    #         tg[row, col] = np.uint8((mid[row, col] / _max) * 255)
    # tg = cp.uint8((mid / _max) * 255)
    tg = (mid / _max) * 255  # Compute the normalized values
    tg = cp.clip(tg, 0, 255)  # Clip the values to the range [0, 255]
    tg = tg.astype(cp.uint8)  # Convert to uint8

    tg = tg.get()
    return tg

def exec(filename, outFile, u,patck_size=7):
    image = cv2.imread(filename)
    # for idx, row in image_annotations.iterrows():
    #     bbox = row['bbox']  # [x, y, width, height]
    #     cropImage=image[int(bbox[1]):bbox[3]+int(bbox[1]),int(bbox[0]):int(bbox[0])+bbox[2]]
    # # Draw rectangle
    # cropImage=image[int(bbox[1]):bbox[3]+int(bbox[1]),int(bbox[0]):int(bbox[0])+bbox[2]]
    # plt.imshow(cropImage)
    # plt.title('Image with Bounding Boxes',fontweight="bold")
    # plt.axis('off')
    # plt.show()
    source = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    rows, cols, _ = source.shape
    print(f'rows: {rows}, cols: {cols}')
    
    tg4 = saliencyMatrix(source, 0,patck_size)
    test1="test1"
    cv2.imwrite(test1 + ".7.png", tg4)
    tg2 = saliencyMatrix(source, u ,patck_size-u)
    cv2.imwrite(test1 + ".5.png", tg2)
    tg0 = saliencyMatrix(source, 2*u,patck_size-2*u)
    cv2.imwrite(test1 + ".3.png", tg0)
    tg = np.zeros((rows, cols), dtype=np.uint8)
    
    mainPart = []
    _max = 0
    for row in range(rows):
        for col in range(cols):
            avg = np.uint8((int(tg4[row, col]) + int(tg2[row, col]) + int(tg0[row, col])) / 3)
            tg[row, col] = avg
            if avg > _max:
                _max = avg
    print(f'tg max: {_max}')
    for row in range(rows):
        for col in range(cols):
            tg[row, col] = np.uint8((tg[row, col] * 255.0) / _max)
            if tg[row, col] > 204:
                mainPart.append(((row, col), tg[row, col]))
    cv2.imwrite(test1 + ".tg.png", tg)

    S = tg.copy()
    print(f'len: {len(mainPart)}')
    print('Optimizing')
    if mainPart:
        print("Calculating...")
        for row in range(rows):
            for col in range(cols):
                value = S[row, col]
                if value > 204:
                    continue
                dis = 1.0
                for p in mainPart:
                    dRow = (p[0][0] - row) / rows
                    dCol = (p[0][1] - col) / cols
                    _dis = math.sqrt(dRow * dRow + dCol * dCol)
                    if _dis < dis:
                        dis = _dis
                S[row, col] = np.uint8(value * (1.0 - dis))
    # cv2.imwrite(filename + ".fin.png", S)
    cv2.imwrite(outFile, S)
# exec("football.png","final.png",2)
if __name__=='__main__':
    with cp.cuda.Device(0):
        exec("TILDA_Fabric.v2-tilda-v2.coco/train/c1r3e2n19_jpg.rf.eda23f406c233ea71a5eb96bf49e082b.jpg","final.png",2)