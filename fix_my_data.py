# import os
# import numpy as np

# DATA_PATH = "NP_Data_Combined"

# def normalize_keypoints(kp):
#     """
#     Process 126-dim keypoints (Left Hand: 0-63, Right Hand: 63-126)
#     to be wrist-relative and scaled.
#     """
#     for start in [0, 63]:
#         hand = kp[start:start+63].reshape(21, 3)
#         if np.all(hand == 0): 
#             continue # Skip if no hand was detected in this frame
        
#         # 1. Make wrist (landmark 0) the origin (0,0,0)
#         wrist = hand[0]
#         relative = hand - wrist
        
#         # 2. Scale by max distance to make it size-invariant
#         max_dist = np.max(np.linalg.norm(relative, axis=1))
#         if max_dist > 0:
#             relative /= max_dist
        
#         kp[start:start+63] = relative.flatten()
#     return kp

# def main():
#     if not os.path.exists(DATA_PATH):
#         print(f"Error: {DATA_PATH} not found.")
#         return

#     print(f"Starting data transformation in {DATA_PATH}...")
#     total_files = 0
#     fixed_files = 0
    
#     for root, _, files in os.walk(DATA_PATH):
#         for file in files:
#             if file.endswith(".npy"):
#                 path = os.path.join(root, file)
#                 total_files += 1
                
#                 try:
#                     kp = np.load(path)
                    
#                     # Detection: If the first 3 values (wrist) are NOT 0, it's likely raw data.
#                     # Normalized data has wrist at (0,0,0).
#                     if not np.allclose(kp[0:3], 0):
#                         fixed_kp = normalize_keypoints(kp)
#                         np.save(path, fixed_kp)
#                         fixed_files += 1
#                 except Exception as e:
#                     print(f"Error processing {path}: {e}")

#     print("-" * 30)
#     print(f"Scan complete.")
#     print(f"Total .npy files found: {total_files}")
#     print(f"Files normalized: {fixed_files}")
#     print(f"Files already normalized: {total_files - fixed_files}")
#     print("\nNext step: Run your training script to retrain the model with robust data.")

# if __name__ == "__main__":
#     main()


import os

import numpy as np
 
DATA_PATH = "NP_Data_Combined"
 
def normalize_keypoints(kp):

    """

    Process 126-dim keypoints (Left Hand: 0-63, Right Hand: 63-126)

    to be wrist-relative and scaled.

    """

    for start in [0, 63]:

        hand = kp[start:start+63].reshape(21, 3)

        if np.all(hand == 0): 

            continue # Skip if no hand was detected in this frame

        # 1. Make wrist (landmark 0) the origin (0,0,0)

        wrist = hand[0]

        relative = hand - wrist

        # 2. Scale by max distance to make it size-invariant

        max_dist = np.max(np.linalg.norm(relative, axis=1))

        if max_dist > 0:

            relative /= max_dist

        kp[start:start+63] = relative.flatten()

    return kp
 
def needs_normalization(kp):

    """

    Check if keypoints need normalization by examining each hand independently.

    Returns True if either hand is in raw (non-normalized) format.

    """

    # Check left hand (indices 0-63)

    left_wrist = kp[0:3]

    # Check right hand (indices 63-66)

    right_wrist = kp[63:66]

    # If either wrist is NOT at origin, the data needs normalization

    left_needs_norm = not np.allclose(left_wrist, 0, atol=1e-5)

    right_needs_norm = not np.allclose(right_wrist, 0, atol=1e-5)

    return left_needs_norm or right_needs_norm
 
def main():

    if not os.path.exists(DATA_PATH):

        print(f"Error: {DATA_PATH} not found.")

        return
 
    print(f"Starting data transformation in {DATA_PATH}...")

    total_files = 0

    fixed_files = 0

    for root, _, files in os.walk(DATA_PATH):

        for file in files:

            if file.endswith(".npy"):

                path = os.path.join(root, file)

                total_files += 1

                try:

                    kp = np.load(path)

                    # Check if EITHER hand needs normalization

                    if needs_normalization(kp):

                        fixed_kp = normalize_keypoints(kp)

                        np.save(path, fixed_kp)

                        fixed_files += 1

                except Exception as e:

                    print(f"Error processing {path}: {e}")
 
    print("-" * 30)

    print(f"Scan complete.")

    print(f"Total .npy files found: {total_files}")

    print(f"Files normalized: {fixed_files}")

    print(f"Files already normalized: {total_files - fixed_files}")

    print("\nNext step: Run your training script to retrain the model with robust data.")
 
if __name__ == "__main__":

    main()
 