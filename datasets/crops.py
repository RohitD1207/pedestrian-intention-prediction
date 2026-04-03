import os
import cv2
import pandas as pd
from tqdm import tqdm

def pre_extract_crops(annotation_file, video_dir, output_dir):
    print("Reading CSV...")
    df = pd.read_csv(annotation_file)
    os.makedirs(output_dir, exist_ok=True)
    
    videos = df['video'].unique()
    print(f"Found {len(videos)} videos to process.")
    
    for video_name in tqdm(videos, desc="Processing Videos"):
        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        
        if not os.path.exists(video_path):
            # Try with an alternative extension or prefix if needed
            print(f" Skipping: {video_path} not found.")
            continue

        cap = cv2.VideoCapture(video_path)
        
        # Optimize: Pre-group annotations by frame for this video
        video_df = df[df['video'] == video_name]
        # This creates a dictionary where keys are frame numbers
        frame_dict = {frame: group for frame, group in video_df.groupby('frame')}
        
        current_frame = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        pbar = tqdm(total=total_frames, desc=f" Extracting {video_name}", leave=False)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Fast dictionary lookup instead of slow Pandas filtering
            if current_frame in frame_dict:
                for _, row in frame_dict[current_frame].iterrows():
                    pid = row['pedestrian_id']
                    x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                    
                    # Ensure coordinates are within image boundaries
                    h, w, _ = frame.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        save_path = os.path.join(output_dir, str(video_name), str(pid))
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        
                        cv2.imwrite(os.path.join(save_path, f"{current_frame:06d}.jpg"), crop)
            
            current_frame += 1
            pbar.update(1)
            
        pbar.close()
        cap.release()

if __name__ == "__main__":
    # DOUBLE CHECK THESE PATHS
    pre_extract_crops(
        annotation_file="datasets/pie_annotations_set03.csv", 
        video_dir="data/PIE_clips/set03", 
        output_dir="data/PIE_crops"
    )