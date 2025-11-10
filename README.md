# Collaborative SLAM Repository

This repository contains all the essential scripts, configuration files, and instructions for my Master's Thesis (TFM) focused on collaborative SLAM (Simultaneous Localization and Mapping) using OAK-D cameras.

## Purpose
The main goal is to record, process, and analyze data from OAK-D cameras to develop and test collaborative SLAM techniques. All scripts and tools here are designed to facilitate data acquisition, conversion, and analysis for this purpose.

## Project Structure

### Core Components
- **Data Recording**: Tools for capturing data from OAK-D cameras
- **Point Cloud Processing**: Conversion from depth data to 3D point clouds
- **Trajectory Extraction**: Camera pose estimation and trajectory analysis
- **Object Detection**: YOLO-based 3D object detection in point clouds
- **Point Cloud Alignment**: Multiple methods for aligning point clouds from different viewpoints:
  - ICP (Iterative Closest Point)
  - Detection-based alignment
  - **USIP (AI-enhanced keypoint matching)** ✨

### USIP Integration (New!)
The repository now includes **USIP (Unsupervised Stable Interest Point detection)** for AI-enhanced point cloud alignment. USIP uses deep learning to extract stable keypoints from point clouds, enabling robust matching and alignment even with challenging viewpoints.

**Key Features:**
- Pre-trained models (ModelNet, Oxford datasets)
- Batch keypoint extraction from point clouds
- RANSAC-based transformation estimation
- Visualization tools for quality assessment
- Achieves ~6.6cm average RMSE on VIDEO4 dataset

**Usage:**
```powershell
# Extract keypoints from all frames
python collaborative_slam/tools/batch_extract_keypoints.py `
  --cloud_dir data/VIDEO4/cloud_points `
  --output_dir data/VIDEO4/keypoints `
  --num_keypoints 512 `
  --model_path "collaborative_slam/external/USIP/checkpoints/modelnet-.../net_detector.pth"

# Match keypoints between consecutive frames
python collaborative_slam/tools/alignment_methods/keypoint_matcher.py `
  --keypoint_dir data/VIDEO4/keypoints `
  --output data/VIDEO4/keypoints/matching_results.json `
  --ransac_threshold 0.1

# Visualize matching results
python collaborative_slam/tools/alignment_methods/visualize_keypoint_matches.py `
  --keypoint_dir data/VIDEO4/keypoints `
  --matching_results data/VIDEO4/keypoints/matching_results.json `
  --plot_stats
```

See `data/VIDEO4/keypoints/USIP_REPORT.md` for detailed results and analysis.

## How to Record Data
To record data from your OAK-D camera, use the following command:

```powershell
sai-cli record oak --output my_recording_folder
```

- Replace `my_recording_folder` with the desired output folder name.
- Make sure your camera is connected and all dependencies are installed (see `requirements.txt`).

## About the ffmpeg-essentials_build Folder
The folder `ffmpeg-8.0-essentials_build` contains the FFmpeg binaries required for video conversion and processing. FFmpeg is necessary to:
- Convert raw video files (e.g., `.h265`) to more common formats like `.mp4`.
- Enable video preview and conversion features in the recording scripts.
- Ensure compatibility with SpectacularAI and DepthAI tools that rely on FFmpeg for video handling.

**Important:**
- The `bin` subfolder must be added to your system PATH for FFmpeg to work from any terminal.
- If FFmpeg is not installed or not in the PATH, some video conversion features may not work.

## Requirements
All required Python packages are listed in `requirements.txt`. Install them with:

```powershell
pip install -r requirements.txt
```

### Additional Setup for USIP
USIP requires CUDA and compiled extensions:
1. Install CUDA Toolkit 12.4 or compatible version
2. Install Visual Studio 2022 Build Tools with C++ support
3. Compile USIP CUDA extensions:
   ```powershell
   cd collaborative_slam/external/USIP/models/index_max_ext
   python setup.py install
   cd ../ball_query_ext
   python setup.py install
   ```

## Contact
For any questions or issues, contact me via GitHub or email.
