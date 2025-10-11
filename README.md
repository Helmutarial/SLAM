# Collaborative SLAM Repository

This repository contains all the essential scripts, configuration files, and instructions for my Master's Thesis (TFM) focused on collaborative SLAM (Simultaneous Localization and Mapping) using OAK-D cameras.

## Purpose
The main goal is to record, process, and analyze data from OAK-D cameras to develop and test collaborative SLAM techniques. All scripts and tools here are designed to facilitate data acquisition, conversion, and analysis for this purpose.

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

## Contact
For any questions or issues, contact me via GitHub or email.
