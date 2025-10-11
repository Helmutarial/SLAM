"""
Video player auxiliary script
Provides functions to play video files synchronized with trajectory visualization
"""

import os
import sys
import subprocess
import threading
import time

def get_video_duration(video_path):
    """Get video duration using ffprobe"""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_format', video_path
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
            return duration
        else:
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None

def play_video_loop(video_path, duration_seconds=None):
    """Play video in loop for specified duration"""
    if not video_path or not os.path.exists(video_path):
        print(f"⚠️ Video file not found: {video_path}")
        return None
    
    print(f"🎥 Starting video playback: {os.path.basename(video_path)}")
    
    def play_video():
        try:
            if duration_seconds:
                start_time = time.time()
                while time.time() - start_time < duration_seconds:
                    if os.name == 'nt':  # Windows
                        # Use VLC if available, otherwise default player
                        try:
                            subprocess.run(['vlc', '--intf', 'dummy', '--play-and-exit', 
                                          '--no-video-title-show', video_path], 
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                                         timeout=duration_seconds)
                        except (FileNotFoundError, subprocess.TimeoutExpired):
                            # Fallback to default Windows player
                            os.startfile(video_path)
                            time.sleep(duration_seconds)
                            break
                    elif os.name == 'posix':  # macOS/Linux
                        if sys.platform == 'darwin':  # macOS
                            subprocess.run(['open', video_path])
                        else:  # Linux
                            subprocess.run(['xdg-open', video_path])
                        time.sleep(duration_seconds)
                        break
            else:
                # Play once without duration limit
                if os.name == 'nt':  # Windows
                    os.startfile(video_path)
                elif os.name == 'posix':  # macOS/Linux
                    if sys.platform == 'darwin':  # macOS
                        subprocess.run(['open', video_path])
                    else:  # Linux
                        subprocess.run(['xdg-open', video_path])
        except Exception as e:
            print(f"⚠️ Error playing video: {e}")
    
    # Start video in background thread
    video_thread = threading.Thread(target=play_video)
    video_thread.daemon = True
    video_thread.start()
    
    return video_thread

def find_video_file(folder_path):
    """Find video file in the specified folder"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.m4v']
    
    for ext in video_extensions:
        import glob
        video_files = glob.glob(os.path.join(folder_path, ext))
        if video_files:
            return video_files[0]  # Return first found video
    
    return None

def play_video_synchronized(video_path, trajectory_duration_seconds):
    """Play video synchronized with trajectory visualization"""
    if not video_path:
        print("⚠️ No video file provided")
        return None
    
    video_duration = get_video_duration(video_path)
    if video_duration:
        print(f"📹 Video duration: {video_duration:.1f}s")
        print(f"🎬 Trajectory duration: {trajectory_duration_seconds:.1f}s")
        
        # Mostrar información de sincronización
        if abs(video_duration - trajectory_duration_seconds) < 1:
            print("✅ Video y trayectoria tienen duración similar - Sincronización perfecta")
        elif trajectory_duration_seconds > video_duration:
            loops_needed = trajectory_duration_seconds / video_duration
            print(f"🔄 Video se repetirá {loops_needed:.1f} veces para cubrir la trayectoria")
        else:
            print("⚡ Video es más largo que la trayectoria")
    
    return play_video_loop(video_path, trajectory_duration_seconds)

if __name__ == "__main__":
    # Test the video player
    print("🎥 VIDEO PLAYER TEST")
    print("="*30)
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        duration = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
        
        print(f"Testing video: {video_path}")
        print(f"Duration: {duration}s")
        
        thread = play_video_synchronized(video_path, duration)
        if thread:
            thread.join()  # Wait for video to finish
    else:
        print("Usage: python video_player.py <video_path> [duration_seconds]")