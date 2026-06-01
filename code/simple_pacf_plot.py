#!/usr/bin/env python3
"""
Simplified PACF Analysis Plot for RadioMotion Dataset
Generate a clean line plot with English labels
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm
import random

def manual_pacf(data, max_lag):
    """Manual PACF calculation for short time series"""
    n = len(data)
    if n <= max_lag:
        return np.full(max_lag + 1, np.nan)
    
    # Normalize data
    data = (data - np.mean(data)) / (np.std(data) + 1e-8)
    
    pacf_values = np.zeros(max_lag + 1)
    pacf_values[0] = 1.0  # lag 0 PACF is always 1
    
    for k in range(1, min(max_lag + 1, n)):
        if k == 1:
            # lag 1 PACF is just correlation coefficient
            if n > 1:
                pacf_values[1] = np.corrcoef(data[:-1], data[1:])[0, 1]
        else:
            # For lag k, fit AR(k) model
            try:
                if n - k <= 0:
                    pacf_values[k] = np.nan
                    continue
                    
                X = np.zeros((n - k, k))
                for i in range(k):
                    X[:, i] = data[i:n-k+i]
                y = data[k:n]
                
                # Check data validity
                if len(y) == 0 or X.shape[0] == 0:
                    pacf_values[k] = np.nan
                    continue
                
                # Least squares solution for AR coefficients
                try:
                    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                    pacf_values[k] = coeffs[-1]  # Last coefficient is PACF value
                except np.linalg.LinAlgError:
                    pacf_values[k] = np.nan
            except:
                pacf_values[k] = np.nan
    
    return pacf_values

def load_all_sequences(gain_dir):
    """Load all available sequences"""
    print("Scanning complete dataset...")
    
    all_sequences = []
    env_folders = [f for f in os.listdir(gain_dir) if f.startswith('env_')]
    
    print(f"Found {len(env_folders)} environment folders")
    
    for env in tqdm(env_folders, desc="Scanning environments"):
        env_path = os.path.join(gain_dir, env)
        if not os.path.isdir(env_path):
            continue
            
        traj_folders = [f for f in os.listdir(env_path) if f.startswith('traj_')]
        for traj in traj_folders:
            traj_path = os.path.join(env_path, traj)
            if not os.path.isdir(traj_path):
                continue
                
            tx_folders = [f for f in os.listdir(traj_path) if f.startswith('tx_')]
            for tx in tx_folders:
                tx_path = os.path.join(traj_path, tx)
                if not os.path.isdir(tx_path):
                    continue
                
                # Check if there are enough PNG files
                frame_files = [f for f in os.listdir(tx_path) if f.endswith('.png')]
                if len(frame_files) >= 15:
                    all_sequences.append((env, traj, tx))
    
    print(f"Found {len(all_sequences)} valid sequences (each with >=15 frames)")
    return all_sequences

def load_sequence_data(env, traj, tx, gain_dir):
    """Load 15-frame sequence data"""
    tx_path = os.path.join(gain_dir, env, traj, tx)
    
    try:
        frame_files = sorted([f for f in os.listdir(tx_path) if f.endswith('.png')], 
                           key=lambda f: int(f.split('_')[-1].split('.')[0]))
        
        if len(frame_files) < 15:
            return None
        
        frames = []
        for i in range(15):  # Only take first 15 frames
            image_path = os.path.join(tx_path, frame_files[i])
            image = io.imread(image_path).astype(np.float32) / 255.0
            frames.append(image)
        
        return np.array(frames)  # shape: (15, H, W)
    except Exception as e:
        return None

def analyze_pacf(sequences, gain_dir, max_lag=14):
    """Analyze PACF for all sequences"""
    print(f"Starting PACF analysis (max_lag={max_lag})...")
    
    all_pacf_results = []
    successful_count = 0
    
    for i, (env, traj, tx) in enumerate(tqdm(sequences, desc="Analyzing sequences")):
        # Load sequence
        sequence_data = load_sequence_data(env, traj, tx, gain_dir)
        if sequence_data is None:
            continue
        
        # Calculate spatial average time series
        spatial_avg = np.mean(sequence_data, axis=(1, 2))
        
        # Check data quality
        if np.std(spatial_avg) < 1e-6:  # Constant sequence
            continue
        
        # Calculate PACF
        pacf_values = manual_pacf(spatial_avg, max_lag)
        
        # Check result validity
        if not np.any(np.isnan(pacf_values)):
            all_pacf_results.append(pacf_values)
            successful_count += 1
            
            if successful_count % 1000 == 0:
                print(f"Successfully analyzed {successful_count} sequences")
    
    print(f"Successfully analyzed {successful_count} sequences")
    return np.array(all_pacf_results) if all_pacf_results else None

def create_simple_plot(pacf_results, output_dir, max_lag=14):
    """Create simplified PACF line plot"""
    if pacf_results is None or len(pacf_results) == 0:
        print("No valid PACF results to plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate statistics
    lags = np.arange(max_lag + 1)
    abs_mean_pacf = np.mean(np.abs(pacf_results), axis=0)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot |PACF| line
    plt.plot(lags, abs_mean_pacf, 'b-', linewidth=2.5, marker='o', markersize=7, 
             label='Mean |PACF|')
    
    # Add vertical line at lag 10
    plt.axvline(x=10, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                label='Frame 10 Boundary')
    
    # Set labels and title
    plt.xlabel('Lag', fontsize=20, fontweight='bold')
    plt.ylabel('|PACF|', fontsize=20, fontweight='bold')
    plt.title('PACF Analysis of RadioMotion Dataset', fontsize=18, fontweight='bold')
    
    # Set y-axis to start from 0
    plt.ylim(bottom=0)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks
    plt.xticks(range(0, max_lag+1, 2), fontsize=18)
    
    plt.yticks(fontsize=18)
    
    # Add legend
    plt.legend(fontsize=18, loc='upper right')
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'simple_pacf_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Simple PACF plot saved: {output_path}")
    
    plt.close()

def main():
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # Configuration
    data_root = '/Data/hgjia/scr/RadioMotion/data'
    gain_dir = os.path.join(data_root, 'gain')
    output_dir = './simple_pacf_results'
    max_lag = 14
    
    print("="*50)
    print("Simple PACF Analysis for RadioMotion Dataset")
    print("="*50)
    
    # Check data directory
    if not os.path.exists(gain_dir):
        print(f"Error: Data directory does not exist {gain_dir}")
        return
    
    # Load all sequences
    sequences = load_all_sequences(gain_dir)
    if not sequences:
        print("Error: No valid sequences found")
        return
    
    # Analyze PACF
    pacf_results = analyze_pacf(sequences, gain_dir, max_lag)
    if pacf_results is None:
        print("Error: PACF analysis failed")
        return
    
    # Create simple plot
    create_simple_plot(pacf_results, output_dir, max_lag)
    
    print("="*50)
    print("Analysis complete!")
    print(f"Results saved in: {output_dir}")
    print("="*50)

if __name__ == '__main__':
    main()
