"""
Route A: STEW Dataset Zip Ingester (Full 48-Subject Edition)
===========================================================
Extracts raw biological human EEG data directly from the compressed
STEW Dataset.zip archive, targets internal .txt structures, handles 
128Hz -> 250Hz upsampling, and slices the 14 raw channels down to the 4 target
frontal channels (AF3, F7, F8, AF4) to align with our 4-channel BCI pipeline.
Upgraded to ingest the entire cohort of 48 subjects.
Ensures zero-centered biological signals by removing DC offsets.
"""

import os
import zipfile
import io
import numpy as np
import pandas as pd
from scipy.signal import resample

# Local Zip Path Configuration
ZIP_PATH = r"C:\Users\HP\Downloads\STEW Dataset.zip"
OUTPUT_DIR = "human_benchmarks"

def process_stew_from_zip(zip_path, output_path, sample_rate_orig=128, sample_rate_target=250):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"❌ Could not find the zip archive file at: {zip_path}")

    print(f"📦 Opening zip archive: {zip_path}")
    os.makedirs(output_path, exist_ok=True)
    
    lo_buffers = []
    hi_buffers = []

    with zipfile.ZipFile(zip_path, 'r') as z:
        all_files = z.namelist()
        
        # Match the explicit nested text paths discovered by inspection
        lo_files = sorted([f for f in all_files if f.endswith('_lo.txt')])
        hi_files = sorted([f for f in all_files if f.endswith('_hi.txt')]) # Corrected pattern match for workload

        # Sourcing all available subjects to establish the full Route A pool (48 subjects)
        num_subjects = min(48, len(lo_files), len(hi_files))
        print(f"🧬 Selected Subject Pool size: {num_subjects} human participants.")
        
        for i in range(num_subjects):
            try:
                # 1. Parse Resting State (_lo.txt)
                with z.open(lo_files[i]) as f:
                    content = io.StringIO(f.read().decode('utf-8'))
                    raw_data = np.loadtxt(content) # Shape (Samples, 14)
                    
                # 2. Parse Cognitive Load State (_hi.txt)
                with z.open(hi_files[i]) as f:
                    content = io.StringIO(f.read().decode('utf-8'))
                    raw_data_hi = np.loadtxt(content)

                # Resampling to align with the 250 Hz BCI pipeline
                n_samples_orig = len(raw_data)
                n_samples_target = int(n_samples_orig * sample_rate_target / sample_rate_orig)
                
                resampled_lo = resample(raw_data, n_samples_target, axis=0)
                resampled_hi = resample(raw_data_hi, n_samples_target, axis=0)
                
                lo_buffers.append(resampled_lo)
                hi_buffers.append(resampled_hi)
                  
                print(f"   ✓ Successfully parsed and resampled subject index {i+1}: {os.path.basename(lo_files[i])}")
            except Exception as e:
                print(f"   ⚠️ Mismatch handling file entry index {i}: {e}")

    if lo_buffers and hi_buffers:
        # Standardize arrays (Shape: Samples x 14)
        rest_block = np.vstack(lo_buffers)
        cog_block = np.vstack(hi_buffers)
        
        # ── Slice 14 channels down to the 4 target frontal channels ──
        # Standard Emotiv EPOC layout mapping: 
        # AF3 (Index 0), F7 (Index 1), F8 (Index 12), AF4 (Index 13)
        print("\n📊 Slicing 14 raw channels down to 4 target frontal channels (AF3, F7, F8, AF4)...")
        target_channels = [0, 1, 12, 13]
        rest_block = rest_block[:, target_channels]
        cog_block = cog_block[:, target_channels]
        
        # ── Subtract the Global DC Offset (Crucial Correction) ──
        # Real biological EEG must oscillate symmetrically around 0.
        # We subtract each channel's mean across the entire timeline to strip out the ADC hardware bias.
        print("⚡ Centering biological human signal profiles (removing DC offsets)...")
        lo_means = np.mean(rest_block, axis=0)
        hi_means = np.mean(cog_block, axis=0)
        
        print(f"   • Resting Channel Means:    {lo_means.round(2)} uV")
        print(f"   • Workload Channel Means:   {hi_means.round(2)} uV")
        
        rest_block = rest_block - lo_means
        cog_block = cog_block - hi_means
        print("   ✓ DC Offset correction applied successfully. Signals are now globally zero-centered.")
        
        columns = ["Filt_Ch1", "Filt_Ch2", "Filt_Ch3", "Filt_Ch4"]
        
        # Exporting data compliant with the evaluation suites
        rest_df = pd.DataFrame(rest_block, columns=columns)
        rest_df.insert(0, "Timestamp", np.linspace(1000, 1000 + len(rest_df)/250, len(rest_df)))
        rest_df.to_csv(os.path.join(output_path, "human_resting_alpha.csv"), index=False)
        
        cog_df = pd.DataFrame(cog_block, columns=columns)
        cog_df.insert(0, "Timestamp", np.linspace(2000, 2000 + len(cog_df)/250, len(cog_df)))
        cog_df.to_csv(os.path.join(output_path, "human_cognitive_load.csv"), index=False)
        
        print(f"\n✅ SUCCESS: Route A Ingestion Complete!")
        print(f"   • Saved resting-state baseline:   {os.path.join(output_path, 'human_resting_alpha.csv')}")
        print(f"   • Saved cognitive workload state: {os.path.join(output_path, 'human_cognitive_load.csv')}")
    else:
        print("❌ Error: No valid data files were parsed.")

if __name__ == "__main__":
    process_stew_from_zip(ZIP_PATH, OUTPUT_DIR)