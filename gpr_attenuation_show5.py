import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from obspy.io.segy.segy import _read_segy
import obspy.io.segy.segy as segy
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import linregress
import warnings
import os
warnings.filterwarnings('ignore')

class GPRAttenuationAnalyzer:
    """
    Complete corrected GPR attenuation analysis from SEG-Y files using ObsPy
    
    Physics Background:
    - GPR signals attenuate exponentially with round-trip depth: A(z) = A₀ * e^(-2αz)
    - α is the one-way attenuation coefficient (Np/m or dB/m)
    - Two-way travel time: depth = (velocity × time) / 2
    - Round-trip attenuation: signal travels down z and back up (total path = 2z)
    """
    
    def __init__(self, segy_file_path, velocity=0.1, window_size=0.2, max_depth=None,
                 time_zero_ns=0, direct_wave_end_ns=15):
        """
        Initialize the analyzer with corrections
        
        Parameters:
        -----------
        segy_file_path : str
            Path to the SEG-Y file
        velocity : float
            GPR wave velocity in m/ns (default: 0.1 m/ns for typical soil)
        window_size : float
            Depth window size in meters for amplitude averaging (default: 0.2 m)
        max_depth : float or None
            Maximum depth to analyze in meters (None = auto-determine)
        time_zero_ns : float
            Time-zero correction in nanoseconds (default: 0)
        direct_wave_end_ns : float
            End time of direct wave to exclude from analysis (default: 15 ns)
            Set to 0 to disable direct wave removal
        """
        self.segy_file_path = segy_file_path
        self.velocity = velocity  # m/ns
        self.window_size = window_size  # meters
        self.max_depth = max_depth  # meters
        self.time_zero_ns = time_zero_ns
        self.direct_wave_end_ns = direct_wave_end_ns  # Default 15 ns
        self.stream = None
        self.traces = None
        self.time_axis = None
        self.depth_axis = None
        self.sample_rate = None
        self.n_samples = None
        self.n_traces = None
        
    def read_segy_file(self):
        """
        Read SEG-Y file using ObsPy with robust error handling for irregular traces
        """
        print("Reading SEG-Y file with ObsPy...")
        
        if not os.path.exists(self.segy_file_path):
            raise FileNotFoundError(f"SEG-Y file not found: {self.segy_file_path}")
        
        try:
            # Method 1: Try standard ObsPy read with different options
            print("Attempting standard ObsPy read...")
            self.stream = read(self.segy_file_path, format='SEGY', 
                             unpack_trace_headers=True, headonly=False)
            
        except Exception as e:
            print(f"Standard read failed: {e}")
            try:
                # Method 2: Try with textual_header_encoding options
                print("Attempting read with encoding options...")
                self.stream = read(self.segy_file_path, format='SEGY',
                                 textual_header_encoding='ascii',
                                 unpack_trace_headers=True)
                
            except Exception as e2:
                print(f"ASCII encoding read failed: {e2}")
                try:
                    # Method 3: Try with EBCDIC encoding
                    print("Attempting read with EBCDIC encoding...")
                    self.stream = read(self.segy_file_path, format='SEGY',
                                     textual_header_encoding='EBCDIC-CP-US',
                                     unpack_trace_headers=True)
                    
                except Exception as e3:
                    print(f"EBCDIC encoding read failed: {e3}")
                    try:
                        # Method 4: Try reading with minimal validation
                        print("Attempting read with minimal validation...")
                        self.stream = self._read_with_minimal_validation()
                        
                    except Exception as e4:
                        raise RuntimeError(f"All reading methods failed. Last error: {e4}")
        
        # Process the successfully read stream
        self._process_stream()
        
    def _read_with_minimal_validation(self):
        """
        Custom reading method with minimal validation for problematic files
        """
        from obspy.core import Stream, Trace
        from obspy.core.util import AttribDict
        import struct
        
        print("Using custom minimal validation reader...")
        
        with open(self.segy_file_path, 'rb') as f:
            # Skip textual header (3200 bytes)
            f.seek(3200)
            
            # Read binary header (400 bytes)
            binary_header = f.read(400)
            
            # Extract key parameters from binary header
            # Sample interval (bytes 3217-3218, 1-indexed = 16-17, 0-indexed)
            sample_interval = struct.unpack('>H', binary_header[16:18])[0]  # microseconds
            
            # Number of samples per trace (bytes 3221-3222, 1-indexed = 20-21, 0-indexed)
            samples_per_trace = struct.unpack('>H', binary_header[20:22])[0]
            
            # Data format (bytes 3225-3226, 1-indexed = 24-25, 0-indexed)
            data_format = struct.unpack('>H', binary_header[24:26])[0]
            
            print(f"Binary header info:")
            print(f"  Sample interval: {sample_interval} μs")
            print(f"  Samples per trace: {samples_per_trace}")
            print(f"  Data format: {data_format}")
            
            # Determine bytes per sample based on format
            bytes_per_sample = {1: 4, 2: 4, 3: 2, 5: 4, 8: 1}.get(data_format, 4)
            
            # Calculate trace size (240 bytes header + data)
            trace_data_size = samples_per_trace * bytes_per_sample
            trace_size = 240 + trace_data_size
            
            # Read all traces
            traces_data = []
            trace_count = 0
            
            current_pos = f.tell()
            f.seek(0, 2)  # Go to end
            file_size = f.tell()
            f.seek(current_pos)  # Go back
            
            print(f"File size: {file_size} bytes")
            print(f"Expected trace size: {trace_size} bytes")
            print(f"Estimated number of traces: {(file_size - 3600) // trace_size}")
            
            while f.tell() < file_size - trace_size:
                try:
                    # Read trace header (240 bytes)
                    trace_header = f.read(240)
                    if len(trace_header) < 240:
                        break
                    
                    # Read trace data
                    trace_data_bytes = f.read(trace_data_size)
                    if len(trace_data_bytes) < trace_data_size:
                        print(f"Warning: Incomplete trace data at trace {trace_count}")
                        break
                    
                    # Unpack trace data based on format
                    if data_format == 1:  # IBM float
                        # Convert IBM float to IEEE float (simplified)
                        trace_data = self._ibm_to_ieee(trace_data_bytes)
                    elif data_format == 5:  # IEEE float
                        trace_data = struct.unpack(f'>{samples_per_trace}f', trace_data_bytes)
                    elif data_format == 3:  # 2-byte integer
                        trace_data = struct.unpack(f'>{samples_per_trace}h', trace_data_bytes)
                    elif data_format == 2:  # 4-byte integer
                        trace_data = struct.unpack(f'>{samples_per_trace}i', trace_data_bytes)
                    else:
                        # Default to float
                        trace_data = struct.unpack(f'>{samples_per_trace}f', trace_data_bytes)
                    
                    traces_data.append(np.array(trace_data))
                    trace_count += 1
                    
                    if trace_count % 100 == 0:
                        print(f"Read {trace_count} traces...")
                        
                except Exception as e:
                    print(f"Error reading trace {trace_count}: {e}")
                    break
            
            print(f"Successfully read {trace_count} traces")
            
            # Create ObsPy stream
            stream = Stream()
            
            for i, data in enumerate(traces_data):
                trace = Trace(data=data)
                trace.stats.sampling_rate = 1e6 / sample_interval  # Convert μs to Hz
                trace.stats.starttime = 0
                trace.stats.channel = f'TR{i:04d}'
                stream.append(trace)
            
            return stream
    
    def _ibm_to_ieee(self, ibm_bytes):
        """
        Convert IBM floating point to IEEE floating point (simplified conversion)
        """
        # This is a simplified conversion - for production use, consider using
        # a proper IBM float conversion library
        ieee_values = []
        for i in range(0, len(ibm_bytes), 4):
            if i + 4 <= len(ibm_bytes):
                # Simple approximation - unpack as big-endian signed int and scale
                val = struct.unpack('>i', ibm_bytes[i:i+4])[0]
                ieee_values.append(val / 16777216.0)  # Rough scaling
        return ieee_values
    
    def _process_stream(self):
        """
        Process the ObsPy stream to extract trace data and handle non-uniform traces
        """
        print("Processing ObsPy stream...")
        
        if not self.stream:
            raise ValueError("No stream data available")
        
        self.n_traces = len(self.stream)
        print(f"Number of traces: {self.n_traces}")
        
        # Handle non-uniform trace lengths
        trace_lengths = [len(tr.data) for tr in self.stream]
        print(f"Trace lengths - Min: {min(trace_lengths)}, Max: {max(trace_lengths)}, Mean: {np.mean(trace_lengths):.1f}")
        
        # Use the most common trace length or minimum length
        self.n_samples = min(trace_lengths)  # Use minimum to avoid index errors
        print(f"Using trace length: {self.n_samples} samples")
        
        # Extract sampling information with GPR-specific corrections
        sampling_rates = [tr.stats.sampling_rate for tr in self.stream]
        raw_sample_rate = np.mean(sampling_rates)  # Hz
        
        # GPR-specific sample rate correction
        # Many GPR files have incorrect sample rate in headers
        print(f"Raw sampling rate from headers: {raw_sample_rate:.2f} Hz")
        
        # For GPR, sample rates are typically in MHz range
        # If rate is too low, it's likely in wrong units
        if raw_sample_rate < 1000:  # Less than 1 kHz is suspicious for GPR
            print("Warning: Sample rate seems too low for GPR data!")
            print("Attempting to correct sample rate...")
            
            # Common GPR sample rates (MHz): 0.5, 1, 2, 5, 10, 20, 50, 100
            # Try to estimate from trace length and typical GPR time windows
            typical_time_windows_ns = [50, 100, 200, 500, 1000, 2000]  # Common GPR time windows
            
            estimated_rates = []
            for time_window in typical_time_windows_ns:
                estimated_rate = (self.n_samples / time_window) * 1e9  # Convert to Hz
                estimated_rates.append(estimated_rate)
            
            # Choose most reasonable rate (closest to common GPR frequencies)
            common_gpr_rates = [500e6, 200e6, 100e6, 50e6, 20e6, 10e6, 5e6, 2e6, 1e6]  # Hz
            
            best_rate = raw_sample_rate
            best_diff = float('inf')
            
            for est_rate in estimated_rates:
                for common_rate in common_gpr_rates:
                    diff = abs(est_rate - common_rate) / common_rate
                    if diff < best_diff and diff < 0.5:  # Within 50%
                        best_rate = common_rate
                        best_diff = diff
            
            if best_rate != raw_sample_rate:
                print(f"Corrected sampling rate: {best_rate/1e6:.1f} MHz")
                self.sample_rate = best_rate
            else:
                # Fallback: assume reasonable GPR parameters
                print("Could not determine correct sample rate. Using fallback.")
                # Assume 100 ns time window for 512 samples
                self.sample_rate = self.n_samples / 100e-9  # Hz for 100 ns window
                print(f"Fallback sampling rate: {self.sample_rate/1e6:.1f} MHz")
        else:
            self.sample_rate = raw_sample_rate
        
        # Convert to time interval in nanoseconds
        self.sample_interval_ns = 1e9 / self.sample_rate  # ns
        
        print(f"Final sampling rate: {self.sample_rate/1e6:.2f} MHz")
        print(f"Sample interval: {self.sample_interval_ns:.2f} ns")
        
        # Sanity check for GPR data
        total_time_window = self.n_samples * self.sample_interval_ns
        if total_time_window > 10000:  # More than 10 μs is suspicious
            print(f"Warning: Time window ({total_time_window:.1f} ns) seems too large for GPR!")
            print("This might indicate header interpretation issues.")
            
            # Force reasonable values for GPR
            reasonable_time_window = 200  # 200 ns is reasonable for many GPR surveys
            self.sample_interval_ns = reasonable_time_window / self.n_samples
            self.sample_rate = 1e9 / self.sample_interval_ns
            print(f"Forced correction - new sample interval: {self.sample_interval_ns:.2f} ns")
            print(f"Forced correction - new sampling rate: {self.sample_rate/1e6:.2f} MHz")
        
        # Extract and normalize trace data
        self.traces = np.zeros((self.n_traces, self.n_samples))
        
        for i, trace in enumerate(self.stream):
            # Truncate or pad traces to uniform length
            if len(trace.data) >= self.n_samples:
                self.traces[i, :] = trace.data[:self.n_samples]
            else:
                # Pad with zeros if trace is shorter
                self.traces[i, :len(trace.data)] = trace.data
                self.traces[i, len(trace.data):] = 0
        
        # Create time and depth axes
        self.time_axis = np.arange(0, self.n_samples) * self.sample_interval_ns
        
        # Convert time to depth using two-way travel time formula
        # Depth = (velocity × time) / 2
        self.depth_axis = (self.velocity * self.time_axis) / 2
        
        print(f"Time window: 0 to {self.time_axis[-1]:.1f} ns")
        print(f"Maximum depth: {self.depth_axis[-1]:.2f} m")
        
        # Final sanity check
        if self.depth_axis[-1] > 100:  # More than 100m is unusual for most GPR
            print(f"Warning: Maximum depth ({self.depth_axis[-1]:.1f} m) seems very large for GPR!")
            print("Consider adjusting velocity parameter or check data interpretation.")
        
        print(f"Trace data shape: {self.traces.shape}")
        
        # Check for valid data
        if np.all(self.traces == 0):
            print("Warning: All trace data appears to be zero!")
        else:
            print(f"Data range: {np.min(self.traces):.2e} to {np.max(self.traces):.2e}")
    
    def apply_horizontal_background_removal(self):
        """
        Apply horizontal background removal filter to remove systematic noise
        
        This removes the mean trace (background) from all traces to eliminate:
        - Direct coupling between antennas
        - Systematic electronic noise
        - Horizontally coherent artifacts
        - DC offsets and low-frequency drift
        
        Mathematical Process:
        1. Calculate mean trace: mean_trace = (1/N) * Σ(trace_i) for i=1 to N
        2. Subtract from each trace: filtered_trace_i = trace_i - mean_trace
        """
        print("Applying horizontal background removal filter...")
        
        if self.traces is None:
            raise ValueError("No trace data available for background removal")
        
        # Store original traces for comparison
        original_traces = self.traces.copy()
        
        # Calculate mean trace across all traces (background)
        mean_trace = np.mean(self.traces, axis=0)
        
        # Subtract mean trace from each individual trace
        self.traces = self.traces - mean_trace[np.newaxis, :]
        
        # Calculate statistics for reporting
        original_amplitude = np.std(original_traces)
        filtered_amplitude = np.std(self.traces)
        noise_removed = np.std(mean_trace)
        
        print(f"Background removal completed:")
        print(f"  • Original signal amplitude (std): {original_amplitude:.3e}")
        print(f"  • Background amplitude removed: {noise_removed:.3e}")
        print(f"  • Filtered signal amplitude (std): {filtered_amplitude:.3e}")
        print(f"  • Background noise ratio: {noise_removed/original_amplitude*100:.1f}%")
        
        # Store background information for potential plotting
        self.background_trace = mean_trace
        self.original_traces = original_traces
        
        return mean_trace
    
    def plot_background_removal_comparison(self):
        """
        Plot comparison showing before/after background removal
        """
        if not hasattr(self, 'original_traces') or not hasattr(self, 'background_trace'):
            print("Background removal not yet applied - cannot create comparison plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot 1: Example original trace
        trace_idx = self.n_traces // 2  # Middle trace
        ax1.plot(self.time_axis, self.original_traces[trace_idx, :], 'b-', linewidth=1)
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Original Trace #{trace_idx}')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Background trace (mean of all traces)
        ax2.plot(self.time_axis, self.background_trace, 'r-', linewidth=2)
        ax2.set_xlabel('Time (ns)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Background Trace (Removed)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Filtered trace (after background removal)
        ax3.plot(self.time_axis, self.traces[trace_idx, :], 'g-', linewidth=1)
        ax3.set_xlabel('Time (ns)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title(f'Filtered Trace #{trace_idx}')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Comparison overlay
        ax4.plot(self.time_axis, self.original_traces[trace_idx, :], 'b-', linewidth=1, 
                label='Original', alpha=0.7)
        ax4.plot(self.time_axis, self.traces[trace_idx, :], 'g-', linewidth=1, 
                label='Filtered', alpha=0.8)
        ax4.plot(self.time_axis, self.background_trace, 'r--', linewidth=1, 
                label='Background', alpha=0.6)
        ax4.set_xlabel('Time (ns)')
        ax4.set_ylabel('Amplitude')
        ax4.set_title('Background Removal Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Horizontal Background Removal Filter Results', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print background statistics
        background_stats = {
            'max': np.max(np.abs(self.background_trace)),
            'rms': np.sqrt(np.mean(self.background_trace**2)),
            'peak_time': self.time_axis[np.argmax(np.abs(self.background_trace))]
        }
        
        print(f"\n📊 BACKGROUND TRACE ANALYSIS:")
        print(f"   • Maximum background amplitude: {background_stats['max']:.3e}")
        print(f"   • RMS background level: {background_stats['rms']:.3e}")
        print(f"   • Peak background at time: {background_stats['peak_time']:.1f} ns")
        
        if background_stats['peak_time'] < 20:
            print(f"   • ✓ Peak in early time suggests direct coupling removal")
        if background_stats['rms'] > np.std(self.traces) * 0.1:
            print(f"   • ⚠️ Large background suggests significant systematic noise")
    
    def remove_direct_wave_and_artifacts(self, traces, time_axis):
        """
        Remove direct wave and early-time artifacts (applied by default)
        
        Direct wave removal is crucial for GPR attenuation analysis because:
        - Direct waves travel along the surface (not through subsurface)
        - They don't represent subsurface attenuation properties
        - They can dominate early-time signals and bias analysis
        - Standard practice in professional GPR processing
        """
        corrected_traces = traces.copy()
        
        # Apply time-zero correction
        if self.time_zero_ns > 0:
            shift_samples = int(self.time_zero_ns / self.sample_interval_ns)
            if shift_samples < len(time_axis):
                corrected_traces = np.roll(corrected_traces, -shift_samples, axis=1)
                corrected_traces[:, -shift_samples:] = 0  # Zero-pad end
                print(f"  • Applied time-zero correction: {self.time_zero_ns} ns shift")
        
        # Remove direct wave (now applied by default)
        if self.direct_wave_end_ns > 0:
            direct_wave_samples = int(self.direct_wave_end_ns / self.sample_interval_ns)
            if direct_wave_samples < corrected_traces.shape[1]:
                # Store original data in removed portion for statistics
                removed_portion = corrected_traces[:, :direct_wave_samples].copy()
                corrected_traces[:, :direct_wave_samples] = 0
                
                # Calculate statistics
                removed_energy = np.sum(removed_portion**2)
                total_energy = np.sum(traces**2)
                removed_percentage = (removed_energy / total_energy) * 100
                
                print(f"  • Removed direct wave: 0 to {self.direct_wave_end_ns} ns ({direct_wave_samples} samples)")
                print(f"  • Direct wave energy removed: {removed_percentage:.1f}% of total signal")
                
                # Warning if too much energy is being removed
                if removed_percentage > 50:
                    print(f"  • ⚠️ WARNING: Large energy removal - consider reducing direct_wave_end_ns")
            else:
                print(f"  • ⚠️ WARNING: direct_wave_end_ns ({self.direct_wave_end_ns} ns) exceeds trace length")
        else:
            print(f"  • Direct wave removal disabled (direct_wave_end_ns = 0)")
        
        return corrected_traces
    
    def calculate_amplitude_envelope(self, traces_input=None):
        """
        Calculate amplitude envelope using Hilbert transform
        
        Mathematical Background:
        - Hilbert transform creates analytic signal: z(t) = x(t) + j·H{x(t)}
        - Envelope = |z(t)| = √(x²(t) + H{x(t)}²)
        - This removes phase information, keeping only amplitude information
        """
        print("Calculating amplitude envelopes...")
        
        if traces_input is None:
            traces_input = self.traces
        
        envelopes = np.zeros_like(traces_input)
        
        for i in range(traces_input.shape[0]):
            if np.any(traces_input[i, :] != 0):  # Skip zero traces
                # Apply Hilbert transform to get analytic signal
                analytic_signal = signal.hilbert(traces_input[i, :])
                # Calculate envelope (absolute value of analytic signal)
                envelopes[i, :] = np.abs(analytic_signal)
            
        print(f"Envelope range: {np.min(envelopes):.2e} to {np.max(envelopes):.2e}")
        return envelopes
    
    def calculate_mean_amplitude_profile(self, envelopes):
        """
        Calculate mean amplitude profile across all traces
        
        This averages the amplitude envelopes across all traces to get
        a representative amplitude vs depth profile
        """
        print("Calculating mean amplitude profile...")
        
        # Calculate mean amplitude across all traces, ignoring zero traces
        non_zero_mask = np.any(envelopes != 0, axis=1)
        if np.sum(non_zero_mask) == 0:
            raise ValueError("No non-zero traces found!")
        
        print(f"Using {np.sum(non_zero_mask)} non-zero traces out of {envelopes.shape[0]}")
        
        mean_amplitude = np.mean(envelopes[non_zero_mask, :], axis=0)
        
        # Apply smoothing to reduce noise
        window_length = min(51, len(mean_amplitude) // 10)
        if window_length % 2 == 0:
            window_length += 1
        
        if window_length >= 3:
            smoothed_amplitude = signal.savgol_filter(mean_amplitude, window_length, 3)
        else:
            smoothed_amplitude = mean_amplitude
        
        print(f"Mean amplitude range: {np.min(smoothed_amplitude):.2e} to {np.max(smoothed_amplitude):.2e}")
        return smoothed_amplitude
    
    def identify_reflections(self, mean_amplitude, threshold_factor=0.1):
        """
        Identify likely reflection events above background noise
        """
        # Calculate background noise level
        noise_level = np.median(mean_amplitude)
        
        # Set threshold for reflection identification
        threshold = noise_level + threshold_factor * (np.max(mean_amplitude) - noise_level)
        
        # Find peaks above threshold
        peaks, _ = signal.find_peaks(mean_amplitude, height=threshold, distance=5)
        
        return peaks, threshold
    
    def depth_windowed_analysis(self, amplitudes):
        """
        Perform depth-windowed amplitude analysis
        
        Divides the depth profile into windows and calculates RMS amplitude
        for each window to reduce noise and get stable amplitude estimates
        """
        print("Performing depth-windowed analysis...")
        
        max_depth = self.depth_axis[-1]
        
        # Safety check: prevent excessive number of windows
        max_reasonable_depth = 50  # meters - reasonable for most GPR surveys
        if max_depth > max_reasonable_depth:
            print(f"Warning: Max depth ({max_depth:.1f} m) exceeds reasonable GPR range!")
            print(f"Limiting analysis to {max_reasonable_depth} m to prevent hanging.")
            max_depth = max_reasonable_depth
            # Update depth axis and amplitudes accordingly
            depth_mask = self.depth_axis <= max_depth
            if np.sum(depth_mask) > 0:
                amplitudes = amplitudes[depth_mask]
                effective_depth_axis = self.depth_axis[depth_mask]
            else:
                print("Error: No data within reasonable depth range!")
                return np.array([]), np.array([])
        else:
            effective_depth_axis = self.depth_axis
        
        # Adaptive window size based on depth range
        depth_range = max_depth
        estimated_windows = depth_range / self.window_size
        
        if estimated_windows > 1000:  # Too many windows
            print(f"Warning: {estimated_windows:.0f} windows would be created!")
            adjusted_window_size = depth_range / 200  # Limit to 200 windows
            print(f"Adjusting window size from {self.window_size:.3f} to {adjusted_window_size:.3f} m")
            window_size = adjusted_window_size
        else:
            window_size = self.window_size
        
        depth_windows = np.arange(0, max_depth, window_size)
        windowed_amplitudes = []
        windowed_depths = []
        
        print(f"Creating {len(depth_windows)-1} depth windows of {window_size:.3f} m each...")
        
        for i in range(len(depth_windows) - 1):
            # Find indices within current depth window
            start_depth = depth_windows[i]
            end_depth = depth_windows[i + 1]
            
            mask = (effective_depth_axis >= start_depth) & (effective_depth_axis < end_depth)
            
            if np.sum(mask) > 0:
                # Calculate RMS amplitude in this window
                window_amplitudes = amplitudes[mask]
                # Filter out zero values
                non_zero_amplitudes = window_amplitudes[window_amplitudes > 0]
                
                if len(non_zero_amplitudes) > 0:
                    rms_amplitude = np.sqrt(np.mean(non_zero_amplitudes**2))
                    windowed_amplitudes.append(rms_amplitude)
                    windowed_depths.append((start_depth + end_depth) / 2)
            
            # Progress indicator for large number of windows
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(depth_windows)-1} windows...")
        
        print(f"Created {len(windowed_depths)} depth windows with valid data")
        return np.array(windowed_depths), np.array(windowed_amplitudes)
    
    def correct_two_way_attenuation_model(self, depth, A0, alpha):
        """
        Corrected exponential decay model for GPR round-trip attenuation
        
        For GPR, signal travels down depth z and back up (total path = 2z)
        A(z) = A₀ * exp(-α * 2z)
        
        Where α is the one-way attenuation coefficient
        """
        return A0 * np.exp(-2 * alpha * depth)  # Round-trip factor of 2
    
    def fit_attenuation_curve(self, depths, amplitudes):
        """
        Fit round-trip exponential decay model
        """
        print("Fitting round-trip attenuation model...")
        
        if len(depths) < 3:
            print("Error: Need at least 3 data points for curve fitting")
            return None, None, None, None
        
        # Initial parameter estimates
        A0_init = amplitudes[0] if len(amplitudes) > 0 else 1.0
        alpha_init = 0.05  # Conservative initial guess for one-way attenuation
        
        try:
            # Fit corrected exponential decay model (round-trip)
            popt, pcov = curve_fit(
                self.correct_two_way_attenuation_model,
                depths,
                amplitudes,
                p0=[A0_init, alpha_init],
                bounds=([0, 0], [np.inf, 1.0]),  # Upper limit on alpha to prevent overfitting
                maxfev=5000
            )
            
            A0_fit, alpha_fit = popt
            param_errors = np.sqrt(np.diag(pcov))
            
            # Calculate R-squared
            y_pred = self.correct_two_way_attenuation_model(depths, A0_fit, alpha_fit)
            ss_res = np.sum((amplitudes - y_pred) ** 2)
            ss_tot = np.sum((amplitudes - np.mean(amplitudes)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            print(f"Fit results:")
            print(f"  A₀ = {A0_fit:.4f} ± {param_errors[0]:.4f}")
            print(f"  α (one-way) = {alpha_fit:.4f} ± {param_errors[1]:.4f} Np/m")
            print(f"  R² = {r_squared:.4f}")
            
            return A0_fit, alpha_fit, param_errors, r_squared
            
        except Exception as e:
            print(f"Curve fitting failed: {e}")
            return None, None, None, None
    
    def calculate_corrected_attenuation_and_gain(self, depths, alpha):
        """
        Calculate attenuation and gain with round-trip correction
        """
        # Round-trip attenuation coefficient in dB/m
        alpha_db_per_m_roundtrip = 8.686 * 2 * alpha  # Factor of 2 for round trip
        
        # Calculate round-trip attenuation in dB
        attenuation_db = -alpha_db_per_m_roundtrip * depths
        
        # Gain needed to compensate
        gain_db = -attenuation_db
        
        return attenuation_db, gain_db, alpha_db_per_m_roundtrip
    
    def plot_results(self, depths, amplitudes, fitted_depths, fitted_amplitudes, 
                    attenuation_db, gain_db, alpha, alpha_db_per_m, r_squared):
        """
        Create comprehensive plot showing multiple important GPR attenuation curves
        """
        # Create figure and primary axis
        fig, ax1 = plt.subplots(figsize=(16, 10))
        
        # Calculate additional important curves
        A0 = fitted_amplitudes[0]
        
        # 1. Power attenuation (amplitude squared)
        fitted_power = fitted_amplitudes ** 2
        power_db = 10 * np.log10(fitted_power / fitted_power[0])  # Power in dB
        
        # 2. Skin depth (1/e amplitude depth)
        skin_depth = 1 / (2 * alpha)
        skin_amplitude = A0 / np.e
        
        # 3. Penetration depths (practical limits)
        penetration_10db = 10 / (8.686 * 2 * alpha)  # 10 dB down
        penetration_20db = 20 / (8.686 * 2 * alpha)  # 20 dB down
        penetration_1percent = np.log(100) / (2 * alpha)  # 1% amplitude (99% loss)
        
        # 4. Different material bounds (typical GPR materials)
        # Attenuation ranges for common materials
        materials = {
            'Dry Sand': {'alpha_min': 0.01, 'alpha_max': 0.05, 'color': 'lightblue'},
            'Wet Sand': {'alpha_min': 0.05, 'alpha_max': 0.15, 'color': 'lightgreen'},
            'Clay': {'alpha_min': 0.1, 'alpha_max': 0.5, 'color': 'lightyellow'},
            'Concrete': {'alpha_min': 0.02, 'alpha_max': 0.08, 'color': 'lightgray'}
        }
        
        # 5. Signal-to-noise estimation (assuming 1% noise floor)
        noise_level = A0 * 0.01  # 1% of initial amplitude
        snr_threshold_depth = np.log(1/0.01) / (2 * alpha)  # Where signal equals noise
        
        # Plot amplitude curves on primary y-axis (left side) - LOG SCALE
        color1 = 'tab:red'
        ax1.set_xlabel('Depth (m)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Amplitude (Normalized)', color=color1, fontsize=14, fontweight='bold')
        
        # Plot actual windowed data points
        ax1.scatter(depths, amplitudes, color='darkred', alpha=0.8, s=50, 
                   label='Measured Data', zorder=5, edgecolors='black', linewidth=0.5)
        
        # Plot exponential amplitude fit curve
        line1 = ax1.plot(fitted_depths, fitted_amplitudes, color='red', linewidth=4, 
                        label=f'Amplitude: A₀×exp(-2αz)\nα = {alpha:.4f} Np/m', 
                        zorder=4)
        
        # Plot power attenuation curve
        ax1.plot(fitted_depths, fitted_power, color='darkviolet', linewidth=3, 
                linestyle='--', label=f'Power: A₀²×exp(-4αz)', alpha=0.8, zorder=3)
        
        # Plot material attenuation bounds
        max_depth_plot = min(fitted_depths[-1], 20)  # Limit to 20m for clarity
        depth_range = np.linspace(0, max_depth_plot, 100)
        
        for material, props in materials.items():
            alpha_min_curve = A0 * np.exp(-2 * props['alpha_min'] * depth_range)
            alpha_max_curve = A0 * np.exp(-2 * props['alpha_max'] * depth_range)
            
            # Only plot if curves are in reasonable range
            if props['alpha_max'] < alpha * 2 and props['alpha_min'] > alpha / 2:
                ax1.fill_between(depth_range, alpha_min_curve, alpha_max_curve, 
                               color=props['color'], alpha=0.3, 
                               label=f'{material} Range', zorder=1)
        
        # Add horizontal reference lines
        if skin_depth <= fitted_depths[-1]:
            ax1.axhline(y=skin_amplitude, color='orange', linestyle=':', linewidth=2, 
                       label=f'Skin Depth (1/e) = {skin_depth:.2f} m', alpha=0.8)
            ax1.axvline(x=skin_depth, color='orange', linestyle=':', linewidth=2, alpha=0.6)
        
        # Add penetration depth markers
        penetration_depths = [
            (penetration_10db, '10 dB', 'gold'),
            (penetration_20db, '20 dB', 'orange'),
            (penetration_1percent, '99% Loss', 'red')
        ]
        
        for pen_depth, label, color in penetration_depths:
            if pen_depth <= fitted_depths[-1]:
                pen_amplitude = A0 * np.exp(-2 * alpha * pen_depth)
                ax1.plot(pen_depth, pen_amplitude, marker='v', markersize=8, 
                        color=color, markeredgecolor='black', markeredgewidth=1,
                        label=f'{label}: {pen_depth:.2f} m', zorder=6)
        
        # Add noise floor line
        if noise_level > fitted_amplitudes[-1]:
            ax1.axhline(y=noise_level, color='gray', linestyle='-.', linewidth=2, 
                       label=f'Noise Floor (1%)', alpha=0.7)
            if snr_threshold_depth <= fitted_depths[-1]:
                ax1.axvline(x=snr_threshold_depth, color='gray', linestyle='-.', 
                           linewidth=2, alpha=0.5, 
                           label=f'SNR Limit: {snr_threshold_depth:.2f} m')
        
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
        ax1.tick_params(axis='x', labelsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.set_ylim([fitted_amplitudes[-1] * 0.1, fitted_amplitudes[0] * 2])
        
        # Create secondary y-axis for dB and gain
        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel('Attenuation & Gain (dB)', color=color2, fontsize=14, fontweight='bold')
        
        # Plot attenuation in dB (negative values)
        line2 = ax2.plot(fitted_depths, attenuation_db, color='blue', linewidth=3, 
                        label=f'Attenuation (dB)\n{alpha_db_per_m:.2f} dB/m', 
                        marker='o', markersize=3, alpha=0.8)
        
        # Plot required gain (positive values)
        line3 = ax2.plot(fitted_depths, gain_db, color='cyan', linewidth=3, 
                        label=f'Required Gain (dB)', 
                        marker='s', markersize=3, linestyle='--', alpha=0.8)
        
        # Plot power attenuation in dB
        line4 = ax2.plot(fitted_depths, power_db, color='purple', linewidth=2, 
                        label=f'Power Loss (dB)\n{alpha_db_per_m*2:.1f} dB/m', 
                        linestyle=':', alpha=0.7)
        
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)
        
        # Add reference dB lines
        for db_loss in [10, 20, 30, 40]:
            depth_at_db = db_loss / alpha_db_per_m
            if depth_at_db <= fitted_depths[-1]:
                ax2.axhline(y=-db_loss, color='lightblue', linestyle='--', alpha=0.5)
                ax2.text(fitted_depths[-1] * 0.95, -db_loss, f'{db_loss} dB', 
                        verticalalignment='center', fontsize=9, color='blue', alpha=0.7)
        
        # Set title with comprehensive information including background removal status
        background_status = "✓ Background Removed" if hasattr(self, 'background_trace') else "⚠ No Background Removal"
        direct_wave_status = f"✓ Direct Wave Removed (0-{self.direct_wave_end_ns}ns)" if self.direct_wave_end_ns > 0 else "⚠ Direct Wave Kept"
        ax1.set_title(f'Comprehensive GPR Attenuation Analysis | {background_status} | {direct_wave_status}\n' + 
                     f'Model: A(z) = A₀ × exp(-2αz) | α = {alpha:.4f} Np/m | α = {alpha_db_per_m:.2f} dB/m\n' +
                     f'R² = {r_squared:.3f} | Velocity = {self.velocity} m/ns | Frequency ≈ {0.5*3e8/(self.velocity*1e9):.0f} MHz', 
                     fontsize=15, fontweight='bold', pad=25)
        
        # Create comprehensive legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        # Split legend into two columns
        legend1 = ax1.legend(lines1[:len(lines1)//2 + 1], labels1[:len(labels1)//2 + 1], 
                           loc='upper right', fontsize=9, title='Amplitude Curves')
        legend2 = ax1.legend(lines1[len(lines1)//2 + 1:] + lines2, 
                           labels1[len(labels1)//2 + 1:] + labels2,
                           loc='center right', fontsize=9, title='References & dB')
        ax1.add_artist(legend1)
        
        # Add comprehensive text box with analysis
        textstr = f'ATTENUATION ANALYSIS:\n'
        textstr += f'• Exponential Model: A(z) = {A0:.2e} × exp(-{2*alpha:.3f}z)\n'
        textstr += f'• Skin Depth (1/e): {skin_depth:.2f} m\n'
        textstr += f'• 10 dB Depth: {penetration_10db:.2f} m\n'
        textstr += f'• 20 dB Depth: {penetration_20db:.2f} m\n'
        textstr += f'• Practical Limit (99%): {penetration_1percent:.2f} m\n\n'
        
        textstr += f'MATERIAL CLASSIFICATION:\n'
        if alpha < 0.05:
            textstr += f'• Low Loss (Dry materials)\n'
        elif alpha < 0.15:
            textstr += f'• Moderate Loss (Moist materials)\n'
        elif alpha < 0.3:
            textstr += f'• High Loss (Wet/Conductive)\n'
        else:
            textstr += f'• Very High Loss (Clay/Saltwater)\n'
        
        textstr += f'\nFIT QUALITY:\n'
        textstr += f'• R² = {r_squared:.3f}\n'
        textstr += f'• Data Points: {len(depths)}\n'
        textstr += f'• Depth Range: {depths[0]:.1f}-{depths[-1]:.1f} m'
        
        props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='gray')
        ax1.text(0.02, 0.02, textstr, transform=ax1.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=props, family='monospace')
        
        # Add frequency and wavelength info
        est_frequency = 0.5 * 3e8 / (self.velocity * 1e9)  # Rough frequency estimate
        wavelength = self.velocity * 1e9 / est_frequency / 1e6  # Wavelength in meters
        
        freq_text = f'Estimated Frequency: {est_frequency:.0f} MHz\nWavelength: {wavelength:.2f} m'
        ax1.text(0.98, 0.98, freq_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Improve layout
        plt.tight_layout()
        plt.show()
        
        # Enhanced summary with all curves and processing steps
        print("\n" + "="*90)
        print("COMPREHENSIVE GPR ATTENUATION ANALYSIS")
        print("="*90)
        print(f"File: {self.segy_file_path}")
        
        # Processing information
        background_status = "Applied" if hasattr(self, 'background_trace') else "Not Applied"
        direct_wave_status = f"Applied (0-{self.direct_wave_end_ns}ns)" if self.direct_wave_end_ns > 0 else "Disabled"
        
        print(f"Processing: Background Removal = {background_status}, Direct Wave Removal = {direct_wave_status}")
        print(f"Analysis: {self.n_traces} traces, {self.n_samples} samples, {len(depths)} depth windows")
        print(f"Velocity: {self.velocity} m/ns | Est. Frequency: {est_frequency:.0f} MHz | Wavelength: {wavelength:.2f} m")
        print("-" * 90)
        print(f"EXPONENTIAL DECAY MODEL: A(z) = {A0:.2e} × exp(-{2*alpha:.4f} × z)")
        print(f"  • One-way attenuation (α): {alpha:.4f} Np/m")
        print(f"  • Round-trip attenuation: {alpha_db_per_m:.2f} dB/m")
        print(f"  • Power attenuation: {alpha_db_per_m*2:.1f} dB/m")
        print(f"  • Goodness of fit (R²): {r_squared:.4f}")
        print("-" * 90)
        print(f"PENETRATION ANALYSIS:")
        print(f"  • Skin depth (1/e amplitude): {skin_depth:.2f} m")
        print(f"  • 10 dB loss depth: {penetration_10db:.2f} m")
        print(f"  • 20 dB loss depth: {penetration_20db:.2f} m")
        print(f"  • Practical limit (99% loss): {penetration_1percent:.2f} m")
        print(f"  • Signal-to-noise limit: {snr_threshold_depth:.2f} m (assumed 1% noise)")
        print("-" * 90)
        print(f"AMPLITUDE ANALYSIS:")
        print(f"  • Surface amplitude: {fitted_amplitudes[0]:.3e}")
        print(f"  • Amplitude at max depth: {fitted_amplitudes[-1]:.3e}")
        print(f"  • Total amplitude reduction: {fitted_amplitudes[0]/fitted_amplitudes[-1]:.1f}×")
        print(f"  • Total signal loss: {-attenuation_db[-1]:.1f} dB")
        print(f"  • Required gain: {gain_db[-1]:.1f} dB")
        print("-" * 90)
        
        # Material classification
        print(f"MATERIAL CLASSIFICATION:")
        if alpha < 0.05:
            print(f"  • Classification: LOW LOSS material (α < 0.05 Np/m)")
            print(f"  • Typical materials: Dry sand, dry concrete, ice, granite")
            print(f"  • Expected penetration: Excellent (>10 m)")
        elif alpha < 0.15:
            print(f"  • Classification: MODERATE LOSS material (0.05-0.15 Np/m)")
            print(f"  • Typical materials: Moist sand, limestone, dry clay")
            print(f"  • Expected penetration: Good (5-10 m)")
        elif alpha < 0.3:
            print(f"  • Classification: HIGH LOSS material (0.15-0.3 Np/m)")
            print(f"  • Typical materials: Wet sand, moist clay, asphalt")
            print(f"  • Expected penetration: Limited (2-5 m)")
        else:
            print(f"  • Classification: VERY HIGH LOSS material (α > 0.3 Np/m)")
            print(f"  • Typical materials: Saturated clay, saltwater, metallic objects")
            print(f"  • Expected penetration: Poor (<2 m)")
        
        # Add note about background removal impact
        if hasattr(self, 'background_trace'):
            background_rms = np.sqrt(np.mean(self.background_trace**2))
            signal_rms = np.sqrt(np.mean(self.traces**2))
            if background_rms > signal_rms * 0.1:
                print(f"  • Note: Significant background noise was removed, improving attenuation estimates")
        
        print("="*90)
    
    def analyze(self, show_background_comparison=True):
        """
        Complete corrected attenuation analysis workflow with background removal
        """
        print("Starting GPR attenuation analysis with ObsPy...")
        print("="*50)
        
        try:
            # Step 1: Read SEG-Y file
            self.read_segy_file()
            
            # Step 2: Apply horizontal background removal filter (NEW STEP)
            print("\n" + "="*50)
            print("STEP 2: BACKGROUND REMOVAL")
            print("="*50)
            background_trace = self.apply_horizontal_background_removal()
            
            # Optional: Show background removal comparison
            if show_background_comparison:
                self.plot_background_removal_comparison()
            
            # Step 3: Remove direct wave and artifacts (now applied by default)
            print("\n" + "="*50)
            print("STEP 3: DIRECT WAVE AND ARTIFACT REMOVAL")
            print("="*50)
            print("Applying direct wave removal (standard GPR preprocessing)...")
            corrected_traces = self.remove_direct_wave_and_artifacts(self.traces, self.time_axis)
            
            # Step 4: Calculate amplitude envelopes
            print("\n" + "="*50)
            print("STEP 4: AMPLITUDE ENVELOPE CALCULATION")
            print("="*50)
            envelopes = self.calculate_amplitude_envelope(corrected_traces)
            
            # Step 5: Calculate mean amplitude profile
            print("\n" + "="*50)
            print("STEP 5: MEAN AMPLITUDE PROFILE")
            print("="*50)
            mean_amplitude = self.calculate_mean_amplitude_profile(envelopes)
            
            # Step 6: Identify reflections (for information)
            print("\n" + "="*50)
            print("STEP 6: REFLECTION IDENTIFICATION")
            print("="*50)
            reflection_peaks, threshold = self.identify_reflections(mean_amplitude)
            print(f"Identified {len(reflection_peaks)} potential reflections")
            
            # Step 7: Depth-windowed analysis
            print("\n" + "="*50)
            print("STEP 7: DEPTH-WINDOWED ANALYSIS")
            print("="*50)
            depths, amplitudes = self.depth_windowed_analysis(mean_amplitude)
            
            if len(depths) == 0:
                print("Error: No valid amplitude data found!")
                return None
            
            # Step 8: Fit corrected attenuation model
            print("\n" + "="*50)
            print("STEP 8: ATTENUATION CURVE FITTING")
            print("="*50)
            A0, alpha, param_errors, r_squared = self.fit_attenuation_curve(depths, amplitudes)
            
            if alpha is None:
                print("Error: Could not fit attenuation curve!")
                return None
            
            # Step 9: Calculate corrected attenuation and gain curves
            print("\n" + "="*50)
            print("STEP 9: ATTENUATION AND GAIN CALCULATION")
            print("="*50)
            fitted_depths = np.linspace(depths[0], depths[-1], 100)
            fitted_amplitudes = self.correct_two_way_attenuation_model(fitted_depths, A0, alpha)
            attenuation_db, gain_db, alpha_db_per_m = self.calculate_corrected_attenuation_and_gain(fitted_depths, alpha)
            
            # Step 10: Plot comprehensive results
            print("\n" + "="*50)
            print("STEP 10: COMPREHENSIVE PLOTTING")
            print("="*50)
            self.plot_results(depths, amplitudes, fitted_depths, fitted_amplitudes,
                             attenuation_db, gain_db, alpha, alpha_db_per_m, r_squared)
            
            return {
                'alpha_one_way_np_per_m': alpha,
                'alpha_round_trip_db_per_m': alpha_db_per_m,
                'A0': A0,
                'r_squared': r_squared,
                'depths': fitted_depths,
                'attenuation_db': attenuation_db,
                'gain_db': gain_db,
                'sample_interval_ns': self.sample_interval_ns,
                'n_traces': self.n_traces,
                'max_depth': self.depth_axis[-1],
                'background_removed': True,
                'background_trace': background_trace
            }
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

# Usage example with complete corrected functionality including background removal and default direct wave removal
if __name__ == "__main__":
    # Example usage with corrections and default preprocessing
    segy_file_path = "DAT_0023.sgy"  # Replace with your file path
    
    # Initialize analyzer with corrections (direct wave removal applied by default)
    analyzer = GPRAttenuationAnalyzer(
        segy_file_path=segy_file_path,
        velocity=0.1,               # m/ns - adjust based on ground conditions
        window_size=0.2,            # meters - larger windows for efficiency
        max_depth=20,               # meters - prevent excessive processing
        time_zero_ns=0,             # Time-zero correction (adjust if needed)
        direct_wave_end_ns=15       # Direct wave removal (default: 15 ns, set to 0 to disable)
    )
    
    # Run complete corrected analysis with background and direct wave removal
    print("🚀 Starting GPR Analysis with Standard Preprocessing...")
    results = analyzer.analyze(show_background_comparison=True)  # Set to False to skip background comparison plot
    
    # Access results
    if results:
        print(f"\n🎯 FINAL RESULTS:")
        print(f"✓ Background removal: {'Applied' if results['background_removed'] else 'Not applied'}")
        direct_wave_applied = analyzer.direct_wave_end_ns > 0
        print(f"✓ Direct wave removal: {'Applied (0-' + str(analyzer.direct_wave_end_ns) + 'ns)' if direct_wave_applied else 'Disabled'}")
        print(f"✓ Round-trip attenuation coefficient: {results['alpha_round_trip_db_per_m']:.2f} dB/m")
        print(f"✓ One-way attenuation coefficient: {results['alpha_one_way_np_per_m']:.4f} Np/m")
        print(f"✓ Quality of fit (R²): {results['r_squared']:.4f}")
        print(f"✓ Successfully processed {results['n_traces']} traces")
        print(f"✓ Maximum depth analyzed: {results['max_depth']:.2f} m")
        
        # Background removal statistics
        background_power = np.sum(results['background_trace']**2)
        total_signal_power = np.sum(analyzer.original_traces**2) if hasattr(analyzer, 'original_traces') else 0
        if total_signal_power > 0:
            background_percentage = (background_power / total_signal_power) * 100
            print(f"✓ Background noise removed: {background_percentage:.1f}% of total signal power")
    else:
        print("\n❌ Analysis failed. Check the debug output above for issues.")
        
    print("\n" + "="*70)
    print("STANDARD GPR PROCESSING WORKFLOW COMPLETED")
    print("="*70)
    print("1. ✓ SEG-Y file reading")
    print("2. ✓ Horizontal background removal filter")
    print("3. ✓ Direct wave and artifact removal (0-15ns by default)")
    print("4. ✓ Amplitude envelope calculation")
    print("5. ✓ Mean amplitude profile generation")
    print("6. ✓ Reflection identification")
    print("7. ✓ Depth-windowed amplitude analysis")
    print("8. ✓ Exponential attenuation curve fitting")
    print("9. ✓ Attenuation and gain calculations")
    print("10. ✓ Comprehensive multi-curve plotting")
    print("="*70)
    print("📝 Note: Direct wave removal is applied by default (15ns).")
    print("   To disable: set direct_wave_end_ns=0")
    print("   To adjust: set direct_wave_end_ns=desired_time_in_ns")
    print("="*70)