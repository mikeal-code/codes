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
    Complete GPR attenuation analysis from SEG-Y files using ObsPy
    
    Physics Background:
    - GPR signals attenuate exponentially with depth: A(z) = A₀ * e^(-αz)
    - α is the attenuation coefficient (Np/m or dB/m)
    - Two-way travel time: depth = (velocity × time) / 2
    - Attenuation in dB: α_dB = 8.686 × α_Np (conversion factor)
    """
    
    def __init__(self, segy_file_path, velocity=0.1, window_size=0.2, max_depth=None):
        """
        Initialize the analyzer
        
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
        """
        self.segy_file_path = segy_file_path
        self.velocity = velocity  # m/ns
        self.window_size = window_size  # meters
        self.max_depth = max_depth  # meters
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
    
    def calculate_amplitude_envelope(self):
        """
        Calculate amplitude envelope using Hilbert transform
        
        Mathematical Background:
        - Hilbert transform creates analytic signal: z(t) = x(t) + j·H{x(t)}
        - Envelope = |z(t)| = √(x²(t) + H{x(t)}²)
        - This removes phase information, keeping only amplitude information
        """
        print("Calculating amplitude envelopes...")
        
        envelopes = np.zeros_like(self.traces)
        
        for i in range(self.n_traces):
            if np.any(self.traces[i, :] != 0):  # Skip zero traces
                # Apply Hilbert transform to get analytic signal
                analytic_signal = signal.hilbert(self.traces[i, :])
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
        
        print(f"Using {np.sum(non_zero_mask)} non-zero traces out of {self.n_traces}")
        
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
    
    def exponential_decay_model(self, depth, A0, alpha):
        """
        Exponential decay model for GPR attenuation
        
        Mathematical Model:
        A(z) = A₀ * exp(-α * z)
        
        Where:
        - A(z): Amplitude at depth z
        - A₀: Initial amplitude at surface
        - α: Attenuation coefficient (Np/m)
        - z: Depth (m)
        """
        return A0 * np.exp(-alpha * depth)
    
    def fit_attenuation_curve(self, depths, amplitudes):
        """
        Fit exponential decay model to amplitude data
        
        Uses non-linear least squares to fit the exponential decay model
        and extract the attenuation coefficient
        """
        print("Fitting exponential decay model...")
        
        if len(depths) < 3:
            print("Error: Need at least 3 data points for curve fitting")
            return None, None, None, None
        
        # Initial parameter estimates
        A0_init = amplitudes[0] if len(amplitudes) > 0 else 1.0
        alpha_init = 0.1  # Initial guess for attenuation coefficient
        
        try:
            # Fit exponential decay model
            popt, pcov = curve_fit(
                self.exponential_decay_model,
                depths,
                amplitudes,
                p0=[A0_init, alpha_init],
                bounds=([0, 0], [np.inf, np.inf]),
                maxfev=5000
            )
            
            A0_fit, alpha_fit = popt
            
            # Calculate parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))
            
            # Calculate R-squared
            y_pred = self.exponential_decay_model(depths, A0_fit, alpha_fit)
            ss_res = np.sum((amplitudes - y_pred) ** 2)
            ss_tot = np.sum((amplitudes - np.mean(amplitudes)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            print(f"Fit results:")
            print(f"  A₀ = {A0_fit:.4f} ± {param_errors[0]:.4f}")
            print(f"  α = {alpha_fit:.4f} ± {param_errors[1]:.4f} Np/m")
            print(f"  R² = {r_squared:.4f}")
            
            return A0_fit, alpha_fit, param_errors, r_squared
            
        except Exception as e:
            print(f"Curve fitting failed: {e}")
            return None, None, None, None
    
    def calculate_attenuation_and_gain(self, depths, alpha):
        """
        Calculate attenuation and gain in dB
        
        Mathematical Formulations:
        
        1. Attenuation in dB:
           Attenuation(dB) = 20 * log₁₀(A(z)/A₀)
           For exponential decay: A(z) = A₀ * e^(-αz)
           Therefore: Attenuation(dB) = 20 * log₁₀(e^(-αz)) = -20 * α * z * log₁₀(e)
           Since log₁₀(e) ≈ 0.434: Attenuation(dB) = -8.686 * α * z
        
        2. Gain in dB (compensation needed):
           Gain(dB) = -Attenuation(dB) = 8.686 * α * z
        
        3. Conversion factor:
           1 Neper = 8.686 dB
        """
        
        # Convert attenuation coefficient from Np/m to dB/m
        alpha_db_per_m = 8.686 * alpha  # dB/m
        
        # Calculate attenuation in dB as function of depth
        attenuation_db = -alpha_db_per_m * depths  # Negative because it's loss
        
        # Calculate gain needed to compensate (positive)
        gain_db = -attenuation_db
        
        return attenuation_db, gain_db, alpha_db_per_m
    
    def plot_results(self, depths, amplitudes, fitted_depths, fitted_amplitudes, 
                    attenuation_db, gain_db, alpha, alpha_db_per_m, r_squared):
        """
        Create a single plot with dual y-axes showing attenuation and gain vs depth
        """
        # Create figure and primary axis
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Plot attenuation on primary y-axis (left side)
        color1 = 'tab:red'
        ax1.set_xlabel('Depth (m)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Attenuation (dB)', color=color1, fontsize=14, fontweight='bold')
        line1 = ax1.plot(fitted_depths, attenuation_db, color=color1, linewidth=3, 
                        label=f'Attenuation (α = {alpha_db_per_m:.2f} dB/m)', marker='o', markersize=4)
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
        ax1.tick_params(axis='x', labelsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Create secondary y-axis for gain
        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel('Gain (dB)', color=color2, fontsize=14, fontweight='bold')
        line2 = ax2.plot(fitted_depths, gain_db, color=color2, linewidth=3, 
                        label=f'Gain Compensation', marker='s', markersize=4)
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)
        
        # Set title
        ax1.set_title(f'GPR Attenuation and Gain vs Depth\n' + 
                     f'α = {alpha_db_per_m:.2f} dB/m, R² = {r_squared:.3f}, ' +
                     f'Velocity = {self.velocity} m/ns', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Add combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
        
        # Add text box with key results
        textstr = f'Attenuation Coefficient:\n  {alpha:.4f} Np/m\n  {alpha_db_per_m:.2f} dB/m\n\n'
        textstr += f'At Max Depth ({fitted_depths[-1]:.1f} m):\n'
        textstr += f'  Attenuation: {attenuation_db[-1]:.1f} dB\n'
        textstr += f'  Required Gain: {gain_db[-1]:.1f} dB\n\n'
        textstr += f'Quality of Fit: R² = {r_squared:.3f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        # Improve layout
        plt.tight_layout()
        
        # Add zero reference lines
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.show()
        
        # Print summary
        print("\n" + "="*70)
        print("GPR ATTENUATION ANALYSIS SUMMARY")
        print("="*70)
        print(f"File: {self.segy_file_path}")
        print(f"Traces processed: {self.n_traces}")
        print(f"Samples per trace: {self.n_samples}")
        print(f"Sample interval: {self.sample_interval_ns:.2f} ns")
        print(f"GPR velocity: {self.velocity} m/ns")
        print("-" * 70)
        print(f"Attenuation coefficient (α): {alpha:.4f} Np/m")
        print(f"Attenuation coefficient (α): {alpha_db_per_m:.2f} dB/m")
        print(f"Goodness of fit (R²): {r_squared:.4f}")
        print(f"Maximum analysis depth: {fitted_depths[-1]:.2f} m")
        print(f"Total attenuation at max depth: {attenuation_db[-1]:.1f} dB")
        print(f"Required gain at max depth: {gain_db[-1]:.1f} dB")
        print("="*70)
    
    def analyze(self):
        """
        Complete attenuation analysis workflow
        """
        print("Starting GPR attenuation analysis with ObsPy...")
        print("="*50)
        
        try:
            # Step 1: Read SEG-Y file
            self.read_segy_file()
            
            # Step 2: Calculate amplitude envelopes
            envelopes = self.calculate_amplitude_envelope()
            
            # Step 3: Calculate mean amplitude profile
            mean_amplitude = self.calculate_mean_amplitude_profile(envelopes)
            
            # Step 4: Depth-windowed analysis
            depths, amplitudes = self.depth_windowed_analysis(mean_amplitude)
            
            if len(depths) == 0:
                print("Error: No valid amplitude data found!")
                return None
            
            # Step 5: Fit exponential decay model
            A0, alpha, param_errors, r_squared = self.fit_attenuation_curve(depths, amplitudes)
            
            if alpha is None:
                print("Error: Could not fit attenuation curve!")
                return None
            
            # Step 6: Calculate attenuation and gain curves
            fitted_depths = np.linspace(depths[0], depths[-1], 100)
            fitted_amplitudes = self.exponential_decay_model(fitted_depths, A0, alpha)
            attenuation_db, gain_db, alpha_db_per_m = self.calculate_attenuation_and_gain(fitted_depths, alpha)
            
            # Step 7: Plot results
            self.plot_results(depths, amplitudes, fitted_depths, fitted_amplitudes,
                             attenuation_db, gain_db, alpha, alpha_db_per_m, r_squared)
            
            return {
                'alpha_np_per_m': alpha,
                'alpha_db_per_m': alpha_db_per_m,
                'A0': A0,
                'r_squared': r_squared,
                'depths': fitted_depths,
                'attenuation_db': attenuation_db,
                'gain_db': gain_db,
                'sample_interval_ns': self.sample_interval_ns,
                'n_traces': self.n_traces,
                'max_depth': self.depth_axis[-1]
            }
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

# Usage example
if __name__ == "__main__":
    # Example usage
    segy_file_path = "DAT_0075.sgy"  # Replace with your file path
    
    # Initialize analyzer with appropriate parameters
    # velocity: GPR wave velocity in m/ns (typical values: 0.06-0.15 m/ns)
    # window_size: depth window for amplitude averaging in meters
    # max_depth: maximum depth to analyze (prevents excessive processing)
    analyzer = GPRAttenuationAnalyzer(
        segy_file_path=segy_file_path,
        velocity=0.1,        # m/ns - adjust based on your ground conditions
        window_size=0.2,     # meters - larger windows for efficiency
        max_depth=20         # meters - prevent excessive processing
    )
    
    # Run complete analysis
    results = analyzer.analyze()
    
    # Access results
    if results:
        print(f"\nFinal Results:")
        print(f"Attenuation coefficient: {results['alpha_db_per_m']:.2f} dB/m")
        print(f"Quality of fit (R²): {results['r_squared']:.4f}")
        print(f"Successfully processed {results['n_traces']} traces")
        print(f"Maximum depth analyzed: {results['max_depth']:.2f} m")