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
    
    def __init__(self, segy_file_path, velocity=0.1, window_size=0.1):
        """
        Initialize the analyzer
        
        Parameters:
        -----------
        segy_file_path : str
            Path to the SEG-Y file
        velocity : float
            GPR wave velocity in m/ns (default: 0.1 m/ns for typical soil)
        window_size : float
            Depth window size in meters for amplitude averaging
        """
        self.segy_file_path = segy_file_path
        self.velocity = velocity  # m/ns
        self.window_size = window_size  # meters
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
        
        # Extract sampling information
        sampling_rates = [tr.stats.sampling_rate for tr in self.stream]
        self.sample_rate = np.mean(sampling_rates)  # Hz
        
        # Convert to time interval in nanoseconds
        self.sample_interval_ns = 1e9 / self.sample_rate  # ns
        
        print(f"Sampling rate: {self.sample_rate:.2f} Hz")
        print(f"Sample interval: {self.sample_interval_ns:.2f} ns")
        
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
        depth_windows = np.arange(0, max_depth, self.window_size)
        windowed_amplitudes = []
        windowed_depths = []
        
        for i in range(len(depth_windows) - 1):
            # Find indices within current depth window
            start_depth = depth_windows[i]
            end_depth = depth_windows[i + 1]
            
            mask = (self.depth_axis >= start_depth) & (self.depth_axis < end_depth)
            
            if np.sum(mask) > 0:
                # Calculate RMS amplitude in this window
                window_amplitudes = amplitudes[mask]
                # Filter out zero values
                non_zero_amplitudes = window_amplitudes[window_amplitudes > 0]
                
                if len(non_zero_amplitudes) > 0:
                    rms_amplitude = np.sqrt(np.mean(non_zero_amplitudes**2))
                    windowed_amplitudes.append(rms_amplitude)
                    windowed_depths.append((start_depth + end_depth) / 2)
        
        print(f"Created {len(windowed_depths)} depth windows")
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
        Create comprehensive plots of the analysis results
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Raw amplitude vs depth with exponential fit
        ax1.semilogy(depths, amplitudes, 'bo-', alpha=0.7, label='Measured Amplitude')
        ax1.semilogy(fitted_depths, fitted_amplitudes, 'r-', linewidth=2, 
                    label=f'Exponential Fit (α = {alpha:.4f} Np/m)')
        ax1.set_xlabel('Depth (m)')
        ax1.set_ylabel('Amplitude (linear scale)')
        ax1.set_title('GPR Amplitude vs Depth\n(Semi-log plot)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        # Plot 2: Linear scale amplitude
        ax2.plot(depths, amplitudes, 'bo-', alpha=0.7, label='Measured Amplitude')
        ax2.plot(fitted_depths, fitted_amplitudes, 'r-', linewidth=2, 
                label=f'Exponential Fit')
        ax2.set_xlabel('Depth (m)')
        ax2.set_ylabel('Amplitude (linear scale)')
        ax2.set_title('GPR Amplitude vs Depth\n(Linear scale)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Attenuation in dB
        ax3.plot(fitted_depths, attenuation_db, 'g-', linewidth=2, label='Attenuation')
        ax3.set_xlabel('Depth (m)')
        ax3.set_ylabel('Attenuation (dB)')
        ax3.set_title(f'Signal Attenuation vs Depth\n(α = {alpha_db_per_m:.2f} dB/m)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Gain compensation needed
        ax4.plot(fitted_depths, gain_db, 'purple', linewidth=2, label='Gain Compensation')
        ax4.set_xlabel('Depth (m)')
        ax4.set_ylabel('Gain (dB)')
        ax4.set_title(f'Required Gain Compensation vs Depth\n(α = {alpha_db_per_m:.2f} dB/m)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\n" + "="*60)
        print("ATTENUATION ANALYSIS SUMMARY")
        print("="*60)
        print(f"Attenuation coefficient (α): {alpha:.4f} Np/m")
        print(f"Attenuation coefficient (α): {alpha_db_per_m:.2f} dB/m")
        print(f"Goodness of fit (R²): {r_squared:.4f}")
        print(f"GPR velocity used: {self.velocity} m/ns")
        print(f"Sample interval: {self.sample_interval_ns:.2f} ns")
        print(f"Maximum analysis depth: {fitted_depths[-1]:.2f} m")
        print(f"Total attenuation at max depth: {attenuation_db[-1]:.1f} dB")
        print(f"Required gain at max depth: {gain_db[-1]:.1f} dB")
        print("="*60)
    
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
    segy_file_path = "DAT_0023.SGY"  # Replace with your file path
    
    # Initialize analyzer with appropriate parameters
    # velocity: GPR wave velocity in m/ns (typical values: 0.06-0.15 m/ns)
    # window_size: depth window for amplitude averaging in meters
    analyzer = GPRAttenuationAnalyzer(
        segy_file_path=segy_file_path,
        velocity=0.1,  # m/ns - adjust based on your ground conditions
        window_size=0.05  # meters - adjust based on your depth resolution needs
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
