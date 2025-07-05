import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from scipy import signal
from scipy.optimize import curve_fit
import warnings
import os
warnings.filterwarnings('ignore')

class GPREnvelopeAttenuationAnalyzer:
    """
    GPR Attenuation Analysis using Envelope Detection
    
    The envelope method uses Hilbert transform to extract signal amplitude,
    which is then used to determine attenuation characteristics:
    
    1. Apply Hilbert transform to get analytic signal
    2. Calculate envelope as magnitude of analytic signal  
    3. Average envelopes across traces to get mean amplitude profile
    4. Fit exponential decay model: A(z) = A₀ * exp(-2αz)
    """
    
    def __init__(self, segy_file_path, velocity=0.1, window_size=0.2, 
                 direct_wave_end_ns=15, time_zero_ns=0):
        """
        Initialize the envelope-based attenuation analyzer
        
        Parameters:
        -----------
        segy_file_path : str
            Path to the SEG-Y file
        velocity : float
            GPR wave velocity in m/ns (default: 0.1 m/ns)
        window_size : float
            Depth window size in meters for averaging (default: 0.2 m)
        direct_wave_end_ns : float
            End time of direct wave to exclude (default: 15 ns)
        time_zero_ns : float
            Time-zero correction in nanoseconds (default: 0)
        """
        self.segy_file_path = segy_file_path
        self.velocity = velocity
        self.window_size = window_size
        self.direct_wave_end_ns = direct_wave_end_ns
        self.time_zero_ns = time_zero_ns
        
        # Data containers
        self.traces = None
        self.envelopes = None
        self.time_axis = None
        self.depth_axis = None
        self.sample_interval_ns = None
        self.n_traces = None
        self.n_samples = None
    
    def read_segy_data(self):
        """Read and process SEG-Y file data"""
        print("📁 Reading SEG-Y file...")
        
        if not os.path.exists(self.segy_file_path):
            raise FileNotFoundError(f"SEG-Y file not found: {self.segy_file_path}")
        
        try:
            # Read with ObsPy
            stream = read(self.segy_file_path, format='SEGY', unpack_trace_headers=True)
            
            # Extract basic parameters
            self.n_traces = len(stream)
            trace_lengths = [len(tr.data) for tr in stream]
            self.n_samples = min(trace_lengths)
            
            # Get sampling rate and convert to time interval
            sample_rate = np.mean([tr.stats.sampling_rate for tr in stream])
            
            # GPR-specific corrections for sample rate
            if sample_rate < 1000:  # Likely incorrect units
                # Estimate based on typical GPR parameters
                estimated_time_window = 200  # ns (typical)
                sample_rate = self.n_samples / (estimated_time_window * 1e-9)
                print(f"⚠️  Corrected sampling rate: {sample_rate/1e6:.1f} MHz")
            
            self.sample_interval_ns = 1e9 / sample_rate
            
            # Extract trace data
            self.traces = np.zeros((self.n_traces, self.n_samples))
            for i, trace in enumerate(stream):
                self.traces[i, :] = trace.data[:self.n_samples]
            
            # Create time and depth axes
            self.time_axis = np.arange(self.n_samples) * self.sample_interval_ns
            self.depth_axis = (self.velocity * self.time_axis) / 2  # Two-way travel time
            
            print(f"✅ Loaded {self.n_traces} traces with {self.n_samples} samples each")
            print(f"   Time window: 0 to {self.time_axis[-1]:.1f} ns")
            print(f"   Depth range: 0 to {self.depth_axis[-1]:.2f} m")
            
        except Exception as e:
            print(f"❌ Error reading SEG-Y file: {e}")
            raise
    
    def apply_preprocessing(self):
        """Apply standard GPR preprocessing"""
        print("🔧 Applying preprocessing...")
        
        # 1. Background removal (subtract mean trace)
        mean_trace = np.mean(self.traces, axis=0)
        self.traces = self.traces - mean_trace[np.newaxis, :]
        print("   ✓ Background removal applied")
        
        # 2. Direct wave removal
        if self.direct_wave_end_ns > 0:
            direct_samples = int(self.direct_wave_end_ns / self.sample_interval_ns)
            if direct_samples < self.n_samples:
                self.traces[:, :direct_samples] = 0
                print(f"   ✓ Direct wave removed (0-{self.direct_wave_end_ns} ns)")
        
        # 3. Time-zero correction
        if self.time_zero_ns > 0:
            shift_samples = int(self.time_zero_ns / self.sample_interval_ns)
            if shift_samples < self.n_samples:
                self.traces = np.roll(self.traces, -shift_samples, axis=1)
                self.traces[:, -shift_samples:] = 0
                print(f"   ✓ Time-zero correction applied ({self.time_zero_ns} ns)")
    
    def calculate_envelope_using_hilbert(self):
        """
        Calculate amplitude envelope using Hilbert transform
        
        Process:
        1. Apply Hilbert transform to each trace to get analytic signal
        2. Calculate envelope as |analytic_signal| = sqrt(real² + imag²)
        3. This gives instantaneous amplitude independent of phase
        """
        print("📊 Calculating amplitude envelopes using Hilbert transform...")
        
        self.envelopes = np.zeros_like(self.traces)
        
        for i in range(self.n_traces):
            if np.any(self.traces[i, :] != 0):  # Skip zero traces
                # Apply Hilbert transform to get analytic signal
                # analytic_signal = x(t) + j*H{x(t)} where H{} is Hilbert transform
                analytic_signal = signal.hilbert(self.traces[i, :])
                
                # Calculate envelope: |analytic_signal| = sqrt(real² + imag²)
                self.envelopes[i, :] = np.abs(analytic_signal)
        
        # Calculate statistics
        envelope_max = np.max(self.envelopes)
        envelope_min = np.min(self.envelopes[self.envelopes > 0])
        
        print(f"   ✓ Envelopes calculated for {self.n_traces} traces")
        print(f"   📈 Envelope range: {envelope_min:.2e} to {envelope_max:.2e}")
        
        return self.envelopes
    
    def compute_mean_amplitude_profile(self):
        """
        Compute mean amplitude profile by averaging envelopes across traces
        """
        print("📉 Computing mean amplitude profile...")
        
        # Average envelopes across all non-zero traces
        non_zero_mask = np.any(self.envelopes != 0, axis=1)
        active_traces = np.sum(non_zero_mask)
        
        if active_traces == 0:
            raise ValueError("No non-zero envelope data found!")
        
        mean_envelope = np.mean(self.envelopes[non_zero_mask, :], axis=0)
        
        # Apply smoothing to reduce noise
        window_length = min(51, len(mean_envelope) // 10)
        if window_length % 2 == 0:
            window_length += 1
        
        if window_length >= 3:
            smoothed_envelope = signal.savgol_filter(mean_envelope, window_length, 3)
        else:
            smoothed_envelope = mean_envelope
        
        print(f"   ✓ Mean profile computed from {active_traces} traces")
        print(f"   🎯 Applied Savitzky-Golay smoothing (window: {window_length})")
        
        return smoothed_envelope
    
    def depth_windowed_amplitude_analysis(self, mean_envelope):
        """
        Perform depth-windowed analysis to get stable amplitude estimates
        """
        print("🔍 Performing depth-windowed amplitude analysis...")
        
        max_analysis_depth = min(self.depth_axis[-1], 20)  # Limit to 20m for efficiency
        depth_windows = np.arange(0, max_analysis_depth, self.window_size)
        
        windowed_depths = []
        windowed_amplitudes = []
        
        for i in range(len(depth_windows) - 1):
            start_depth = depth_windows[i]
            end_depth = depth_windows[i + 1]
            
            # Find samples within this depth window
            mask = (self.depth_axis >= start_depth) & (self.depth_axis < end_depth)
            
            if np.sum(mask) > 0:
                window_amplitudes = mean_envelope[mask]
                non_zero_amplitudes = window_amplitudes[window_amplitudes > 0]
                
                if len(non_zero_amplitudes) > 0:
                    # Use RMS amplitude for stability
                    rms_amplitude = np.sqrt(np.mean(non_zero_amplitudes**2))
                    windowed_amplitudes.append(rms_amplitude)
                    windowed_depths.append((start_depth + end_depth) / 2)
        
        windowed_depths = np.array(windowed_depths)
        windowed_amplitudes = np.array(windowed_amplitudes)
        
        print(f"   ✓ Created {len(windowed_depths)} depth windows")
        print(f"   📏 Window size: {self.window_size} m")
        print(f"   🎯 Depth range: {windowed_depths[0]:.2f} to {windowed_depths[-1]:.2f} m")
        
        return windowed_depths, windowed_amplitudes
    
    def fit_exponential_attenuation_model(self, depths, amplitudes):
        """
        Fit exponential attenuation model: A(z) = A₀ * exp(-2αz)
        
        The factor of 2 accounts for round-trip propagation in GPR
        """
        print("📈 Fitting exponential attenuation model...")
        
        if len(depths) < 3:
            raise ValueError("Need at least 3 data points for curve fitting")
        
        def attenuation_model(z, A0, alpha):
            return A0 * np.exp(-2 * alpha * z)
        
        # Initial parameter estimates
        A0_init = amplitudes[0]
        alpha_init = 0.05  # Conservative initial guess
        
        try:
            # Fit the model
            popt, pcov = curve_fit(
                attenuation_model,
                depths,
                amplitudes,
                p0=[A0_init, alpha_init],
                bounds=([0, 0], [np.inf, 1.0]),
                maxfev=5000
            )
            
            A0_fit, alpha_fit = popt
            param_errors = np.sqrt(np.diag(pcov))
            
            # Calculate goodness of fit
            y_pred = attenuation_model(depths, A0_fit, alpha_fit)
            ss_res = np.sum((amplitudes - y_pred) ** 2)
            ss_tot = np.sum((amplitudes - np.mean(amplitudes)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Convert to dB/m for practical use
            alpha_db_per_m = 8.686 * 2 * alpha_fit  # Round-trip dB/m
            
            print(f"   ✅ Model fitted successfully!")
            print(f"   📊 A₀ = {A0_fit:.4f} ± {param_errors[0]:.4f}")
            print(f"   📊 α = {alpha_fit:.4f} ± {param_errors[1]:.4f} Np/m")
            print(f"   📊 α = {alpha_db_per_m:.2f} dB/m (round-trip)")
            print(f"   📊 R² = {r_squared:.4f}")
            
            return A0_fit, alpha_fit, alpha_db_per_m, r_squared, attenuation_model
            
        except Exception as e:
            print(f"   ❌ Curve fitting failed: {e}")
            return None, None, None, None, None
    
    def plot_attenuation_analysis(self, depths, amplitudes, A0, alpha, alpha_db, r_squared, model_func):
        """
        Create comprehensive attenuation analysis plot
        """
        print("🎨 Creating attenuation analysis plots...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Example trace and its envelope
        trace_idx = self.n_traces // 2
        ax1.plot(self.time_axis, self.traces[trace_idx, :], 'b-', linewidth=1, 
                label='Original Trace', alpha=0.7)
        ax1.plot(self.time_axis, self.envelopes[trace_idx, :], 'r-', linewidth=2, 
                label='Envelope (Hilbert)')
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Trace #{trace_idx}: Original vs Envelope')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean envelope profile vs depth
        mean_envelope = self.compute_mean_amplitude_profile()
        ax2.plot(self.depth_axis, mean_envelope, 'g-', linewidth=2, label='Mean Envelope')
        ax2.set_xlabel('Depth (m)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Mean Amplitude Profile')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Windowed amplitude data and fitted curve
        fitted_depths = np.linspace(depths[0], depths[-1], 100)
        fitted_amplitudes = model_func(fitted_depths, A0, alpha)
        
        ax3.scatter(depths, amplitudes, color='red', s=50, alpha=0.8, 
                   label='Windowed Data', zorder=5)
        ax3.plot(fitted_depths, fitted_amplitudes, 'blue', linewidth=3, 
                label=f'Fitted Model\nA(z) = {A0:.2e}×exp(-{2*alpha:.3f}z)', zorder=4)
        
        ax3.set_xlabel('Depth (m)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title(f'Attenuation Curve Fit (R² = {r_squared:.3f})')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Attenuation and gain curves
        attenuation_db = -alpha_db * fitted_depths
        gain_db = -attenuation_db
        
        ax4_twin = ax4.twinx()
        
        line1 = ax4.plot(fitted_depths, attenuation_db, 'red', linewidth=3, 
                        label=f'Attenuation\n{alpha_db:.1f} dB/m')
        line2 = ax4_twin.plot(fitted_depths, gain_db, 'blue', linewidth=3, 
                             label=f'Required Gain\n{alpha_db:.1f} dB/m')
        
        ax4.set_xlabel('Depth (m)')
        ax4.set_ylabel('Attenuation (dB)', color='red')
        ax4_twin.set_ylabel('Required Gain (dB)', color='blue')
        ax4.set_title('Attenuation & Gain vs Depth')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')
        
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='y', labelcolor='red')
        ax4_twin.tick_params(axis='y', labelcolor='blue')
        
        plt.suptitle(f'GPR Envelope-Based Attenuation Analysis\n' + 
                     f'File: {os.path.basename(self.segy_file_path)} | ' +
                     f'Velocity: {self.velocity} m/ns | α = {alpha_db:.1f} dB/m',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Calculate penetration depths
        skin_depth = 1 / (2 * alpha)
        penetration_10db = 10 / alpha_db
        penetration_20db = 20 / alpha_db
        
        return {
            'skin_depth': skin_depth,
            'penetration_10db': penetration_10db,
            'penetration_20db': penetration_20db
        }
    
    def analyze_attenuation(self):
        """
        Complete envelope-based attenuation analysis workflow
        """
        print("🚀 Starting GPR Envelope-Based Attenuation Analysis")
        print("=" * 60)
        
        try:
            # Step 1: Read data
            self.read_segy_data()
            
            # Step 2: Preprocessing
            self.apply_preprocessing()
            
            # Step 3: Calculate envelopes using Hilbert transform
            self.calculate_envelope_using_hilbert()
            
            # Step 4: Compute mean amplitude profile
            mean_envelope = self.compute_mean_amplitude_profile()
            
            # Step 5: Depth-windowed analysis
            depths, amplitudes = self.depth_windowed_amplitude_analysis(mean_envelope)
            
            if len(depths) == 0:
                raise ValueError("No valid amplitude data for analysis!")
            
            # Step 6: Fit attenuation model
            A0, alpha, alpha_db, r_squared, model_func = self.fit_exponential_attenuation_model(depths, amplitudes)
            
            if alpha is None:
                raise ValueError("Could not fit attenuation curve!")
            
            # Step 7: Create comprehensive plots
            penetration_info = self.plot_attenuation_analysis(depths, amplitudes, A0, alpha, 
                                                            alpha_db, r_squared, model_func)
            
            # Step 8: Generate summary report
            self.generate_summary_report(A0, alpha, alpha_db, r_squared, penetration_info)
            
            return {
                'A0': A0,
                'alpha_np_per_m': alpha,
                'alpha_db_per_m': alpha_db,
                'r_squared': r_squared,
                'depths': depths,
                'amplitudes': amplitudes,
                'penetration_info': penetration_info
            }
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_summary_report(self, A0, alpha, alpha_db, r_squared, penetration_info):
        """Generate comprehensive summary report"""
        print("\n" + "=" * 80)
        print("📋 GPR ENVELOPE-BASED ATTENUATION ANALYSIS REPORT")
        print("=" * 80)
        print(f"📁 File: {self.segy_file_path}")
        print(f"📊 Dataset: {self.n_traces} traces × {self.n_samples} samples")
        print(f"⏱️  Time window: 0 to {self.time_axis[-1]:.1f} ns")
        print(f"📏 Depth range: 0 to {self.depth_axis[-1]:.2f} m")
        print(f"🌊 Wave velocity: {self.velocity} m/ns")
        print("-" * 80)
        print("🔬 ENVELOPE ANALYSIS RESULTS:")
        print(f"   📈 Exponential model: A(z) = {A0:.2e} × exp(-{2*alpha:.4f} × z)")
        print(f"   📉 One-way attenuation (α): {alpha:.4f} Np/m")
        print(f"   📉 Round-trip attenuation: {alpha_db:.2f} dB/m")
        print(f"   🎯 Goodness of fit (R²): {r_squared:.4f}")
        print("-" * 80)
        print("🎯 PENETRATION ANALYSIS:")
        print(f"   🔍 Skin depth (1/e): {penetration_info['skin_depth']:.2f} m")
        print(f"   📊 10 dB penetration: {penetration_info['penetration_10db']:.2f} m")
        print(f"   📊 20 dB penetration: {penetration_info['penetration_20db']:.2f} m")
        print("-" * 80)
        print("🧱 MATERIAL CLASSIFICATION:")
        if alpha < 0.05:
            print("   ✅ LOW LOSS material (α < 0.05 Np/m)")
            print("   🏞️  Typical: Dry sand, concrete, ice, granite")
        elif alpha < 0.15:
            print("   ⚠️  MODERATE LOSS material (0.05-0.15 Np/m)")
            print("   🏞️  Typical: Moist sand, limestone, dry clay")
        elif alpha < 0.3:
            print("   ❗ HIGH LOSS material (0.15-0.3 Np/m)")
            print("   🏞️  Typical: Wet sand, moist clay, asphalt")
        else:
            print("   🚨 VERY HIGH LOSS material (α > 0.3 Np/m)")
            print("   🏞️  Typical: Saturated clay, saltwater")
        print("=" * 80)

# Example usage
if __name__ == "__main__":
    # Example with your .sgy file
    segy_file_path = "DAT_0023.SGY"  # Replace with actual file path
    
    # Initialize analyzer
    analyzer = GPREnvelopeAttenuationAnalyzer(
        segy_file_path=segy_file_path,
        velocity=0.1,              # m/ns - adjust for your ground conditions
        window_size=0.2,           # meters
        direct_wave_end_ns=15,     # Remove first 15 ns (direct wave)
        time_zero_ns=0            # Time-zero correction if needed
    )
    
    # Run complete analysis
    results = analyzer.analyze_attenuation()
    
    if results:
        print(f"\n🎉 Analysis completed successfully!")
        print(f"🔍 Attenuation coefficient: {results['alpha_db_per_m']:.2f} dB/m")
        print(f"📊 Model quality (R²): {results['r_squared']:.3f}")
    else:
        print("\n❌ Analysis failed - check file path and data quality")

# # For demonstration with synthetic data (if no .sgy file available)
# def create_synthetic_gpr_demo():
#     """
#     Create synthetic GPR data to demonstrate envelope analysis
#     """
#     print("🎭 Creating synthetic GPR data for demonstration...")
    
#     # Synthetic parameters
#     n_traces = 100
#     n_samples = 512
#     sample_interval_ns = 0.4  # 0.4 ns = 2.5 GHz sampling
#     velocity = 0.1  # m/ns
#     alpha_true = 0.08  # Np/m (true attenuation)
    
#     # Create time and depth axes
#     time_axis = np.arange(n_samples) * sample_interval_ns
#     depth_axis = (velocity * time_axis) / 2
    
#     # Create synthetic traces with exponential decay and noise
#     traces = np.zeros((n_traces, n_samples))
    
#     for i in range(n_traces):
#         # Create base signal with reflections
#         signal_base = np.zeros(n_samples)
        
#         # Add some reflections at different depths
#         reflection_times = [20, 45, 80, 120, 180]  # ns
#         for refl_time in reflection_times:
#             if refl_time < time_axis[-1]:
#                 refl_idx = int(refl_time / sample_interval_ns)
#                 if refl_idx < n_samples:
#                     # Ricker wavelet
#                     t_peak = 10  # ns
#                     freq = 50e6  # 50 MHz
#                     t_wavelet = np.arange(-t_peak, t_peak, sample_interval_ns)
#                     ricker = (1 - 2 * (np.pi * freq * t_wavelet * 1e-9)**2) * \
#                             np.exp(-(np.pi * freq * t_wavelet * 1e-9)**2)
                    
#                     # Add wavelet to signal
#                     start_idx = max(0, refl_idx - len(ricker)//2)
#                     end_idx = min(n_samples, start_idx + len(ricker))
#                     wavelet_start = max(0, len(ricker)//2 - refl_idx)
#                     wavelet_end = wavelet_start + (end_idx - start_idx)
                    
#                     signal_base[start_idx:end_idx] += ricker[wavelet_start:wavelet_end]
        
#         # Apply exponential attenuation
#         attenuation_factor = np.exp(-2 * alpha_true * depth_axis)
#         attenuated_signal = signal_base * attenuation_factor
        
#         # Add noise
#         noise_level = 0.1 * np.max(np.abs(attenuated_signal))
#         noise = np.random.normal(0, noise_level, n_samples)
        
#         traces[i, :] = attenuated_signal + noise
    
#     # Save as simple numpy arrays for demo
#     demo_data = {
#         'traces': traces,
#         'time_axis': time_axis,
#         'depth_axis': depth_axis,
#         'sample_interval_ns': sample_interval_ns,
#         'velocity': velocity,
#         'alpha_true': alpha_true,
#         'n_traces': n_traces,
#         'n_samples': n_samples
#     }
    
#     print(f"✅ Created synthetic dataset:")
#     print(f"   📊 {n_traces} traces × {n_samples} samples")
#     print(f"   ⏱️  Time window: 0 to {time_axis[-1]:.1f} ns")
#     print(f"   📏 Depth range: 0 to {depth_axis[-1]:.2f} m")
#     print(f"   🎯 True attenuation: {alpha_true:.3f} Np/m ({8.686*2*alpha_true:.1f} dB/m)")
    
#     return demo_data

# # Uncomment to run synthetic demo:
# demo_data = create_synthetic_gpr_demo()