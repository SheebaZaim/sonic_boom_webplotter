"""
debug_test_runner.py
Diagnostic version with detailed error catching
"""

import sys
import traceback

print("\n" + "="*70)
print("DEBUG MODE - CHECKING IMPORTS")
print("="*70)

try:
    print("▶ Importing numpy...")
    import numpy as np
    print("  ✓ numpy OK")
except Exception as e:
    print(f"  ✗ numpy FAILED: {e}")
    sys.exit(1)

try:
    print("▶ Importing pandas...")
    import pandas as pd
    print("  ✓ pandas OK")
except Exception as e:
    print(f"  ✗ pandas FAILED: {e}")
    sys.exit(1)

try:
    print("▶ Importing matplotlib...")
    import matplotlib.pyplot as plt
    print("  ✓ matplotlib OK")
except Exception as e:
    print(f"  ✗ matplotlib FAILED: {e}")
    sys.exit(1)

try:
    print("▶ Importing custom modules...")
    from propagate_frequency_domain import propagate_linear_fft, apply_turbulence_envelope
    from nonlinear_correction import nonlinear_correction
    print("  ✓ Custom modules OK")
except Exception as e:
    print(f"  ✗ Custom modules FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("ALL IMPORTS SUCCESSFUL - RUNNING TEST")
print("="*70)

import os
import argparse

def load_csv(csvfile):
    """Load CSV with error checking"""
    try:
        print(f"\n▶ Loading CSV: {csvfile}")
        if not os.path.exists(csvfile):
            raise FileNotFoundError(f"File not found: {csvfile}")
        
        df = pd.read_csv(csvfile)
        print(f"  ✓ File loaded")
        print(f"  ✓ Columns: {list(df.columns)}")
        print(f"  ✓ Rows: {len(df)}")
        
        if 't' not in df.columns or 'p' not in df.columns:
            raise ValueError(f"CSV must have 't' and 'p' columns. Found: {list(df.columns)}")
        
        t = df['t'].values
        p = df['p'].values
        
        print(f"  ✓ Time range: {t[0]:.6f} to {t[-1]:.6f} s")
        print(f"  ✓ Pressure range: {np.min(p):.4f} to {np.max(p):.4f} Pa")
        
        return t, p
    
    except Exception as e:
        print(f"  ✗ ERROR loading CSV: {e}")
        traceback.print_exc()
        raise

def compute_perceived_loudness(t, p):
    """Simplified PLdB calculation with error checking"""
    try:
        print("\n▶ Computing Perceived Loudness...")
        dt = t[1] - t[0]
        p_ac = p - np.mean(p)
        p_abs = np.abs(p_ac)
        
        exponent = 2.67
        integrand = p_abs ** exponent
        integral = np.trapz(integrand, dx=dt)
        
        if integral > 0:
            PLdB = 10 * np.log10(integral) + 80.0
            print(f"  ✓ PLdB = {PLdB:.5f} dB")
        else:
            PLdB = -np.inf
            print(f"  ⚠ PLdB = -inf (integral = 0)")
        
        return PLdB
    
    except Exception as e:
        print(f"  ✗ ERROR computing PLdB: {e}")
        traceback.print_exc()
        raise

def run_single_case(input_csv, distance, azimuth=0):
    """Run single propagation case with detailed error tracking"""
    
    try:
        # Load input
        t, p = load_csv(input_csv)
        
        # Resample
        print(f"\n▶ Resampling to 200kHz...")
        fs_req = 200000.0
        dt = 1.0 / fs_req
        t_uniform = np.arange(t[0], t[-1], dt)
        p_uniform = np.interp(t_uniform, t, p)
        print(f"  ✓ Resampled to {len(t_uniform)} points")
        
        # Linear propagation
        print(f"\n▶ Linear propagation (FFT)...")
        c0 = 340.0
        t_out, p_lin = propagate_linear_fft(
            t_uniform, p_uniform, distance,
            c0=c0, temp_c=20.0, rh=50.0, p_pa=101325.0
        )
        print(f"  ✓ Linear propagation complete")
        print(f"  ✓ Output pressure range: {np.min(p_lin):.4f} to {np.max(p_lin):.4f} Pa")
        
        # Nonlinear correction
        print(f"\n▶ Nonlinear correction (Burgers solver)...")
        dx = 5.0
        dt_burgers = 1e-5
        nu = 5e-5
        n_steps = int(distance / dx)
        print(f"  • n_steps = {n_steps}")
        print(f"  • This will take ~{n_steps/100:.1f} seconds")
        
        p_calc = nonlinear_correction(
            p_lin, dx=dx, dt=dt_burgers, nu=nu, n_steps=n_steps
        )
        print(f"  ✓ Nonlinear correction complete")
        
        # Calculate PLdB
        PLdB = compute_perceived_loudness(t_out, p_calc)
        
        # Save results
        print(f"\n▶ Saving results...")
        os.makedirs('outputs', exist_ok=True)
        output_csv = f'outputs/debug_az{int(azimuth)}_propagated.csv'
        pd.DataFrame({'t': t_out, 'p': p_calc}).to_csv(output_csv, index=False)
        print(f"  ✓ Saved to: {output_csv}")
        
        # Plot
        print(f"\n▶ Creating plot...")
        plt.figure(figsize=(10, 5))
        plt.plot(t_out * 1000, p_calc, 'b-', linewidth=1.5, label=f'Az={azimuth}°')
        plt.xlabel('Time (ms)')
        plt.ylabel('Pressure (Pa)')
        plt.title(f'Ground Signature - Az={azimuth}° | PLdB={PLdB:.2f} dB')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_file = f'outputs/debug_az{int(azimuth)}_plot.png'
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"  ✓ Plot saved to: {plot_file}")
        
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"Azimuth:  {azimuth}°")
        print(f"PLdB:     {PLdB:.5f} dB")
        print(f"Distance: {distance} m")
        print("="*70)
        
        return PLdB
    
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR OCCURRED")
        print("="*70)
        print(f"Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("="*70)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--distance', type=float, default=15760.0)
    parser.add_argument('--azimuth', type=float, default=0.0)
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("DIAGNOSTIC TEST RUNNER")
    print("="*70)
    print(f"Input:    {args.input}")
    print(f"Distance: {args.distance} m")
    print(f"Azimuth:  {args.azimuth}°")
    print("="*70)
    
    result = run_single_case(args.input, args.distance, args.azimuth)
    
    if result is not None:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed - see errors above")
        sys.exit(1)