"""
Virtual Human Brain Simulator v4 (LSL Edition) - Calibrator Fixed
=================================================================
Updated to natively handle the '--manual' command-line flag for structured
clinical calibration loops alongside its default stochastic autonomous behavior.
"""

import time
import threading
import random
import sys
import numpy as np

try:
    from pylsl import StreamInfo, StreamOutlet
except ImportError:
    raise ImportError("Install pylsl: pip install pylsl")

FS       = 250
CHANNELS = 4

# ── Per-session individual subject parameters ─────────────────────────────────
RNG = np.random.default_rng(int(time.time()) % 100000)
SUBJECT = {
    "alpha_peak_hz":    float(RNG.uniform(8.5, 12.5)),
    "amplitude_scale":  float(RNG.uniform(0.7, 1.4)),
    "beta_reactivity":  float(RNG.uniform(0.8, 1.6)),
    "baseline_fatigue": float(RNG.uniform(0.0, 0.3)),
    "beta_peak_hz":     float(RNG.uniform(18.0, 25.0)),
}

print("=" * 60)
print("  VIRTUAL HUMAN BRAIN SIMULATOR v4 (LSL)")
print("=" * 60)
print(f"  Alpha peak:       {SUBJECT['alpha_peak_hz']:.1f} Hz")
print(f"  Amplitude scale:  {SUBJECT['amplitude_scale']:.2f}x")
print(f"  Beta reactivity:  {SUBJECT['beta_reactivity']:.2f}x")
print(f"  Baseline fatigue: {SUBJECT['baseline_fatigue']:.2f}")
print(f"  Beta peak:        {SUBJECT['beta_peak_hz']:.1f} Hz")
print("=" * 60)


# ── AR(2) stochastic oscillator ───────────────────────────────────────────────
class AR2Oscillator:
    """
    Discrete-time damped harmonic oscillator driven by white noise.
    x[t] = 2r·cos(ω)·x[t-1] - r²·x[t-2] + σ·ε[t]
    """
    def __init__(self, pole_hz, damping, noise_std=1.0):
        self.omega  = 2 * np.pi * pole_hz / FS
        self.r      = damping
        self.sigma  = noise_std
        self.x1     = 0.0
        self.x2     = 0.0

    def step(self):
        x = (2 * self.r * np.cos(self.omega) * self.x1
             - self.r ** 2 * self.x2
             + self.sigma * np.random.standard_normal())
        self.x2 = self.x1
        self.x1 = x
        return x

    def warm_up(self, n=500):
        for _ in range(n):
            self.step()
        return self

    @property
    def long_run_std(self):
        return self.sigma / np.sqrt(1 - self.r**2 + 1e-9)


# ── Oscillators (subject-specific peak frequencies) ───────────────────────────
osc_theta = AR2Oscillator(SUBJECT["alpha_peak_hz"] * 0.6 + 0.25, 0.90, noise_std=6.0).warm_up()
osc_alpha = AR2Oscillator(SUBJECT["alpha_peak_hz"] - 0.25,        0.93, noise_std=8.0).warm_up()
osc_beta  = AR2Oscillator(SUBJECT["beta_peak_hz"]  + 1.25,        0.88, noise_std=4.0).warm_up()


# ── Pink noise floor (AR(1) cascade) ─────────────────────────────────────────
class PinkNoiseGenerator:
    def __init__(self, n_octaves=4, std=2.5):
        self.states = np.zeros(n_octaves)
        self.poles  = np.array([0.99, 0.97, 0.93, 0.85])[:n_octaves]
        self.std    = std / n_octaves

    def step(self):
        w            = np.random.standard_normal(len(self.states)) * self.std
        self.states  = self.poles * self.states + w
        return float(np.sum(self.states))

bg_gens = [PinkNoiseGenerator() for _ in range(CHANNELS)]


# ── Alpha spindle gate (Markov on/off) ────────────────────────────────────────
class SpindleGate:
    def __init__(self):
        self.on           = False
        self.envelope     = 0.0
        self.ramp_step    = 1.0 / (FS * 0.3)
        self.direction    = 0

    def step(self, state):
        p_on  = 0.0015 if state == "REST" else 0.0002
        p_off = 0.0008

        if not self.on:
            if random.random() < p_on:
                self.on = True
                self.direction = 1
        else:
            if self.direction == 1:
                self.envelope = min(1.0, self.envelope + self.ramp_step)
                if self.envelope >= 1.0:
                    self.direction = 0
            elif self.direction == 0 and random.random() < p_off:
                self.direction = -1
            elif self.direction == -1:
                self.envelope = max(0.0, self.envelope - self.ramp_step)
                if self.envelope <= 0.0:
                    self.on = False
        return self.envelope

spindle_gate = SpindleGate()


# ── Fatigue modulator ─────────────────────────────────────────────────────────
class ArousalModulator:
    def __init__(self, baseline):
        self.baseline     = baseline
        self.slow_phase   = random.uniform(0, 2 * np.pi)
        self.slow_freq    = random.uniform(0.005, 0.02)
        self.task_fatigue = 0.0
        self.t            = 0

    def step(self, state):
        self.t += 1 / FS
        slow    = 0.15 * np.sin(2 * np.pi * self.slow_freq * self.t + self.slow_phase)
        if state == "WORKLOAD":
            self.task_fatigue = min(0.5, self.task_fatigue + 0.00005)
        else:
            self.task_fatigue = max(0.0, self.task_fatigue - 0.0001)
        return float(np.clip(self.baseline + slow + self.task_fatigue, 0, 1))

arousal = ArousalModulator(SUBJECT["baseline_fatigue"])


# ── Slow DC drift (sweat battery simulation) ──────────────────────────────────
drift     = np.zeros(CHANNELS)


# ── Spatial mixing matrix (4 sources → 4 frontal channels) ───────────────────
A_MIX = np.array([
    [0.85, 0.20, 0.10, 0.60],
    [0.80, 0.22, 0.12, 0.60],
    [0.55, 0.45, 0.35, 0.55],
    [0.52, 0.48, 0.38, 0.55],
])


# ── Blink model ───────────────────────────────────────────────────────────────
class BlinkModel:
    def __init__(self):
        self.active   = False
        self.envelope = np.array([])
        self.idx      = 0
        self.countdown = int(np.random.exponential(4.0) * FS)

    def step(self):
        self.countdown -= 1
        if self.countdown <= 0 and not self.active:
            dur            = int(random.uniform(0.08, 0.14) * FS)
            amp            = random.uniform(60, 120)
            self.envelope  = amp * np.hanning(dur)
            self.idx       = 0
            self.active    = True
            self.countdown = int(np.random.exponential(4.0) * FS)

        if self.active:
            val = self.envelope[self.idx] if self.idx < len(self.envelope) else 0.0
            self.idx += 1
            if self.idx >= len(self.envelope):
                self.active = False
            return np.array([val, val * 0.95, val * 0.15, val * 0.12])
        return np.zeros(CHANNELS)

blink_model = BlinkModel()


# ── Muscle model ──────────────────────────────────────────────────────────────
class MuscleModel:
    def __init__(self):
        self.active    = False
        self.remaining = 0
        self.amplitude = 1.0
        self.side      = 0
        self.countdown = int(np.random.exponential(30.0) * FS)

    def step(self):
        self.countdown -= 1
        if self.countdown <= 0 and not self.active:
            self.remaining = int(random.uniform(0.3, 0.7) * FS)
            self.amplitude = random.uniform(25, 45)
            self.side      = random.randint(0, 1)
            self.active    = True
            self.countdown = int(np.random.exponential(30.0) * FS)

        if self.active:
            a, b = np.random.standard_normal(2)
            hf   = (a - b) * self.amplitude * 0.6
            self.remaining -= 1
            if self.remaining <= 0:
                self.active = False
            w = np.zeros(CHANNELS)
            if self.side == 0:
                w[0] = 1.0; w[2] = 0.7
            else:
                w[1] = 1.0; w[3] = 0.7
            return w * hf
        return np.zeros(CHANNELS)

muscle_model = MuscleModel()


# ── State Control Engine (Handles --manual flag parsing) ──────────────────────
STATES        = ["REST", "WORKLOAD", "MOTOR"]
current_state = "REST"
state_lock    = threading.Lock()
is_manual_mode = "--manual" in sys.argv

def state_machine_autonomous():
    global current_state
    T_MATRIX = np.array([
        [0.95, 0.04, 0.01],  # from REST
        [0.05, 0.93, 0.02],  # from WORKLOAD
        [0.08, 0.02, 0.90],  # from MOTOR
    ])
    print("\n🤖 Autonomous Markov state machine active (shifting organically)...")
    state_idx = 0
    while True:
        time.sleep(1.0)
        row   = T_MATRIX[state_idx]
        state_idx = np.random.choice(3, p=row)
        with state_lock:
            new = STATES[state_idx]
            if new != current_state:
                current_state = new
                print(f"\n  [STATE → {current_state}]")

def state_machine_manual():
    global current_state
    print("\n🎛️ Manual Calibration Control Active.")
    print("   Type 1, 2, or 3 and press Enter to shift brain states:")
    print("   [1] REST  |  [2] WORKLOAD  |  [3] MOTOR\n")
    while True:
        try:
            cmd = input("Select state [1-3]: ").strip()
            if cmd == "1":
                with state_lock: current_state = "REST"
                print("🧠 Brain profile locked to: REST")
            elif cmd == "2":
                with state_lock: current_state = "WORKLOAD"
                print("🧠 Brain profile locked to: WORKLOAD")
            elif cmd == "3":
                with state_lock: current_state = "MOTOR"
                print("🧠 Brain profile locked to: MOTOR")
            else:
                print("❌ Invalid selection. Please enter 1, 2, or 3.")
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception:
            continue

# Launch execution thread based on configuration parameter
if is_manual_mode:
    threading.Thread(target=state_machine_manual, daemon=True).start()
else:
    threading.Thread(target=state_machine_autonomous, daemon=True).start()


# ── LSL stream setup ──────────────────────────────────────────────────────────
info   = StreamInfo('VirtualBrain_EEG_v4', 'EEG', CHANNELS, FS, 'float32', 'vbrain_v4_001')
outlet = StreamOutlet(info)

print("📡 LSL stream 'VirtualBrain_EEG_v4' active at 250 Hz.")
print("   Connect calibration_orchestrator_v2.py and realtime_inference_engine_v2.py\n")

sample_idx  = 0
start_time  = time.time()

try:
    while True:
        with state_lock:
            state = current_state

        S  = SUBJECT["amplitude_scale"]
        BR = SUBJECT["beta_reactivity"]

        # ── Generate neural source signals ────────────────────────────────────
        spindle_env = spindle_gate.step(state)
        fatigue     = arousal.step(state)

        theta_raw = osc_theta.step() / (osc_theta.long_run_std + 1e-9)
        alpha_raw = osc_alpha.step() / (osc_alpha.long_run_std + 1e-9)
        beta_raw  = osc_beta.step()  / (osc_beta.long_run_std  + 1e-9)

        if state == "REST":
            theta_src = S * 10.0 * (1 + 0.2 * fatigue)              * theta_raw
            alpha_src = S * 18.0 * (1 + spindle_env * 0.8) * (1 - 0.3 * fatigue) * alpha_raw
            beta_src  = S * 3.0  * BR                                * beta_raw
        elif state == "WORKLOAD":
            theta_src = S * 12.0 * (1 + 0.3 * fatigue)              * theta_raw
            alpha_src = S * 4.0  * (1 - 0.5 * fatigue)              * alpha_raw
            beta_src  = S * 10.0 * BR * (1 - 0.2 * fatigue)         * beta_raw
        else:  # MOTOR
            theta_src = S * 8.0                                       * theta_raw
            alpha_src = S * 10.0 * (1 - 0.3 * spindle_env)          * alpha_raw
            beta_src  = S * 7.0  * BR                                * beta_raw

        # Mix through spatial matrix
        bg      = np.array([g.step() for g in bg_gens])
        sources = np.array([theta_src, alpha_src, beta_src, 0.0])
        eeg     = A_MIX @ sources + bg

        # Slow drift
        drift[:] = 0.9998 * drift + 0.05 * np.random.standard_normal(CHANNELS)
        eeg      += drift

        # Artifacts
        eeg += blink_model.step()
        eeg += muscle_model.step()

        # Push to LSL
        outlet.push_sample(eeg.tolist())
        sample_idx += 1

        if sample_idx % (FS * 30) == 0:
            if is_manual_mode:
                # Use a clean telemetry string to avoid formatting collisions during menu inputs
                print(f"\n[Telemetry] t={sample_idx//FS:4d}s | Active State={state:<8} | Fatigue={fatigue:.2f}")
            else:
                print(f"  t={sample_idx//FS:4d}s  state={state:<9}  fatigue={fatigue:.2f}  spindle={spindle_gate.on}", end="\r")

        # Precise timing
        expected = start_time + sample_idx / FS
        delta    = expected - time.time()
        if delta > 0:
            time.sleep(delta)

except KeyboardInterrupt:
    print(f"\n🛑 Stopped after {sample_idx//FS}s.")