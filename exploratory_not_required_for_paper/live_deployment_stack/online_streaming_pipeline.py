import time
import sqlite3
import numpy as np
from scipy.signal import butter, lfilter

FS = 250             
WINDOW_SIZE = 500    
STEP_SIZE = 50       
CHANNELS = 4         
DB_NAME = "bci_production_analytics.db"

class LeakyIntegrateAndFireNeuron:
    def __init__(self, neuron_id):
        self.id = neuron_id
        self.v = -70.0        
        self.v_rest = -70.0   
        self.v_thresh = -55.0 
        self.v_reset = -75.0  
        self.tau_m = 20.0     
        self.last_spike_time = -999.0

    def evaluate_ms_step(self, current_input, t_ms):
        leak = -(self.v - self.v_rest)
        self.v += (leak + current_input) * (1.0 / self.tau_m)
        if self.v >= self.v_thresh:
            self.v = self.v_reset
            self.last_spike_time = t_ms
            return True
        return False

class HebbianSTDPConnection:
    def __init__(self, pre_id, post_id, w_init=4.5):
        self.pre_id = pre_id
        self.post_id = post_id
        self.weight = w_init
        self.tau_stdp = 20.0  
        self.w_max = 12.0     
        self.w_min = 0.5      

    def apply_plastic_rule(self, pre_spike, post_spike, t_ms):
        if pre_spike == t_ms:
            dt = t_ms - post_spike
            if dt > 0: self.weight -= 1.1 * np.exp(-dt / self.tau_stdp)  
        if post_spike == t_ms:
            dt = t_ms - pre_spike
            if dt > 0: self.weight += 1.5 * np.exp(-dt / self.tau_stdp)  
        self.weight = np.clip(self.weight, self.w_min, self.w_max)

def establish_storage_environment():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS user_activities (activity_id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT DEFAULT CURRENT_TIMESTAMP, task_label TEXT, total_duration_secs INTEGER)")
    cursor.execute("CREATE TABLE IF NOT EXISTS cognitive_telemetry (telemetry_id INTEGER PRIMARY KEY AUTOINCREMENT, activity_id INTEGER, timeline_offset REAL, decoded_state TEXT, stability_index REAL, synaptic_weight_ch0 REAL)")
    conn.commit()
    conn.close()

def create_causal_bandpass_filter(lowcut=1.0, highcut=45.0, fs=250, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    from scipy.signal import lfilter_zi
    return b, a, lfilter_zi(b, a)

def launch_production_bci_pipeline(active_user_task, session_ticks=60):
    establish_storage_environment()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO user_activities (task_label, total_duration_secs) VALUES (?, 0)", (active_user_task,))
    active_activity_id = cursor.lastrowid
    conn.commit()
    
    sensory_layer = [LeakyIntegrateAndFireNeuron(neuron_id=i) for i in range(3)]
    executive_center = LeakyIntegrateAndFireNeuron(neuron_id=3)
    synaptic_highways = [HebbianSTDPConnection(pre_id=i, post_id=3) for i in range(3)]
    
    ring_buffer = np.zeros((WINDOW_SIZE, CHANNELS))
    b, a, filter_zi = create_causal_bandpass_filter(fs=FS)
    channel_memories = [filter_zi.copy() for _ in range(CHANNELS)]
    
    sample_counter = 0
    epochs_evaluated = 0
    
    print(f"🔌 Starting live processing simulation for: {active_user_task}")
    while epochs_evaluated < session_ticks:
        sample_counter += 1
        current_ms = sample_counter * (1000 // FS)
        is_focusing = active_user_task in ["Studying Math", "Software Development"]
        sensory_currents = [24.0 if is_focusing else 3.0, np.random.normal(4, 2), np.random.normal(3, 1)]
        
        ms_spikes = np.zeros(CHANNELS)
        for ms_offset in range(4):
            t_sub = current_ms + ms_offset
            for idx, neuron in enumerate(sensory_layer):
                if neuron.evaluate_ms_step(sensory_currents[idx], t_sub):
                    ms_spikes[idx] += 1
                    synaptic_highways[idx].apply_plastic_rule(neuron.last_spike_time, executive_center.last_spike_time, t_sub)
            integrated_synaptic_input = sum([synaptic_highways[i].weight for i in range(3) if t_sub - sensory_layer[i].last_spike_time <= 2])
            if executive_center.evaluate_ms_step(integrated_synaptic_input, t_sub):
                ms_spikes[3] += 1
                for idx, neuron in enumerate(sensory_layer):
                    synaptic_highways[idx].apply_plastic_rule(neuron.last_spike_time, executive_center.last_spike_time, t_sub)

        raw_hardware_frame = np.zeros(CHANNELS)
        for ch in range(3): raw_hardware_frame[ch] = (sensory_layer[ch].v * 0.4) + (ms_spikes[ch] * 35.0)
        raw_hardware_frame[3] = (executive_center.v * 0.5) + (ms_spikes[3] * 55.0)
        
        filtered_frame = np.zeros(CHANNELS)
        for ch in range(CHANNELS):
            f_val, channel_memories[ch] = lfilter(b, a, [raw_hardware_frame[ch]], zi=channel_memories[ch])
            filtered_frame[ch] = f_val[0]
            
        ring_buffer = np.roll(ring_buffer, -1, axis=0)
        ring_buffer[-1, :] = filtered_frame
        
        if sample_counter >= WINDOW_SIZE and sample_counter % STEP_SIZE == 0:
            epochs_evaluated += 1
            timeline_seconds = epochs_evaluated * (STEP_SIZE / FS)
            mean_sig = np.mean(ring_buffer, axis=1)
            fft_vals = np.abs(np.fft.rfft(mean_sig))
            freqs = np.fft.rfftfreq(len(mean_sig), d=1.0/FS)
            alpha_power = np.sum(fft_vals[(freqs >= 8) & (freqs <= 12)]) + 1e-5
            beta_power = np.sum(fft_vals[(freqs >= 13) & (freqs <= 30)]) + 1e-5
            ratio = beta_power / alpha_power
            state_decision = "ENGAGED" if ratio > 0.85 else "RESTING"
            cursor.execute("INSERT INTO cognitive_telemetry (activity_id, timeline_offset, decoded_state, stability_index, synaptic_weight_ch0) VALUES (?, ?, ?, ?, ?)", (active_activity_id, timeline_seconds, state_decision, 85.0, synaptic_highways[0].weight))
            conn.commit()
    conn.close()
    print("✅ Complete.")

if __name__ == "__main__":
    launch_production_bci_pipeline("Studying Math", session_ticks=5)