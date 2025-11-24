
def ssvep_cca_classification(data, duration_sec, frequencies=None,
                             n_harmonics=2, min_confidence=0.5, return_scores=False):
    """
    Perform SSVEP frequency classification on 0.5-second EEG data chunks using CCA
    with OSCAR-style preprocessing (adaptive spatial filtering + bandpass).
    """
    if frequencies is None:
        frequencies = [6, 10, 12, 15]  # Default stimulation frequencies

    # Select occipital channels (Pz, PO7, Oz, PO8)
    selected_channels = [4,5,6,7]
    # print(f"Raw data shape (before selection): {data.shape}")
    
    # 1. Select the channels for processing
    data_selected_channels = data[selected_channels, :]
    # print(f"Selected data shape: {data_selected_channels.shape}")

    # Validate input
    n_channels_selected, n_samples_chunk = data_selected_channels.shape # Use n_channels_selected here
    sfreq = n_samples_chunk / duration_sec # Calculate sfreq based on chunk samples

    # --------------------------
    # OSCAR-style preprocessing
    # --------------------------

    def oscar_preprocess(eeg_data, fs):
        """Simplified OSCAR-like processing pipeline"""
        # 1. Common Average Reference (spatial filtering)
        car_data = eeg_data - np.mean(eeg_data, axis=0, keepdims=True)

        # 2. Bandpass filter (5-30Hz for SSVEP)
        b, a = butter(4, [5, 30], btype='band', fs=fs)
        filtered_data = filtfilt(b, a, car_data)

        return filtered_data

    # Apply preprocessing to the entire selected data chunk *before* the CCA windowing loop.
    # This is where 'processed_selected_channels_data' is defined and populated.
    processed_selected_channels_data = oscar_preprocess(data_selected_channels, sfreq)
    # print(f"Shape of processed data (selected channels): {processed_selected_channels_data.shape}")


    # --------------------------
    # CCA Processing (0.5s windows)
    # --------------------------

    # CCA will now operate on the already processed 'processed_selected_channels_data'
    # Note: n_samples is the total samples in the chunk for CCA processing, which is now n_samples_chunk
    window_length = int(0.5 * sfreq)
    num_windows = int(np.floor(n_samples_chunk / window_length)) # Use n_samples_chunk for calculation
    t = np.arange(window_length) / sfreq

    cca = CCA(n_components=1)
    predicted_freqs = []
    all_scores = []

    for window_idx in range(num_windows):
        start = window_idx * window_length
        end = start + window_length
        # Extract window from the *already processed* data
        eeg_window = processed_selected_channels_data[:, start:end].T # CCA expects (samples, channels)

        max_corr = -np.inf
        best_freq = None
        window_scores = {}

        for freq in frequencies:
            # Reference signal with harmonics
            reference = np.zeros((window_length, n_harmonics * 2))
            for i in range(1, n_harmonics + 1):
                reference[:, 2*(i-1)] = np.sin(2 * np.pi * freq * i * t)
                reference[:, 2*(i-1) + 1] = np.cos(2 * np.pi * freq * i * t)

            # CCA computation
            cca.fit(eeg_window, reference)
            U, V = cca.transform(eeg_window, reference)
            corr = np.corrcoef(U.T, V.T)[0, 1]

            window_scores[freq] = corr
            if corr > max_corr:
                max_corr = corr
                best_freq = freq

        all_scores.append(window_scores)
        if max_corr >= min_confidence:
            predicted_freqs.append(best_freq)

    # Majority voting
    final_prediction = float(mode(predicted_freqs).mode) if predicted_freqs else None

    # 'processed_selected_channels_data' is now properly defined and accessible here
    if return_scores:
        return all_scores, predicted_freqs, final_prediction, processed_selected_channels_data
    return predicted_freqs, final_prediction, processed_selected_channels_data

# The print_cca_scores function remains unchanged
def print_cca_scores(all_scores, predicted_freqs):
    """Display CCA scores for each window"""
    print("\nCCA Scores for 0.5-second Windows:")
    print("--------------------------------")
    for i, (scores, pred) in enumerate(zip(all_scores, predicted_freqs)):
        print(f"Window {i+1}:")
        for freq, score in sorted(scores.items()):
            mark = " *" if freq == pred else ""
            print(f"  {freq:2}Hz: {score:.3f}{mark}")
        print()

def _calculate_beta_alpha_ratio(eeg_window, fs):
    """Calculate beta/alpha power ratio for all channels"""
    alpha_powers = []
    beta_powers = []
    
    for ch_data in eeg_window:
        freqs, psd = welch(ch_data, fs=fs, nperseg=len(ch_data))
        
        # Alpha power (8-13Hz)
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        alpha_power = np.trapz(psd[alpha_mask], freqs[alpha_mask])
        
        # Beta power (13-30Hz)
        beta_mask = (freqs >= 13) & (freqs <= 30)
        beta_power = np.trapz(psd[beta_mask], freqs[beta_mask])
        
        alpha_powers.append(alpha_power)
        beta_powers.append(beta_power)
    
    # Add small epsilon to prevent division by zero
    return float(np.mean(beta_powers) / (np.mean(alpha_powers) + 1e-10))


def Focus_classification(data, model_path, fs=250, min_confidence=0.5):
    """
    Perform focus state classification on 1-second EEG windows using beta/alpha ratio.
    
    Args:
        data: EEG data array (channels × samples)
        model_path: Path to saved model (.joblib)
        fs: Sampling frequency (default 250Hz)
        min_confidence: Minimum probability threshold (default 0.5)
    
    Returns:
        tuple: (prediction, beta_alpha_ratio, confidence)
    """
    # 1. Channel selection (occipital channels)
    selected_channels = [0, 1, 2, 3, 4, 5, 6, 7]
    eeg_data = data[selected_channels, :]
    # Debug print for 'eeg_data'
    #print(f"DEBUG: Focus_classification - 'eeg_data' type: {type(eeg_data)}, shape: {eeg_data.shape if isinstance(eeg_data, np.ndarray) else 'N/A'}")
    # 2. Validate input is exactly 1 second
    expected_samples = fs * 1
    if eeg_data.shape[1] != expected_samples:
        raise ValueError(f"Expected {expected_samples} samples, got {eeg_data.shape[1]}")

    # 3. OSCAR preprocessing
    def oscar_filter(data_to_filter): # Renamed 'data' to 'data_to_filter' to avoid confusion with outer 'data'
        # Debug print inside oscar_filter
        #print(f"DEBUG: oscar_filter - Input 'data_to_filter' type: {type(data_to_filter)}, shape: {data_to_filter.shape if isinstance(data_to_filter, np.ndarray) else 'N/A'}")
        
        car_data = data_to_filter - np.mean(data_to_filter, axis=0, keepdims=True)
        b, a = butter(4, [8, 30], btype='band', fs=fs)
        filtered_data = filtfilt(b, a, car_data)
        
        # Debug print for output of oscar_filter
        #print(f"DEBUG: oscar_filter - Output 'filtered_data' type: {type(filtered_data)}, shape: {filtered_data.shape if isinstance(filtered_data, np.ndarray) else 'N/A'}")
        return filtered_data
    

    processed_data = oscar_filter(eeg_data)
    # Debug print for 'processed_data'
    #print(f"DEBUG: Focus_classification - 'processed_data' type: {type(processed_data)}, shape: {processed_data.shape if isinstance(processed_data, np.ndarray) else 'N/A'}")
    
    try:
        # Calculate beta/alpha ratio
        beta_alpha = _calculate_beta_alpha_ratio(processed_data, fs)
        
        # Load model and predict (properly formatted input)
        model = load(model_path)
        
        # Convert to 2D array and ensure no feature names
        X = np.array([[beta_alpha]])  # Shape: (1, 1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = model.predict(X)[0]  # Get single prediction
            proba = model.predict_proba(X)[0]  # Get probabilities
            confidence = np.max(proba)
            
            if confidence >= min_confidence:
                return int(pred), float(beta_alpha), float(confidence)
            return None, float(beta_alpha), float(confidence)
            
    except Exception as e:
        print(f"Focus classification error: {str(e)}")
        return None, None, None

# ALL placement logic including focus meter done: final 
#Final Version with all the logic and shape placement--------------------------------------------------------------------------- #
# BCI Drawing Interface – minimal patch for two final issues                                                                     #
#                                                                                                                                 #
# • MOD-1 (already present) : keypad digits 1-4 map to shapes                                                                   #
# • MOD-2 (already present) : keypad-ENTER accepted for size = Large                                                            #
# • MOD-3 (already present) : shape auto-drawn during “Shape Placement”                                                         #
# • MOD-4 (already present) : Static yellow dot is now drawn on *every* frame                                                 #
#                         (so it is always visible, not just in the Static_Dot                                                   #
#                         phase).                                                                                                 #
# • MOD-5 **IMPROVED EEG CHUNKING/STORAGE**: EEG data is continuously pulled to avoid LSL buffer overflow,                        #
#                         but collected/stored into the phase buffer (buf) ONLY for "Selection_Phase" and "Static_Dot".         #
#                                                                                                                                 #
# No other logic was touched:                                                                                                    #
#   - Selecting “Do Nothing” still shows the text feedback and never places                                                      #
#     a shape on the canvas.                                                                                                 #
#   - Matplotlib plotting blocks remain for visualizing collected EEG data.                                                       #
# --------------------------------------------------------------------------- #

from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop
import time, pygame
import numpy as np
import matplotlib.pyplot as plt # Kept for visualizing collected EEG data
import sys
import random # Add this line
import math   # Add this line
## Final version
import numpy as np
from scipy.signal import butter, filtfilt, welch
import warnings
from joblib import load
## Final Version, do not touch

import numpy as np
from sklearn.cross_decomposition import CCA
from scipy.stats import mode
from scipy.signal import butter, filtfilt

# ---------- LSL SET‑UP -------------------------------------------------------
print("Connecting to Unicorn EEG stream ...")
eeg_streams = resolve_byprop('name', 'UN-2023.04.55', timeout=5)
if not eeg_streams:
    raise RuntimeError("Unicorn EEG stream 'UN-2023.04.55' not found!")
eeg_inlet = StreamInlet(eeg_streams[0])
print("✓ EEG stream connected.")

# LSL StreamInfo for markers (as per original code and your preference)
marker_info = StreamInfo(
    name='Unicorn_Events', type='annotations',
    channel_count=1, nominal_srate=0, channel_format='string', source_id='bci_v1'
)
marker_outlet = StreamOutlet(marker_info)
print("✓ Marker outlet 'Unicorn_Events' created.")

def send_marker(code): marker_outlet.push_sample([str(code)], time.time())
event_marker = send_marker
# -----------------------------------------------------------------------------

# EEG parameters (kept as they define the data structure)
# These are essential for understanding the collected data
EEG_SAMPLING_RATE = 250  # Hz, assumed from your Unicorn device
NUM_EEG_CHANNELS = 17    # Total channels in Unicorn stream (confirm this matches your data)


# -------------- PYGAME SETUP -------------------------------------------------
pygame.init()
screen_width, screen_height = 1200, 800
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("BCI Drawing Interface")

WHITE, BLACK = (255,255,255), (0,0,0)
GRAY, RED, GREEN, YELLOW, BLUE = (100,100,100), (255,0,0), (0,255,0), (255,255,0), (0,0,255)

circle_pos, rect_pos, triangle_pos, do_nothing_pos = (150,100),(150,280),(150,460),(150,640)
canvas_x, canvas_y, canvas_w, canvas_h = 500, 100, 500, 600

font = pygame.font.Font(None,36); small_font = pygame.font.Font(None,24)

flicker_info = {
    "Circle":    {"color": RED,   "pos": circle_pos,    "hz": 10, "last": time.time(), "on": True, "key": pygame.K_1},
    "Rectangle": {"color": BLUE,  "pos": rect_pos,      "hz": 12, "last": time.time(), "on": True, "key": pygame.K_2},
    "Triangle":  {"color": GREEN, "pos": triangle_pos,  "hz": 15, "last": time.time(), "on": True, "key": pygame.K_3},
    "Do Nothing":{"color": WHITE, "pos": do_nothing_pos,"hz":  6, "last": time.time(), "on": True, "key": pygame.K_4},
}


key_to_shape_map = {
    pygame.K_1:"Circle",     pygame.K_KP1:"Circle",
    pygame.K_2:"Rectangle",  pygame.K_KP2:"Rectangle",
    pygame.K_3:"Triangle",   pygame.K_KP3:"Triangle",
    pygame.K_4:"Do Nothing", pygame.K_KP4:"Do Nothing",
}
SPACE_KEYS = {pygame.K_SPACE}
ENTER_KEYS = {pygame.K_RETURN, pygame.K_KP_ENTER}

selected_shape = None
last_placed_shape, last_placed_size = None, None
focus_val, focus_dir = 0,1

def interpolate(val):
    r=int(RED[0]+(GREEN[0]-RED[0])*val/100)
    g=int(RED[1]+(GREEN[1]-RED[1])*val/100)
    b=int(RED[2]+(GREEN[2]-RED[2])*val/100)
    return r,g,b

def draw_flicker(phase):
    now=time.time()
    for name,info in flicker_info.items():
        col=GRAY
        if phase=="Selection_Phase":
            if now-info["last"]>=0.5/info["hz"]:
                info["on"]=not info["on"]; info["last"]=now
            col=info["color"] if info["on"] else GRAY
        elif phase in ["Shape Selected","Static_Dot","Size selected"]:
            col=info["color"] if selected_shape==name else GRAY
        
        if name=="Circle":
            pygame.draw.circle(screen,col,info["pos"],50)
            screen.blit(small_font.render("Circle",True,BLACK),
                         small_font.render("Circle",True,BLACK).get_rect(center=info["pos"]))
        elif name=="Rectangle":
            pygame.draw.rect(screen,col,(info["pos"][0]-50,info["pos"][1]-30,100,60))
            screen.blit(small_font.render("Rectangle",True,BLACK),
                         small_font.render("Rectangle",True,BLACK).get_rect(center=info["pos"]))
        elif name=="Triangle":
            pts=[(info["pos"][0],info["pos"][1]-50),
                 (info["pos"][0]-50,info["pos"][1]+50),
                 (info["pos"][0]+50,info["pos"][1]+50)]
            pygame.draw.polygon(screen,col,pts)
            screen.blit(small_font.render("Triangle",True,BLACK),
                         small_font.render("Triangle",True,BLACK).get_rect(center=(info["pos"][0],info["pos"][1]+20)))
        else:
            pygame.draw.rect(screen,col,(info["pos"][0]-70,info["pos"][1]-20,140,40))
            screen.blit(small_font.render("Do Nothing",True,BLACK),
                         small_font.render("Do Nothing",True,BLACK).get_rect(center=info["pos"]))

# Static dot properties (derived from your draw_static_dot function)
static_dot_center = (350, 350)
static_dot_radius = 100

moving_dot_radius = 10
moving_dot_color = (0, 0, 0) # Black
moving_dot_x = 350
moving_dot_y = 350

# Moving dot velocity components
dx = 0.0
dy = 0.0
dot_speed = 10.0 # Pixels per frame

# Control flag for dot movement
is_moving_dot_active = False

# --- Your modified draw_static_dot function ---
def draw_static_dot():
    # Draw the main static dot
    pygame.draw.circle(screen, YELLOW, (350, 350), 100)

    # Draw the "Focus Here" text
    text_surface = small_font.render(" ", True, BLACK)
    text_rect = text_surface.get_rect(center=(350, 350))
    screen.blit(text_surface, text_rect)

    # Draw the moving black dot (ball)
    pygame.draw.circle(screen, moving_dot_color, (int(moving_dot_x), int(moving_dot_y)), moving_dot_radius)

def draw_canvas():
    pygame.draw.rect(screen,WHITE,(canvas_x,canvas_y,canvas_w,canvas_h))
    pygame.draw.rect(screen,BLACK,(canvas_x,canvas_y,canvas_w,canvas_h),2)

def place_shape(shape,is_large=False,y_off=0):
    if shape=="Do Nothing": return
    cx,cy=canvas_x+250,canvas_y+200+y_off
    col=flicker_info[shape]["color"]
    if shape=="Circle":
        pygame.draw.circle(screen,col,(cx,cy),70 if is_large else 50)
    elif shape=="Rectangle":
        w,h=(140,80) if is_large else (100,60)
        pygame.draw.rect(screen,col,(cx-w/2,cy-h/2,w,h))
    elif shape=="Triangle":
        off=70 if is_large else 50
        pts=[(cx,cy-(90 if is_large else 70)),(cx-off,cy+off/2),(cx+off,cy+off/2)]
        pygame.draw.polygon(screen,col,pts)

def draw_focus_meter(current_phase, focus_prediction):
    mx, my, mw, mh = 1050, 150, 40, 400
    pygame.draw.rect(screen, WHITE, (mx, my, mw, mh), 2) # Outer rectangle

    if current_phase == "Size selected":
        if focus_prediction == 1: # High focus (focused)
            fill_color = GREEN
        elif focus_prediction == 0: # Low focus (not focused)
            fill_color = RED
        else: # No definitive prediction or neutral
            fill_color = GRAY 
        
        # Fill the entire meter with the determined color
        pygame.draw.rect(screen, fill_color, (mx, my, mw, mh))
    else:
        # Static and neutral (gray)
        pygame.draw.rect(screen, GRAY, (mx, my, mw, mh))

session_plan = [
    ("Session Start", 5, "Start"),
    ("Look at the flickering shape", 5, "Selection_Phase"),
    ("Shape Selected", 3, "Shape Selected"),
    ("Next Focus on the DOT ", 5, "Break"),
    ("Focus/Unfocus - Look at the Dot", 7, "Static_Dot"),
    ("Size selected", 5, "Size selected"),
    ("", 5, "Shape Placement"),
    ("Session END", 1, "END")
]

phase_markers = {
    "Start": ("1", "2"),
    "Selection_Phase": ("3", "4"),
    "Shape Selected": ("5", "6"),
    "Break": ("7", "8"),
    "Static_Dot": ("9", "10"),
    "Size selected": ("11", "12"),
    "Shape Placement": ("13", "14"),
    "END": ("15", "16")
}

# --- NEW: map each flicker‑frequency to its logical shape ---------------
hz_to_shape = {info["hz"]: name for name, info in flicker_info.items()}


def main():
    global selected_shape, last_placed_shape, last_placed_size
    global moving_dot_x, moving_dot_y, dx, dy, is_moving_dot_active
    placed_shapes, placed_large = [], []
    eeg_segments = {}
    eeg_processed_ssvep_plotting_segments = {}
    clock = pygame.time.Clock()
    pending_shape = None    

    # Processing parameters
    CHUNK_SIZE = int(EEG_SAMPLING_RATE * 0.50)
    current_predictions = [] 
    focus_predictions = [] 
    FOCUS_WINDOW = int(EEG_SAMPLING_RATE * 1.0)
    current_focus_prediction = None
    focus_size_trigger = None  # Stores size decision from focus (Large/Small)

    for caption, dur, phase in session_plan:
        t0 = time.time()
        current_caption = caption
        
        # Initialize buffers for this phase
        eeg_buffer_SSVEP = []
        eeg_raw_data_for_saving = []
        eeg_processed_ssvep_current_phase_buffer = []
        eeg_buffer_Focus = []

        # Phase setup
        if phase in phase_markers:
            event_marker(phase_markers[phase][0])

        if phase == "Selection_Phase":
            selected_shape = last_placed_shape = last_placed_size = None
            current_predictions = []
            classification_type = "SSVEP"
        elif phase == "Shape Selected":
            if pending_shape:
                selected_shape = pending_shape
                event_marker(f"Shape_Selected_{selected_shape}")
                pending_shape = None
            classification_type = None
        elif phase == "Static_Dot":
            focus_predictions = [] 
            classification_type = "Focus"
            is_moving_dot_active = True
            # Initialize random direction when entering this phase
            angle = random.uniform(0, 2 * math.pi)
            dx = dot_speed * math.cos(angle)
            dy = dot_speed * math.sin(angle)
            moving_dot_x, moving_dot_y = static_dot_center
        elif phase == "Size selected":
            focus_predictions = []
            classification_type = "Focus"
            focus_size_trigger = None  # Reset size trigger
            is_moving_dot_active = False
        else:
            classification_type = None
            is_moving_dot_active = False

        running = True
        while running:
            if time.time() - t0 >= dur:
                running = False

            # Collect EEG samples
            chunk, _ = eeg_inlet.pull_chunk(timeout=0.1, max_samples=512)

            if chunk:
                eeg_raw_data_for_saving.extend(chunk)
                
                if classification_type == "SSVEP":
                    eeg_buffer_SSVEP.extend(chunk)
                elif classification_type == "Focus":
                    eeg_buffer_Focus.extend(chunk)

            # SSVEP Processing
            if classification_type == "SSVEP" and len(eeg_buffer_SSVEP) >= CHUNK_SIZE:
                window = np.array(eeg_buffer_SSVEP[:CHUNK_SIZE])  
                
                if window.ndim == 2 and window.shape[0] == CHUNK_SIZE:
                    window = window.T
                
                all_scores, chunk_preds, final_pred, processed_data_for_plot = ssvep_cca_classification(
                    window,
                    duration_sec=0.50,
                    frequencies=[10, 12, 14, 16],
                    n_harmonics=2,
                    min_confidence=0.6,
                    return_scores=True
                )
                
                eeg_processed_ssvep_current_phase_buffer.extend(processed_data_for_plot.T)
                
                if chunk_preds:
                    current_predictions.extend(chunk_preds)
                    print(f"New prediction: {chunk_preds[-1]} Hz")
                
                eeg_buffer_SSVEP = eeg_buffer_SSVEP[CHUNK_SIZE:]

            # Focus Processing - handles both Static_Dot and Size selected phases
            elif classification_type == "Focus" and len(eeg_buffer_Focus) >= FOCUS_WINDOW:
                window = np.array(eeg_buffer_Focus[:FOCUS_WINDOW]).T
                
                try:
                    prediction, beta_alpha_ratio, confidence = Focus_classification(
                        data=window, 
                        model_path=r"C:\Twente course\BCI\ADV BCI\New_model.joblib",
                        fs=250,
                    )

                    if prediction is not None:
                        focus_predictions.append(prediction)
                        print(f"\nFocus Window {len(focus_predictions)}:")
                        print(f"  Beta/Alpha Ratio: {beta_alpha_ratio:.2f}")
                        print(f"  Prediction: {'Focused' if prediction == 1 else 'Not Focused'}")
                        
                        # For Size selected phase, handle size selection
                        if phase == "Size selected" and selected_shape:
                            if selected_shape == "Do Nothing":
                                focus_size_trigger = "N/A"
                                running = False
                            elif prediction == 1:  # Focused = Large
                                focus_size_trigger = "Large"
                                running = False
                            elif len(focus_predictions) >= 3 and prediction == 0:  # Not Focused for 3 windows = Small
                                focus_size_trigger = "Small" 
                                running = False
                        
                except Exception as e:
                    print(f"Focus classification error: {str(e)}")
                
                eeg_buffer_Focus = eeg_buffer_Focus[FOCUS_WINDOW:]

            # Handle events (only for Shape Selected now)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if ev.type == pygame.KEYDOWN:
                    if phase == "Shape Selected" and ev.key in key_to_shape_map:
                        selected_shape = key_to_shape_map[ev.key]
                        event_marker(f"Shape_Selected_{selected_shape}")
                    
            # Handle focus-based size selection
            if phase == "Size selected" and focus_size_trigger:
                last_placed_shape, last_placed_size = selected_shape, focus_size_trigger
                running = False
                
            # --- Movement logic for the small ball ---
            if is_moving_dot_active: # <-- Only moves if flag is true
                moving_dot_x += dx
                moving_dot_y += dy

                distance_from_center = math.sqrt(
                    (moving_dot_x - static_dot_center[0])**2 +
                    (moving_dot_y - static_dot_center[1])**2
                )

                max_allowed_distance = static_dot_radius - moving_dot_radius

                if distance_from_center > max_allowed_distance:
                    if distance_from_center != 0:
                        scale_factor = max_allowed_distance / distance_from_center
                        moving_dot_x = static_dot_center[0] + (moving_dot_x - static_dot_center[0]) * scale_factor
                        moving_dot_y = static_dot_center[1] + (moving_dot_y - static_dot_center[1]) * scale_factor

                    angle = random.uniform(0, 2 * math.pi) # <-- New random direction on bounce
                    dx = dot_speed * math.cos(angle)
                    dy = dot_speed * math.sin(angle)

            # Drawing code
            screen.fill(BLACK)
            draw_canvas()
            draw_flicker(phase)
            draw_static_dot()
            
            # Handle shape placement display
            if phase == "Shape Placement":
                if last_placed_shape == "Do Nothing":
                    current_caption = "Nothing Selected"
                    nothing_text = font.render("Nothing Selected", True, WHITE)
                    text_rect = nothing_text.get_rect(center=(canvas_x + canvas_w//2, canvas_y + canvas_h//2))
                    screen.blit(nothing_text, text_rect)
                elif last_placed_shape:
                    current_caption = f"Shape Placed: {last_placed_shape} ({last_placed_size})"
                    place_shape(last_placed_shape, last_placed_size == "Large")

            # Draw all placed shapes
            for i, sh in enumerate(placed_shapes):
                place_shape(sh, placed_large[i], y_off=i*140)

            # Draw focus meter
            draw_focus_meter(phase, current_focus_prediction)
            
            # Draw current caption
            screen.blit(font.render(current_caption, True, WHITE), (canvas_x, canvas_y - 40))
            pygame.display.update()
            clock.tick(60)

        # Phase completion processing
        if phase == "Selection_Phase" and current_predictions:
            # SSVEP Majority Voting
            print(f"\nAll SSVEP predictions: {current_predictions}")
            
            frequencies = [10, 12, 14, 16]  
            max_streaks_ssvep = {freq: 0 for freq in frequencies}
            current_streaks_ssvep = {freq: 0 for freq in frequencies}
            
            for freq in current_predictions:
                for f in frequencies:
                    if f != freq:
                        current_streaks_ssvep[f] = 0
                current_streaks_ssvep[freq] += 1
                if current_streaks_ssvep[freq] > max_streaks_ssvep[freq]:
                    max_streaks_ssvep[freq] = current_streaks_ssvep[freq]
            
            print("Maximum consecutive windows per frequency:")
            for freq in frequencies:
                print(f"  {freq} Hz: {max_streaks_ssvep[freq]}")
            
            max_freq = max(max_streaks_ssvep, key=max_streaks_ssvep.get)
            max_count = max_streaks_ssvep[max_freq]
            
            if sum(1 for v in max_streaks_ssvep.values() if v == max_count) > 1:
                print("Tie detected - using most recent longest streak")
                last_max_freq = None
                current_length = 0
                for freq in reversed(current_predictions):
                    if current_length == 0 or freq != last_max_freq:
                        current_length = 1
                        last_max_freq = freq
                    else:
                        current_length += 1
                    
                    if current_length == max_count:
                        max_freq = freq
                        break
            
            print(f"\nFinal SSVEP Prediction: {max_freq} Hz (max {max_count} consecutive windows)")
            pending_shape = hz_to_shape.get(max_freq, None)

        elif classification_type == "Focus" and focus_predictions:
            # Focus Majority Voting
            print(f"\nAll Focus predictions: {focus_predictions}")
            
            max_streaks_focus = {0: 0, 1: 0}
            current_streaks_focus = {0: 0, 1: 0}
            
            for pred in focus_predictions:
                current_streaks_focus[1-pred] = 0
                current_streaks_focus[pred] += 1
                if current_streaks_focus[pred] > max_streaks_focus[pred]:
                    max_streaks_focus[pred] = current_streaks_focus[pred]
            
            print("Maximum consecutive windows per class:")
            print(f"  Focused (1): {max_streaks_focus[1]}")
            print(f"  Not Focused (0): {max_streaks_focus[0]}")
            
            final_pred = max(max_streaks_focus, key=max_streaks_focus.get)
            max_count = max_streaks_focus[final_pred]
            
            if sum(1 for v in max_streaks_focus.values() if v == max_count) > 1:
                print("Tie detected - using most recent longest streak")
                last_pred = None
                current_length = 0
                for pred in reversed(focus_predictions):
                    if current_length == 0 or pred != last_pred:
                        current_length = 1
                        last_pred = pred
                    else:
                        current_length += 1
                    
                    if current_length == max_count:
                        final_pred = pred
                        break
            
            current_focus_prediction = final_pred
            print(f"\nFinal Focus Prediction: {'Focused (1)' if final_pred == 1 else 'Not Focused (0)'} (max {max_count} consecutive windows)")
            
            # Automatic size selection if not already triggered
            if phase == "Size selected" and selected_shape and not focus_size_trigger:
                if selected_shape == "Do Nothing":
                    focus_size_trigger = "N/A"
                    last_placed_shape, last_placed_size = "Do Nothing", "N/A"
                else:
                    if final_pred == 1:  # Focused
                        focus_size_trigger = "Large"
                    else:  # Not Focused
                        focus_size_trigger = "Small"
                    last_placed_shape, last_placed_size = selected_shape, focus_size_trigger
            
            focus_predictions = []  # Clear for next phase

        # Phase end marker
        if phase in phase_markers:
            event_marker(phase_markers[phase][1])

        # Save EEG segment
        if eeg_raw_data_for_saving:
            eeg_segments.setdefault(phase, []).append(np.array(eeg_raw_data_for_saving))
        
        # Save processed SSVEP EEG segment for plotting
        if phase == "Selection_Phase" and eeg_processed_ssvep_current_phase_buffer:
            eeg_processed_ssvep_plotting_segments.setdefault(phase, []).append(np.array(eeg_processed_ssvep_current_phase_buffer))
            print(f"--- Saved {phase} PROCESSED SSVEP segment with {len(eeg_processed_ssvep_current_phase_buffer)} samples ---")

        # Add placed shape if any
        if phase == "Size selected" and last_placed_shape and last_placed_shape != "Do Nothing":
            placed_shapes.append(last_placed_shape)
            placed_large.append(last_placed_size == "Large")

    # Session complete
    screen.fill(BLACK)
    complete_text = font.render("Session Complete. Thank you!", True, WHITE)
    screen.blit(complete_text, complete_text.get_rect(center=(screen_width//2, screen_height//2)))
    pygame.display.update()
    # EEG plotting

    try:
        import matplotlib.pyplot as plt


        # Plotting PROCESSED Selection Phase (using the new buffer)
        if 'Selection_Phase' in eeg_processed_ssvep_plotting_segments and \
           eeg_processed_ssvep_plotting_segments['Selection_Phase'] and \
           eeg_processed_ssvep_plotting_segments['Selection_Phase'][0].size > 0:
            
            sel_processed = eeg_processed_ssvep_plotting_segments['Selection_Phase'][0]
            print(f"Plotting 'Selection_Phase' PROCESSED - sel_processed.shape: {sel_processed.shape}")
            print(f"Selection_Phase PROCESSED min/max values: {np.min(sel_processed):.4f} / {np.max(sel_processed):.4f}")

            plt.figure(figsize=(12,6))
            plt.suptitle("EEG - Selection Phase (PROCESSED)")
            # Note: sel_processed should already be (samples, channels) due to the .T when extending
            for ch in range(sel_processed.shape[1]):
                plt.plot(sel_processed[:, ch], label=f'Ch{ch+1}')
            plt.legend(bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
            # For processed data, if it's scaled to microvolts and filtered,
            # you might not need a fixed ylim, or a narrower one like this:
            # plt.ylim([-100, 100])
        
        if 'Static_Dot' in eeg_segments and eeg_segments['Static_Dot']:
            st = eeg_segments['Static_Dot'][0]
            #print(f"Plotting 'Static_Dot' - st.shape: {st.shape}") # ADD THIS
            #print(f"Plotting 'Static_Dot' - st.dtype: {st.dtype}") # AND THIS
            #print(f"Selection_Phase min/max values: {np.min(sel):.4f} / {np.max(sel):.4f}") # THIS LINE MIGHT BE A TYPO IN YOUR ORIGINAL CODE: 'sel' is not defined here. It should likely be 'st'. I will leave it as is per your request.
            #print(f"Static_Dot min/max values: {np.min(st):.4f} / {np.max(st):.4f}")


            plt.figure(figsize=(12,6))
            plt.suptitle("EEG - Static Dot Phase")
            for ch in range(st.shape[1]):
                plt.plot(st[:, ch], label=f'Ch{ch+1}')
            plt.legend(bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available - skipping EEG visualization")
    except Exception as e:
        print(f"Error during EEG plotting: {e}")
    pygame.time.delay(3000)
    pygame.quit()

    # Summary printout
    print("\n--- EEG Data Collection Summary ---")
    for ph, segs in eeg_segments.items():
        total_samples = sum(s.shape[0] for s in segs)
        print(f"{ph}: {len(segs)} segment(s), {total_samples} samples")
if __name__ == "__main__":
    main()