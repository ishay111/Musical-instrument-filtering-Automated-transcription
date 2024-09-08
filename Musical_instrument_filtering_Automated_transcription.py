import pyaudio
import wave
import threading
import keyboard
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from mido import MidiFile, MidiTrack, Message
import subprocess

class AudioRecorder:
    def __init__(self, input_device_index=None):
        self.chunk = 2048  
        self.sample_format = pyaudio.paInt16
        self.channels = 1  
        self.fs = 44100  
        self.filename = "input.wav"
        self.frames = []
        self.recording = False
        self.stream = None
        self.p = pyaudio.PyAudio()
        self.input_device_index = input_device_index

        
        self.stream = self.p.open(format=self.sample_format,
                                  channels=self.channels,
                                  rate=self.fs,
                                  frames_per_buffer=self.chunk,
                                  input=True,
                                  input_device_index=self.input_device_index)

    def start_recording(self):
        self.frames = []
        self.recording = True
        print("Recording started. Press 'R' again to stop.")

    def stop_recording(self):
        self.recording = False
        print("Recording stopped.")
        self.save_recording()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def save_recording(self):
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print(f"File saved as {self.filename}")

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def run(self):
        print("Press 'R' to start/stop recording.")
        while True:
            if keyboard.is_pressed('r'):
                self.toggle_recording()
                while keyboard.is_pressed('r'):
                    pass  
                if not self.recording:
                    break  
            if self.recording:
                data = self.stream.read(self.chunk)
                self.frames.append(data)

if __name__ == "__main__":
    p = pyaudio.PyAudio()
    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            print(f"Index {i}: {device_info['name']}")
    p.terminate()

    
    input_device_index = int(input("Enter the index of the desired audio input device: "))

    recorder = AudioRecorder(input_device_index=input_device_index)
    recorder.run()

    # Processing the recorded audio
    audio_path = "input.wav"
    
    y, sr = librosa.load(audio_path, sr=None)

    # Define the window length for STFT
    window_length_seconds = 0.1
    n_fft = int(window_length_seconds * sr)
    hop_length = n_fft // 4

    # Compute STFT
    stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    stft_db = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)

    
    note_starts = []
    note_durations = []
    note_pitches = []
    note_tags = []
    active_notes = {}
    combined_stft = np.zeros_like(stft_result)

    # Define frequency ranges for low, middle, and high notes
    frequency_ranges = [
        (10, 20, 0.2, 27.5, 261.63, 'Low'),
        (50, 20, 0.6, 146.83, 1479.98, 'Middle'),
        (20, 10, 0.4, 987.77, 4186.01, 'High'),
    ]

    def is_harmonic(f1, f2, tolerance=0.02):
        ratio = f2 / f1
        return np.abs(ratio - round(ratio)) < tolerance

    for percentile, db_mask, threshold_percentage, min_freq, max_freq, tag in frequency_ranges:
        noise_db = np.percentile(stft_db, axis=1, q=percentile)
        mask = stft_db > (noise_db[:, None] + db_mask)
        denoised_stft = stft_result * mask
        combined_stft += np.abs(denoised_stft)
        stft_magnitude = np.abs(denoised_stft)
        denoised_stft_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)

        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        times = librosa.frames_to_time(np.arange(denoised_stft.shape[1]), sr=sr, hop_length=hop_length)

        for frame_idx in range(denoised_stft.shape[1]):
            frame_magnitude = stft_magnitude[:, frame_idx]
            highest_peak_amplitude = 0
            if np.max(frame_magnitude) > 0:
                highest_peak_amplitude = np.max(frame_magnitude)
            adaptive_threshold = threshold_percentage * highest_peak_amplitude

            peaks, _ = find_peaks(frame_magnitude, height=adaptive_threshold)
            current_frequencies = frequencies[peaks]
            current_frequencies = current_frequencies[(current_frequencies >= min_freq) & (current_frequencies <= max_freq)]
            current_midi_notes = np.round(librosa.hz_to_midi(current_frequencies)).astype(int)

            harmonics = set()
            for i, freq1 in enumerate(librosa.midi_to_hz(current_midi_notes)):
                for j, freq2 in enumerate(librosa.midi_to_hz(current_midi_notes)):
                    if i != j and is_harmonic(freq1, freq2): 
                        if freq1 < freq2:
                            harmonics.add(j)

            current_midi_notes = [note for idx, note in enumerate(current_midi_notes) if idx not in harmonics]

            for note in list(active_notes.keys()):
                if note not in current_midi_notes:
                    start_time = active_notes.pop(note)
                    note_starts.append(start_time)
                    note_durations.append(times[frame_idx] - start_time)
                    note_pitches.append(note)
                    note_tags.append(tag)

            for note in current_midi_notes:
                if note not in active_notes:
                    active_notes[note] = times[frame_idx]

    for note, start_time in active_notes.items():
        note_starts.append(start_time)
        note_durations.append(times[-1] - start_time)
        note_pitches.append(note)
        note_tags.append(tag)

    note_starts = np.array(note_starts)
    note_durations = np.array(note_durations)
    note_pitches = np.array(note_pitches)
    note_tags = np.array(note_tags)

    
    min_note_duration = 0.1
    valid_indices = note_durations >= min_note_duration
    note_starts = note_starts[valid_indices]
    note_durations = note_durations[valid_indices]
    note_pitches = note_pitches[valid_indices]
    note_tags = note_tags[valid_indices]

    
    sorted_indices = np.argsort(note_starts)
    note_starts = note_starts[sorted_indices]
    note_durations = note_durations[sorted_indices]
    note_pitches = note_pitches[sorted_indices]
    note_tags = note_tags[sorted_indices]

    
    unique_notes = []
    unique_starts = []
    unique_durations = []
    unique_pitches = []
    unique_tags = []

    threshold_time_same_pitch = 0.01
    threshold_time_harmony = 0.1

    for start, duration, pitch, tag in zip(note_starts, note_durations, note_pitches, note_tags):
        note_tuple = (start, pitch)
        if unique_notes:
            last_start, last_duration, last_pitch, last_tag = unique_starts[-1], unique_durations[-1], unique_pitches[-1], unique_tags[-1]
            last_end = last_start + last_duration

            if pitch == last_pitch and (start - last_end) <= threshold_time_same_pitch:
                unique_durations[-1] = start + duration - last_start
                continue
            if ((pitch == last_pitch + 12 ) or (pitch == last_pitch + 19)) and (start - last_end) <= threshold_time_harmony:
                continue
            if ((pitch == last_pitch - 12 ) or (pitch == last_pitch - 19)) and (start - last_end) <= threshold_time_harmony:
                unique_notes.pop()
                unique_starts.pop()
                unique_durations.pop()
                unique_pitches.pop()
                unique_tags.pop()
        
        if note_tuple not in unique_notes:
            unique_notes.append(note_tuple)
            unique_starts.append(start)
            unique_durations.append(duration)
            unique_pitches.append(pitch)
            unique_tags.append(tag)

    for start, duration, pitch, tag in zip(unique_starts, unique_durations, unique_pitches, unique_tags):
        print(f"Note: {int(pitch)}, Start Time: {start:.2f}s, Duration: {duration:.2f}s, Tag: {tag}")

    # Plot the STFT magnitude and detected notes
    plt.figure(figsize=(14, 5))
    combined_stft_db = librosa.amplitude_to_db(np.abs(combined_stft), ref=np.max)
    librosa.display.specshow(combined_stft_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.scatter(unique_starts, [librosa.midi_to_hz(pitch) for pitch in unique_pitches], color='red', s=10)
    plt.title('STFT Magnitude and Detected Notes')
    plt.show()

    # Create a MIDI file with the detected notes
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    pre_track = []

    ticks_per_beat = midi.ticks_per_beat

    def seconds_to_ticks(time_in_seconds, ticks_per_beat, tempo_bpm=120):
        seconds_per_beat = 60 / tempo_bpm
        ticks_per_second = ticks_per_beat / seconds_per_beat
        return int(time_in_seconds * ticks_per_second)

    previous_time_in_ticks = 0
    for start, duration, pitch, tag in zip(unique_starts, unique_durations, unique_pitches, unique_tags):
        midi_note = int(pitch)
        start_time_in_ticks = seconds_to_ticks(start, ticks_per_beat)
        duration_in_ticks = seconds_to_ticks(duration, ticks_per_beat)

        delta_time = start_time_in_ticks - previous_time_in_ticks
        if delta_time < 0:
            delta_time = 0
        pre_track.append(Message('note_on', note=midi_note, velocity=64, time=start_time_in_ticks))
        pre_track.append(Message('note_off', note=midi_note, velocity=64, time=start_time_in_ticks + duration_in_ticks))

    pre_track.sort(key=lambda msg: msg.time)
    prev_time = 0
    for message in pre_track:
        track.append(Message(message.type, note=message.note, velocity=message.velocity, time=message.time - prev_time))
        prev_time = message.time

    midi.save('output.mid')

    program_path = "MidiSheetMusic_2.6.2.exe"
    command = [program_path, "output.mid"]

    subprocess.Popen(command)
