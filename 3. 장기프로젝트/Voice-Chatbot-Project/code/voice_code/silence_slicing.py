import json
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def create_json(audio_file):
  intervals_jsons = []

  min_silence_length = 70
  intervals = detect_nonsilent(audio_file,
                               min_silence_len=min_silence_length,
                               silence_thresh=-32.64)
  
  if intervals[0][0] != 0:
    intervals_jsons.append({'start':0,'end':intervals[0][0]/1000,'tag':'silence'}) 
    
  non_silence_start = intervals[0][0]
  before_silence_start = intervals[0][1]

  for interval in intervals:
    interval_audio = audio_file[interval[0]:interval[1]]

    if (interval[0] - before_silence_start) >= 2000:
      intervals_jsons.append({'start':non_silence_start/1000,'end':(before_silence_start+200)/1000,'tag':'talk'}) 
      non_silence_start = interval[0]-200
      intervals_jsons.append({'start':before_silence_start/1000,'end':interval[0]/1000,'tag':'silence'}) 
    before_silence_start = interval[1]

  if non_silence_start != len(audio_file):
    intervals_jsons.append({'start':non_silence_start/1000,'end':len(audio_file)/1000,'tag':'talk'})

  return intervals_jsons

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

if __name__ == "__main__":
    sound = AudioSegment.from_file("data/wav_1.wav", "wav")
    normalized_sound = match_target_amplitude(sound, -20.0)
    alignment_1 = create_json(normalized_sound)
    file_path = 'data/alignment_1.json'
    with open(file_path, 'w') as f:
       json.dump(alignment_1, f, indent="\t")