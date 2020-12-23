# Pyannote Colab Code
### Link: https://github.com/pyannote/pyannote-audio


### 0. 1 Install pyannote

``` 
!pip install -q pyannote.audio==1.1
```

### 0. 2 Load .wav file
```
from pyannote.core import Segment, notebook

import google.colab
own_file, _ = google.colab.files.upload().popitem()
OWN_FILE = {'audio': own_file}
notebook.reset()

# load audio waveform and play it
waveform = RawAudio(sample_rate=16000)(OWN_FILE).data
Audio(data=waveform.squeeze(), rate=16000, autoplay=True)
```

### 1. Diarization
```
import torch
pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia')
diarization = pipeline(OWN_FILE)
diarization
```

### 2. Overlap Detection
```
overlap_detection = torch.hub.load('pyannote/pyannote-audio', 'ovl_ami', pipeline=True)
overlap_detection(OWN_FILE).get_timeline()
```



### 3. Results

- **임수정 뉴스룸 인터뷰 데이터 - 화자분리: Good**

![ex_screenshot](./interview_diarization.png)
![ex_screenshot](./interview_overlap.png)


Overlap Detection은 일반적으로 parameter를 'ovl_ami'보다는 'ovl'으로 설정한 결과가 더 좋은 듯.
