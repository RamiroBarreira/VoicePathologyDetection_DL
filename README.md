# VoicePathologyDetection_DL
 Deep Learning for detecting pathological voices.<br>
 <i>Train and test a Convolutional Neural Network (CNN) for a voice pathology detection system.</i>


## Data
`data/train_data.npz` contains the training data, which includes one numpy array 'X' with (291,227,227,3) shape and two arrays, 'Y' and 'S', of length 291.<br>
`data/test_data.npz` contains the test data, which includes one numpy array 'X_tes' with (32,227,227,3) shape, and two arrays, 'Y_tes' and 'S_tes', of length 32.<br>
'Y' and 'Y_tes' contain the class ('0' for normal and '1' for pathological) associated to each spectrogram set in 'X' and 'X_tes' respectively.<br>
'S' and 'S_tes' associate spectrograms with each corresponding database file.

P.S.: <i>See the <b>Database</b> section for more information on data.<br>
Each file has one or more spectrograms associated with itself.<br>
On the training data, there are 291 spectrograms that are associated to 198 files from the MEEI database 173/53 subset.<br>
On the test data, there are 32 spectrograms that are associated to 22 files from the MEEI database 173/53 subset.<br>
Some files from the MEEI database were not considered because the record was shorter than 928.8ms.</i>


## Usage
Train and validate the model
```bash
python VPD_DL.py
```


## Database
The database was derived from a record subset of the Massachusetts Eye and Ear Infirmary (MEEI) voice pathology DB. It comprises the band-pass filtered spectrograms extracted from the MEEI's database 173/53 subset [1] records. For the power spectra extraction, all the records were downsampled to 22050Hz (instead of the original 25000Hz and 50000Hz sampling rates). The power spectra were calculated from Blackman windowed signal frames. Windows were 46.44ms (1024 samples) long and were applied at 23.22ms (512 samples) steps. Thereafter, 40 triangular mel-scale centered overlapped band-pass filters were applied to each spectrum. The delta and delta-delta-filtered spectra were also calculated. Blocks of 40 consecutive spectra (representing ~928.8ms of audio), delta and delta-delta-spectra, were adopted at first. Each block constitutes a spectrogram set (which includes spectrogram, delta and delta-delta spectrograms). The spectrograms were transformed from 40x40 pixels images to 227x227 pixels images using bicubic interpolation and fed the system's convolutional neural network (CNN). Records longer than 928.8ms (40 consecutive spectra) and smaller than 2 * 928.8ms generated one spectrogram set; records longer than 2 * 928.8ms and smaller than 3 * 928.8ms generated two spectrogram sets associated to the same record. Those greater than 3 * 928.8ms and smaller than 4 * 928.8ms generated three spectrogram sets associated to the same record, and so on. As a final step, the spectrograms were normalized as a whole to get values between 0 and 1 for feeding the CNN.


## References
[1] V. Parsa e G. Jamieson, “Identification of pathological voices using glottal noise measures,” J. Speech, Language, Hearing Res., vol. 43, nº 2, pp. 469-485, 2000. 