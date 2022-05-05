# VoicePathologyDetection_DL
 Deep Learning for detecting pathological voices.<br>
 Train and test a Convolutional Neural Network (CNN) for a voice pathology detection system.


## Data
`data/train_data.npz` contains the train data, which includes one numpy array 'X' with (583,227,227,3) shape and two arrays, 'Y' and 'S', of length 583.<br>
`data/test_data.npz` contains the test data, which includes one numpy array 'X_tes' with (61,227,227,3) shape, and two arrays, 'Y_tes' and 'S_tes' of length 61.<br>
'Y' and 'Y_tes' contain the class ('0' for normal and '1' for pathological) associated to each spectrogram in 'X' and 'X_tes' respectively.<br>
'S' and 'S_tes' associate spectrograms with each corresponding database file.

PS: See the <b>Database</b> section for more information on data.<br>
Each file has one or more spectrograms associated with itself.<br>
On the train data, there are 583 spectrograms that are associated to 203 files from the MEEI database 173/53 subset.<br>
On the test data, there are 61 spectrograms that are associated to 21 files from the MEEI database 173/53 subset.<br>
Some files (~2) from the MEEI database were not considered because the record was shorter than 464.4ms.


## Usage
Train and validate the model
```bash
python VPD_DL.py
```


## Database
The database was derived from a record subset of the Massachusetts Eye and Ear Infirmary (MEEI) voice pathology DB. It comprises the filtered power spectra extracted from the MEEI's database 173/53 subset [1] records. For the power spectra extraction, all the records were downsampled to 22050Hz (instead of the original 25000Hz and 50000Hz sampling rates). The power spectra were calculated from Hamming windowed signal frames. Windows were 46.44ms (1024 samples) long and were applied at 23.22ms (512 samples) steps. They were normalized so that the sum of the samples resulted in 1. Thereafter, 20 rectangular mel-scale centered band-pass filters were applied to each spectrum. The delta and delta-delta-filtered spectra were also calculated. Blocks of 20 consecutive spectra (representing ~464.4ms of audio), delta and delta-delta-spectra, were adopted at first. Such block concatenation leads to a spectrogram. The spectrograms were transformed from 20x20 pixels images to 227x227 pixels images using bicubic interpolation and fed the system's convolutional neural network (CNN). Records longer than 464.4ms (20 consecutive spectra) and smaller than 2 * 464.4ms generated one spectrogram; records longer than 2 * 464.4ms and smaller than 3 * 464.4ms generated two spectrograms associated to the same record. Those greater than 3 * 464.4ms and smaller than 4 * 464.4ms generated three spectrograms associated to the same record, and so on. As a final step, the spectrograms were normalized as a whole to get values between 0 and 1, approximately, for feeding the CNN.


## References
[1] V. Parsa e G. Jamieson, “Identification of pathological voices using glottal noise measures,” J. Speech, Language, Hearing Res., vol. 43, nº 2, pp. 469-485, 2000. 