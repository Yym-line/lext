# LExt

This repository provides a PyTorch reproduction of the method proposed in "[Listen to Extract:Onset-Prompted Target Speaker Extraction]([https://arxiv.org/pdf/2505.05114])" (TASLP).

## Dataset
[Libri2Mix](https://github.com/JorisCos/LibriMix) min wav16k dataset. The `Data` folder contains three subfolders: `train`, `dev`, and `test`. Each subfolder includes three files:
- `mix_both.scp`: Clean mixtures of 2 speakers+noise.
- `ref.scp`: Target speaker’s speech.
- `auxs1.scp`: Enrollment speech from the target speaker, which is different from the target speaker’s speech in the mixture.

The `mix_both.scp` corresponds to the **2-speaker+noise** scenario in the results section.
Note that in this dataset, only the first speaker in the mixed speech is considered the target speaker.
Make sure to update the file paths in the `scp` files to match your local data locations. Also, remember to update the data paths in `conf_unet_tse_32ms.py` accordingly.

## Training
- **`train.sh`**: Shell script that initiates training by setting parameters (e.g., epochs, batch size, GPU settings) and calling the Python script (`train_unet_tse_steplr_clip.py`). To train the model, run:
  ```bash
  ./train.sh
```
  
- **`train_unet_tse_steplr_clip.py`**: Main Python script for training. It initializes the model, sets up data loaders, and manages the training loop.
  
- **`conf_unet_tse_32ms.py`**: Configuration file containing model architecture, data paths, and training hyperparameters.

- **`lext_tfgridnet.py`**: Defines the `LExt` model, which is used in the training script.

## Evaluation

To evaluate the model, use the provided `eval.sh` script. It sets the necessary parameters (e.g., model checkpoint, GPU ID, data paths) and calls `evaluate.py` for performance evaluation.

- **`eval.sh`**: Runs the evaluation by setting paths and calling `evaluate.py`.  
  - Usage:
    ```bash
    ./eval.sh
    ```

- **`evaluate.py`**: Evaluates the model on the test set, computing metrics like SDR, SI-SNR, PESQ, and STOI.

## Results

Results on Libri2Mix TSE tasks:

<table>
  <thead>
    <tr>
      <th rowspan="4">Condition</th>
      <th rowspan="4">Method</th>
      <th colspan="4">Metrics</th>
    </tr>
    <tr>
      <th>SI-SDR</th>
      <th>PESQ</th>
      <th>STOI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4" align="center"><strong>2-speaker+noise</strong></td>
      <td>Mixture</td>
      <td>-2.03</td>
      <td>1.43</td>
      <td>64.65</td>
    </tr>
    <tr>
      <td>SEF-PNet</td>
      <td>8.18</td>
      <td>2.44</td>
      <td>82.67</td>
    </tr>
    <tr>
      <td>CIE-mDPTNet</td>
      <td>9.47</td>
      <td>2.67</td>
      <td>85.35</td>
    </tr>
    <tr>
      <td>LExt</td>
      <td>10.47</td>
      <td>2.74</td>
      <td>87.26</td>
    </tr>
  </tbody>
</table>

### GPU Setup
This code is designed to run on a single GPU. By default, in the `train.sh` script, the `gpuid` is set to `0`. 

To use multiple GPUs, modify `gpuid=0,1,2,...` in `train.sh`. 

Additionally, for multi-GPU setups, comment out the line:
```python
from memonger import SublinearSequential
```
and replace SublinearSequential with nn.Sequential in lext_tfgridnet.py to avoid memory issues.

### Create SCP
The SCP file I provided is from [DPCCN](https://github.com/jyhan03/icassp22-dataset/tree/main/lst/libri2mix). It only uses the first speaker as the target. 

Any problems, contact me at y2379286479@outlook.com, and a reply will be given promptly.
