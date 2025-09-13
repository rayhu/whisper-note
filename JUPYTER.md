# Run Jupyter

1. Create and activate a conda environment:
```bash
conda create -n whisper-note python=3.11
conda activate whisper-note
```


2. Install JupyterLab and kernel support:
```bash
conda install -c conda-forge jupyterlab ipykernel -y
```

3. Install faster-whisper and its dependencies
pip install ctranslate2 faster-whisper

4. Install required packages:
```bash
pip install -r requirements.txt
```

5. Register the current environment as a Jupyter Kernel

```bash
python -m ipykernel install --user --name whisper-note --display-name "whisper-note"
```
jupyter lab # has some compatibility issue with webrtc component

use juputer notebook
