# SDXL-Lightning Image Generator

Generate images using SDXL-Lightning on CUDA or Apple Silicon.

## Usage

Create a .env file with your prompt and a negative prompt:

```sh
PROMPT="A man sitting on a bench with a book."
```

Init python dependencies:

```sh
pip install -r requirements.txt
```

Run the python script:

```sh
python diffuse.py
```

Your output will be in the folder `output/<date>.png`.
