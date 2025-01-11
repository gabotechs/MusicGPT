<h1 align="center">
    <span> MusicGPT</span>
    <img height="30" src="assets/music-icon.svg" alt="Signway logo"/>
</h1>

<p align="center">
    Generate music based on natural language prompts using LLMs running locally.
</p>


https://github.com/gabotechs/MusicGPT/assets/45515538/f0276e7c-70e5-42fc-817a-4d9ee9095b4c
<p align="end">
☝️ Turn up the volume!
</p>


# Overview

MusicGPT is an application that allows running the latest music generation
AI models locally in a performant way, in any platform and without installing heavy dependencies
like Python or machine learning frameworks.

Right now it only supports [MusicGen by Meta](https://audiocraft.metademolab.com/musicgen.html),
but the plan is to support different music generation models transparently to the user.

The main milestones for the project are:
- [x] Text conditioned music generation
- [ ] Melody conditioned music generation
- [ ] Indeterminately long / infinite music streams

# Install

### Mac and Linux

MusicGPT can be installed on Mac and Linux using `brew`:

```shell
brew install gabotechs/taps/musicgpt
```

### Windows

Download and install MusicGPT's executable file [following this link](https://github.com/gabotechs/MusicGPT/releases/latest/download/musicgpt-x86_64-pc-windows-msvc.exe).

### All platforms

Precompiled binaries are available for the following platforms:
- [macOS Apple Silicon](https://github.com/gabotechs/MusicGPT/releases/latest/download/musicgpt-aarch64-apple-darwin)
- [Linux x86_64](https://github.com/gabotechs/MusicGPT/releases/latest/download/musicgpt-x86_64-unknown-linux-gnu)
- [Windows](https://github.com/gabotechs/MusicGPT/releases/latest/download/musicgpt-x86_64-pc-windows-msvc.exe)

Just downloading them and executing them should be enough.

### Docker (Recommend for running with CUDA)

If you want to run MusicGPT with a CUDA enabled GPU, this is the best way, as you only need to have [the basic
NVIDIA drivers](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed in your system.

```shell
docker pull gabotechs/musicgpt
```

Once the image is downloaded, you can run it with:

```shell
docker run -it --gpus all -p 8642:8642 -v ~/.musicgpt:/root/.local/share/musicgpt gabotechs/musicgpt --gpu --ui-expose
```

### With cargo

If you have the [Rust toolchain](https://www.rust-lang.org/tools/install) installed in your system, you can install it
with `cargo`.

```shell
cargo install musicgpt
```

# Usage

There are two ways of interacting with MusicGPT: the UI mode and the CLI mode.

## UI mode

This mode will display a chat-like web application for exchanging prompts with the LLM. It will:
- store your chat history 
- allow you to play the generated music samples whenever you want
- generate music samples in the background
- allow you to use the UI in a device different from the one executing the LLMs

You can run the UI by just executing the following command:

```shell
musicgpt
```

You can also choose different models for running inference, and whether to use a GPU or not, for example:

```shell
musicgpt --gpu --model medium
```

> [!WARNING]  
> Most models require really powerful hardware for running inference

If you want to use a CUDA enabled GPU, it's recommended that you run MusicGPT with Docker:

```shell
docker run -it --gpus all -p 8642:8642 -v ~/.musicgpt:/root/.local/share/musicgpt gabotechs/musicgpt --ui-expose --gpu
```

## CLI mode

This mode will generate and play music directly in the terminal, allowing you to provide multiple
prompts and playing audio as soon as it's generated. You can generate audio based on a prompt with
the following command:

```shell
musicgpt "Create a relaxing LoFi song"
```

By default, it produces a sample of 10s, which can be configured up to 30s:

```shell
musicgpt "Create a relaxing LoFi song" --secs 30
```

There's multiple models available, it will use the smallest one by default, but
you can opt into a bigger model:

```shell
musicgpt "Create a relaxing LoFi song" --model medium
```

> [!WARNING]  
> Most models require really powerful hardware for running inference

If you want to use a CUDA enabled GPU, it's recommended that you run MusicGPT with Docker:

```shell
docker run -it --gpus all -v ~/.musicgpt:/root/.local/share/musicgpt gabotechs/musicgpt --gpu "Create a relaxing LoFi song"
```

You can review all the options available running:

```shell
musicgpt --help
```

# Benchmarks

The following graph shows the inference time taken for generating 10 seconds of audio using
different models on a Mac M1 Pro. For comparison, it's Python equivalent using https://github.com/huggingface/transformers
is shown. 

The command used for generating the 10 seconds of audio was:

 
```shell
musicgpt '80s pop track with bassy drums and synth'
```

<details>
<summary>This is the Python script used for generating the 10 seconds of audio</summary>

```python
import scipy
import time
from transformers import AutoProcessor, MusicgenForConditionalGeneration

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

inputs = processor(
    text=["80s pop track with bassy drums and synth"],
    padding=True,
    return_tensors="pt",
)

start = time.time()
audio_values = model.generate(**inputs, max_new_tokens=500)
print(time.time() - start) # Log time taken in generation

sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())
```

</details>

<p align="center">
<img height=400 src="https://github.com/gabotechs/MusicGPT/assets/45515538/edae3c25-04e3-41c3-a2b5-c0829fa69ee3"/>
</p>

# Storage

MusicGPT needs access to your storage in order to save downloaded models and generated audios along with some
metadata needed for the application to work properly. Assuming your username is `foo`, it will store the data
in the following locations:

- Windows: `C:\Users\foo\AppData\Roaming\gabotechs\musicgpt`
- MacOS: `/Users/foo/Library/Application\ Support/com.gabotechs.musicgpt`
- Linux: `/home/foo/.config/musicgpt`

# License

The code is licensed under a [MIT License](./LICENSE), but the AI model weights that get downloaded
at application startup are licensed under the [CC-BY-NC-4.0 License](https://spdx.org/licenses/CC-BY-NC-4.0)
as they are generated based on the following repositories:

- https://huggingface.co/facebook/musicgen-small
- https://huggingface.co/facebook/musicgen-medium
- https://huggingface.co/facebook/musicgen-large
- https://huggingface.co/facebook/musicgen-melody

