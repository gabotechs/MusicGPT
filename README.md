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

Or by directly downloading the precompiled binaries
from [this link](https://github.com/gabotechs/MusicGPT/releases/latest)

### Windows

On Windows, the executable file can be downloaded
from [this link](https://github.com/gabotechs/MusicGPT/releases/latest/download/x86_64-pc-windows-msvc.tar.gz).

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

# License

The code is licensed under a [MIT License](./LICENSE), but the AI model weights that get downloaded
at application startup are licensed under the [CC-BY-NC-4.0 License](https://spdx.org/licenses/CC-BY-NC-4.0)
as they are generated based on the following repositories:

- https://huggingface.co/facebook/musicgen-small
- https://huggingface.co/facebook/musicgen-medium
- https://huggingface.co/facebook/musicgen-large
- https://huggingface.co/facebook/musicgen-melody

