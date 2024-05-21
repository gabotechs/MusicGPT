<h1 align="center">
    <img height="50" src="assets/music-icon.svg" alt="Signway logo"/>
    <span> MusicGPT</span>
</h1>

Generate music based on natural language prompts using LLMs running locally.

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

If you want to run MusicGPT with a CUDA enabled GPU, this is the best way, as you only need to have the basic
NVIDIA drivers installed in your system, everything else is bundled in the Docker image.

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

This mode will display a chat-like web application for exchanging prompts with the LLM. It will
store your chat history and allow you to play the generated music samples whenever you want.
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
docker run -it --gpus all -p 8642:8642 -v ~/.musicgpt:/root/.local/share/musicgpt gabotechs/musicgpt --gpu
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
docker run -it --gpus all -v ~/.musicgpt:/root/.local/share/musicgpt gabotechs/musicgpt --gpu --ui-expose "Create a relaxing LoFi song"
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

