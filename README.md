# MusicGPT

Generate music samples using a local LLM and natural language prompts

# Install

On Mac and Linux:
```shell
brew install gabotechs/taps/musicgpt
```

Everywhere, including Windows:
```shell
cargo install musicgpt
```

# Usage

Generate and play a music sample from a natural language prompt

```shell
musicgpt "A retro synthwave song with high bpms"
```

By default, it produces a sample of 10s, which can be configured

```shell
musicgpt "Create a pop song with 80's vive" --secs 30
```

There's multiple models available, it will use the smallest one by default, but
users can opt into a bigger model

```shell
musicgpt "a relaxing LoFi song with birds chirping in the background" --model medium
```

More options are available running `musicgpt --help`
