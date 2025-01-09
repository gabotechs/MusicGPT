class Musicgpt < Formula
  desc ""
  homepage "https://github.com/gabotechs/MusicGPT"
  version "<version>"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/gabotechs/MusicGPT/releases/download/v<version>/musicgpt-aarch64-apple-darwin"

      def install
        bin.install "musicgpt-aarch64-apple-darwin" => "musicgpt"
      end
    end
    if Hardware::CPU.intel?
      url "https://github.com/gabotechs/MusicGPT/releases/download/v<version>/musicgpt-x86_64-apple-darwin"

      def install
        bin.install "musicgpt-x86_64-apple-darwin" => "musicgpt"
      end
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/gabotechs/MusicGPT/releases/download/v<version>/musicgpt-x86_64-unknown-linux-gnu"

      def install
        bin.install "musicgpt-x86_64-unknown-linux-gnu" => "musicgpt"
      end
    end
    # TODO: Linux ARM still does not work.
    # if Hardware::CPU.arm? && Hardware::CPU.is_64_bit?
    #   url "https://github.com/gabotechs/MusicGPT/releases/download/v<version>/musicgpt-aarch64-unknown-linux-gnu"
    #
    #   def install
    #     bin.install "musicgpt-aarch64-unknown-linux-gnu" => "musicgpt"
    #   end
    # end
  end
end
