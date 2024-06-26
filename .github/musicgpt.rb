class Musicgpt < Formula
  desc ""
  homepage "https://github.com/gabotechs/MusicGPT"
  version "<version>"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/gabotechs/MusicGPT/releases/download/v<version>/aarch64-apple-darwin.tar.gz"

      def install
        bin.install "musicgpt"
      end
    end
    if Hardware::CPU.intel?
      url "https://github.com/gabotechs/MusicGPT/releases/download/v<version>/x86_64-apple-darwin.tar.gz"

      def install
        bin.install "musicgpt"
      end
    end
  end

  on_linux do
    if Hardware::CPU.intel?
      url "https://github.com/gabotechs/MusicGPT/releases/download/v<version>/x86_64-unknown-linux-gnu.tar.gz"

      def install
        bin.install "musicgpt"
        lib.install Dir["lib/*"]
      end
    end
# TODO: Linux ARM still does not work.
#     if Hardware::CPU.arm? && Hardware::CPU.is_64_bit?
#       url "https://github.com/gabotechs/MusicGPT/releases/download/v<version>/musicgpt-aarch64-unknown-linux-gnu.tar.gz"
#
#       def install
#         bin.install "musicgpt"
#       end
#     end
  end
end
