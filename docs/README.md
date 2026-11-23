# PiKaraoke

<img width="588" alt="pikaraoke-banner" src="https://raw.githubusercontent.com/configure-ng/pikaraoke/master/docs/images/banner.png" />

PiKaraoke is a cross-platform karaoke server that brings the professional "KTV" experience to your home. It transforms your computer or Raspberry Pi into a dedicated karaoke station with a full-screen player and an instant web interface. Guests can join by simply scanning a QR code—no app downloads required—to browse your local library, manage the queue, and access countless karaoke hits from YouTube.

- 📱 Instant Mobile Remote: Search and queue songs from any smartphone—just scan and sing.
- 📺 Dedicated Player: High-performance splash screen that can be opened on any web browser for a true karaoke room feel.
- 🌐 YouTube & Local Media: Play your own files or access more from the web.
- 🎹 Live Pitch Shifting: Adjust the key of any song to match your vocal range.
- 🛠️ Admin Control: Manage the queue and settings via a password-protected admin mode.
- 📊 Session Management: Every night is recorded automatically—browse the play history, then see which songs and performers topped the charts.
- 🎯 Hyper-accurate vocal performance scoring system: (not really, it's random. But kind of fun!)
- 🐧 Lightweight & Versatile: Runs anywhere from a basic Raspberry Pi to a high-end PC.

Love PiKaraoke? This project is independently maintained and free for everyone to enjoy. If PiKaraoke has made your parties better and you'd like to help keep the project alive and growing, feel free to [buy me a coffee](https://www.buymeacoffee.com/vicwomg)! <br/><br/>
<a href="https://www.buymeacoffee.com/vicwomg" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-green.svg)](https://conventionalcommits.org)

## Table of Contents

- [Supported Devices / OS / Platforms](#supported-devices--os--platforms)
- [Quick Install](#quick-install)
- [Manual Installation](#manual-installation-advanced-users)
- [Usage](#usage)
- [Microphones](#microphones)
- [Lyrics](#lyrics-for-videos-without-them)
- [Note guide](#note-guide-and-note-charts)
- [Docker](#docker-instructions)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting-and-guides)

## Supported Devices / OS / Platforms

- OSX
- Windows
- Linux
- Raspberry Pi 4 or higher (Pi3 works ok with overclocking)

## Quick Install

For a streamlined installation that handles all dependencies (uv, ffmpeg, deno) and installs PiKaraoke, run the following in your terminal:

### Linux & macOS

```sh
curl -fsSL https://raw.githubusercontent.com/configure-ng/pikaraoke/master/build_scripts/install/install.sh | bash
```

### Windows (installer)

Download `PiKaraoke-<version>-setup.exe` from the [latest release](https://github.com/configure-ng/pikaraoke/releases) and run it. It adds Start Menu and desktop shortcuts.

Windows shows a blue "Windows protected your PC" screen because the installer is unsigned. Click **More info**, then **Run anyway**.

To uninstall, use **Settings > Apps**.

### Windows (PowerShell)

```powershell
irm https://raw.githubusercontent.com/configure-ng/pikaraoke/master/build_scripts/install/install.ps1 | iex
```

To uninstall, swap `install.ps1` for `uninstall.ps1`.

After installation, you can launch pikaraoke from the command line with `pikaraoke` or from a desktop shortcut. Re-running the above command will update a previous pikaraoke installation to the latest version.

Uninstalling removes the pikaraoke package and its shortcuts. Your songs, settings, ffmpeg, deno and uv are left in place.

## Manual installation (advanced users)

### Prerequisites

- A modern web browser (Chrome/Chromium/Edge recommended)
- Python 3.10 or greater: [Python downloads](https://www.python.org/downloads/)
- FFmpeg (preferably a build with lib-rubberband for transposing): [FFmpeg downloads](https://ffmpeg.org/download.html)
- A js runtime installed to your PATH. [Node.js](https://nodejs.org/en/download/) is most common, [Deno](https://deno.com/) is probably easiest for non-developers.

### Install the pikaraoke package

We recommend installing pikaraoke via [uv](https://github.com/astral-sh/uv).

```sh
uv tool install pikaraoke
```

You may alternately use the standard python `pip install pikaraoke` installer if you are familiar with virtual environments or you are not concerned with global package isolation.

## Usage

Run pikaraoke from the command line with:

```sh
pikaraoke
```

Launches the player in "headed" mode via your default browser. Scan the QR code to connect mobile remotes. Use `pikaraoke --headless` to run as a background server for external browsers.

See the help command `pikaraoke --help` for available options.

To upgrade to the latest version of pikaraoke, run:

```sh
uv tool upgrade pikaraoke
```

## Microphones

On Windows and Linux, plug in a USB microphone (such as a Rock Band / SingStar style game mic) and singers hear themselves through the speakers, with no extra software. USB mics switch on automatically the first time they are detected, including ones plugged in while PiKaraoke is running. Built-in laptop mics, webcams and headsets stay off so they cannot feed back.

- Adjust each mic's volume (up to 200% for quiet mics) from the audio settings icon on the home screen, or under Settings > Playback. On Linux this changes only what PiKaraoke plays, not the mic level other apps record.
- Switch any mic on or off, and tune latency, under Settings > Playback. Your choices are remembered.
- Turn the feature off entirely with `pikaraoke --no-enable-mic-passthrough`.

### Singing scores

After each song, every mic that was sung into gets a real score from 0 to 99, and a duet also shows the two singers' combined score. Karaoke tracks carry no melody to compare against, so PiKaraoke listens to the song's music instead: it scores how well each note sung fits the notes the music is playing at that moment, and how much of the song was sung. Singing in tune and all the way through scores high; mumbling, singing off-key or not singing scores low. A mic left near the speakers does not score from what it overhears.

The scoring cannot tell the melody from a harmony line that fits the music, so it rewards singing in tune rather than knowing the tune note for note. With no mic on, no score screen is shown. Turn scores off under Settings > Scoring.

Linux needs `pactl`, which works with both PulseAudio and PipeWire. The installer adds it (`pulseaudio-utils` on Debian/Ubuntu/Raspberry Pi OS, `libpulse` on Arch). On macOS, mic passthrough is opt-in: run `pikaraoke --enable-mic-passthrough`.

## Lyrics for videos without them

Not every karaoke video shows the words. For those, PiKaraoke can put lyrics on the TV. When a song is downloaded, and for every song already in the library when PiKaraoke starts, it looks up the song's timed lyrics on [LRCLIB](https://lrclib.net), an open lyrics database. The lyrics' timing follows the original recording, which rarely starts where a karaoke video does, so PiKaraoke also fetches the original's audio from YouTube and lines the two up by their music.

Lyrics stay off until someone turns them on. While a song plays, the **Lyrics** buttons on the home page set how they show, and anyone in the room can use them. PiKaraoke remembers the choice for that song.

- **Off / On** for most songs.
- **Off / Romaji / Kana** for Japanese songs. Romaji spells the words in the Latin alphabet; kana spells every kanji out in hiragana.
- **Earlier / Later** nudges the timing half a second at a time, if the lines run ahead of or behind the music.

Japanese readings come from a dictionary and are right for most lyrics, but a poetic reading a lyricist chose on purpose can come out as the word's usual one. Songs LRCLIB does not have get no lyrics. Lyrics show a line at a time; with a note chart (below), the line fills in as it is sung.

## Note guide and note charts

While a song plays, the TV can draw a note guide across the top of the screen: each mic's pitch as a line, moving as it is sung. Anyone can turn it on or off with the **Notes** buttons on the home page.

With the **note chart add-on**, the guide also shows the song's melody as bars scrolling toward a "now" line, and they fill in where a singer hits them. Karaoke videos carry no melody, so the add-on makes one from the song's original recording: it separates the singer's voice out of the music with a neural network, tracks its pitch, and turns that into notes. It uses the same original recording the lyrics are lined up against, or finds one on YouTube by name and checks it against the video. Songs with a chart are scored by the notes hit (an octave up or down counts); songs without one are scored as before.

- Charts are made in the background, one song at a time, as soon as a song is downloaded, and for the rest of the library when PiKaraoke starts. Songs in the singing queue are done first, so a chart is usually ready before the song is sung. A chart that failed (YouTube refusing a download, say) is tried once more each time PiKaraoke starts.
- A chart takes about 30 seconds on a modern graphics card and a few minutes on the CPU. AMD cards compile their kernels the first time, which takes several minutes once.
- **Earlier / Later** moves the lyrics and the note bars together.
- Charts are made by machine, so expect most notes to be right, not all: backing vocals, harmonies and very fast passages can confuse them.

The add-on is optional because it is large: PyTorch and the models take 3-6 GB with an NVIDIA card, about 15 GB with an AMD card on Linux. The installers ask whether to add it (or pass `--charts` to `install.sh`, `-Charts` to `install.ps1`), and pick the PyTorch build for the graphics card: CUDA for NVIDIA, ROCm for AMD on Linux, otherwise the CPU.

## Docker instructions

Run PiKaraoke in Docker using the command below. Note the requirements for port mapping, LAN IP specification, and persistent volume mounts (set to ~/.pikaraoke in the example for simplicity):

```sh
docker run -p 5555:5555 \
  -e TZ=Europe/London \
  -v ~/pikaraoke-songs:/app/pikaraoke-songs \
  -v ~/.pikaraoke:/home/pikaraoke/.pikaraoke \
  vicwomg/pikaraoke:latest \
  -u http://<YOUR_LAN_IP>:5555
```

Set `TZ` to your own timezone, taken from the "TZ identifier" column of the [tz database list](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).

For more information and a configurable docker-compose example, [see official Dockerhub repo](https://hub.docker.com/r/vicwomg/pikaraoke)

To run under a path prefix (e.g., `/karaoke`), add `--base-path /your-path` and set `-u` to the public origin. Example: `--base-path /karaoke -u https://example.com`. PiKaraoke will generate links under `/karaoke` automatically. For reverse proxy setups, the proxy must forward both HTTP and WebSocket traffic for that path to PiKaraoke. Path-based nginx-proxy setups typically also need `VIRTUAL_PATH=/karaoke` with `VIRTUAL_DEST=/`.

## Screenshots

<div style="display: flex; flex-wrap: wrap;">
<img width="250" alt="pikaraoke-nowplaying" src="https://raw.githubusercontent.com/configure-ng/pikaraoke/master/docs/images/screenshot-now-playing.png">
<img width="250" alt="pikaraoke-queue" src="https://raw.githubusercontent.com/configure-ng/pikaraoke/master/docs/images/screenshot-queue.png">
<img width="250" alt="pikaraoke-browse" src="https://raw.githubusercontent.com/configure-ng/pikaraoke/master/docs/images/screenshot-songs-all.png">
<img width="250" alt="pikaraoke-search" src="https://raw.githubusercontent.com/configure-ng/pikaraoke/master/docs/images/screenshot-songs-search.png">
<img width="250" alt="pikaraoke-add-new" src="https://raw.githubusercontent.com/configure-ng/pikaraoke/master/docs/images/screenshot-add-new.png">
<img width="250" alt="pikaraoke-play-history" src="https://raw.githubusercontent.com/configure-ng/pikaraoke/master/docs/images/screenshot-play-history.png">
<img width="400" alt="pikaraoke-tv2" src="https://raw.githubusercontent.com/configure-ng/pikaraoke/master/docs/images/tv.png">
</div>

## Contributing

Contributions are welcome. See the [development guide](https://github.com/vicwomg/pikaraoke/blob/master/docs/development.md) for setup, tests and the PR conventions.

## Troubleshooting and guides

See the [Troubleshooting wiki](https://github.com/vicwomg/pikaraoke/wiki/Troubleshooting) for help with issues.

There are also some great guides [on the wiki](https://github.com/vicwomg/pikaraoke/wiki/) to running pikaraoke in all manner of bizarre places including Android, Chromecast, and embedded TVs!
