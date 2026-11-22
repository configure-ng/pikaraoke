// Tests for the note guide: the melody's notes on the TV and each mic's pitch.

import { createRequire } from "node:module";
import { beforeEach, describe, expect, it, vi } from "vitest";

const require = createRequire(import.meta.url);

const loadGuide = () => {
  globalThis.PikaraokeConfig = { notesUrl: "/api/notes" };
  const path = require.resolve("../../pikaraoke/static/js/note-guide.js");
  delete require.cache[path];
  return require(path);
};

// [start, length, pitch]: three notes, then a gap, then one more.
const NOTES = [
  [10, 0.5, 60],
  [10.5, 0.5, 62],
  [11, 1, 64],
  [14, 1, 67],
];

describe("finding notes", () => {
  const { guideNoteAt } = loadGuide();

  it("finds the note sounding at a time", () => {
    expect(guideNoteAt(NOTES, 10.2)).toBe(0);
    expect(guideNoteAt(NOTES, 10.5)).toBe(1);
    expect(guideNoteAt(NOTES, 11.9)).toBe(2);
  });

  it("finds none in a gap, before the first or after the last", () => {
    expect(guideNoteAt(NOTES, 13)).toBe(-1);
    expect(guideNoteAt(NOTES, 2)).toBe(-1);
    expect(guideNoteAt(NOTES, 20)).toBe(-1);
    expect(guideNoteAt([], 10)).toBe(-1);
  });
});

describe("placing a sung note", () => {
  const { guideFold, guideRange } = loadGuide();

  it("folds a note sung an octave off onto the target", () => {
    expect(guideFold(48.3, 60)).toBeCloseTo(60.3);
    expect(guideFold(73, 60)).toBe(61);
    expect(guideFold(55, 60)).toBe(55);
  });

  it("spans the song's range with a margin, or a voice's range without a chart", () => {
    expect(guideRange(NOTES)).toEqual({ low: 58, high: 69 });
    expect(guideRange([])).toEqual({ low: 52, high: 76 });
  });

  it("leaves a stray note out of the range rather than squash the rest", () => {
    const melody = Array.from({ length: 60 }, (_, i) => [i, 0.5, 60 + (i % 5)]);
    melody.push([61, 0.5, 90]);
    expect(guideRange(melody)).toEqual({ low: 58, high: 66 });
  });
});

describe("the lyrics wipe", () => {
  const { guideLineProgress } = loadGuide();

  it("follows the notes sung in the line", () => {
    expect(guideLineProgress(NOTES, 10, 13, 9.9)).toBe(0);
    expect(guideLineProgress(NOTES, 10, 13, 10.25)).toBeCloseTo(0.5 / 3);
    expect(guideLineProgress(NOTES, 10, 13, 11.5)).toBeCloseTo(2.5 / 3);
    expect(guideLineProgress(NOTES, 10, 13, 12.5)).toBe(1);
  });

  it("has nothing to go by in a line without notes", () => {
    expect(guideLineProgress(NOTES, 12.5, 13.5, 13)).toBeNull();
    expect(guideLineProgress([], 10, 13, 11)).toBeNull();
  });
});

describe("the guide", () => {
  let NoteGuide;
  const playing = (extra = {}) => ({
    now_playing_url: "/stream/song",
    now_playing_chart: "found",
    note_guide: true,
    now_playing_lyrics: { offset: 2 },
    now_playing_transpose: 0,
    ...extra,
  });

  beforeEach(async () => {
    globalThis.fetch = vi.fn(async () => ({ ok: true, json: async () => NOTES }));
    ({ NoteGuide } = loadGuide());
    NoteGuide.update(playing());
    await vi.waitFor(() => expect(NoteGuide.lineProgress(12, 15, 12.6)).not.toBeNull());
  });

  it("fetches the chart and places it by the lyrics offset", () => {
    expect(globalThis.fetch).toHaveBeenCalledWith("/api/notes");
    // The first note, at 10 in the recording, sounds at 12 in the video.
    expect(NoteGuide.lineProgress(12, 15, 12.25)).toBeCloseTo(0.5 / 3);
  });

  it("shows while on with a chart, and not when turned off", () => {
    expect(NoteGuide.visible(12)).toBe(true);
    NoteGuide.update(playing({ note_guide: false }));
    expect(NoteGuide.visible(12)).toBe(false);
  });

  it("without a chart, shows only while someone sings", () => {
    NoteGuide.update(playing({ now_playing_url: "/stream/other", now_playing_chart: "missing" }));
    expect(NoteGuide.visible(5)).toBe(false);
    NoteGuide.onPitch({ mic: 1, frames: [[5, 60.1], [5.02, null]] });
    expect(NoteGuide.visible(5.5)).toBe(true);
    expect(NoteGuide.visible(10)).toBe(false);
  });

  it("forgets the chart when the song changes", () => {
    NoteGuide.update(playing({ now_playing_url: "/stream/other", now_playing_chart: null }));
    expect(NoteGuide.lineProgress(12, 15, 12.6)).toBeNull();
  });
});
