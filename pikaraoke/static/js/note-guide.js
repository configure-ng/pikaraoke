// The note guide: the melody as bars scrolling toward a "now" line across the
// top of the TV, each mic's pitch drawn over them as it is sung, and the bars
// filling in where a singer hits them.
//
// Bars come from the song's note chart (made with the chart add-on); without
// one, the lane still draws each singer's pitch. The chart is in the time of
// the recording it was made from, so the lyrics' offset places it on the video.

// The lane shows this much of the song: a little just sung, more to come.
const GUIDE_PAST_S = 1.5;
const GUIDE_FUTURE_S = 4.5;
// The lane spans the song's range plus this margin, in semitones.
const GUIDE_MARGIN = 2;
// Without a chart, a range a singing voice sits in.
const GUIDE_DEFAULT_RANGE = { low: 52, high: 76 };
// A sung note within this of a bar, octaves aside, hits it.
const GUIDE_HIT_TOLERANCE = 1.0;
// Pitch frames are kept this long, enough to cross the visible past.
const GUIDE_TRAIL_S = 3;
// Without a chart, the lane shows only while someone has sung this recently.
const GUIDE_IDLE_S = 3;
// Each pitch frame covers this much song: the scorer's hop.
const GUIDE_FRAME_S = 0.02;
// The least a bar's thickness may be, as a share of the lane's height.
const GUIDE_BAR_SHARE = 0.09;

// Index of the note sounding at `time`, or -1. notes are [start, length, pitch]
// in video time, sorted by start.
const guideNoteAt = (notes, time) => {
  let lo = 0;
  let hi = notes.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (notes[mid][0] <= time) {
      found = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found >= 0 && time < notes[found][0] + notes[found][1] ? found : -1;
};

// A sung note moved by octaves to sit nearest the target: a man singing a
// woman's melody an octave down is singing it right.
const guideFold = (sung, target) => sung + 12 * Math.round((target - sung) / 12);

// The pitch span the lane draws: the whole song's, so bars do not jump about.
// A stray note far outside the melody is pinned to the edge rather than
// squashing every bar to make room for it.
const guideRange = (notes) => {
  if (!notes || notes.length === 0) return { ...GUIDE_DEFAULT_RANGE };
  const pitches = notes.map((note) => note[2]).sort((a, b) => a - b);
  const at = (share) => pitches[Math.min(pitches.length - 1, Math.floor(share * pitches.length))];
  return { low: at(0.03) - GUIDE_MARGIN, high: at(0.97) + GUIDE_MARGIN };
};

// How far through a lyrics line the singing is, 0-1, from the notes sung in
// it so far; null when the chart has no notes in the line to go by.
const guideLineProgress = (notes, start, end, time) => {
  if (!notes || notes.length === 0) return null;
  let count = 0;
  let done = 0;
  for (const [noteStart, length] of notes) {
    if (noteStart < start - 0.2) continue;
    if (noteStart >= end) break;
    count += 1;
    done += Math.min(1, Math.max(0, (time - noteStart) / length));
  }
  return count ? done / count : null;
};

const NoteGuide = (() => {
  let canvas = null;
  let getTime = () => 0;
  let song = null;
  let fetched = false;
  let rawNotes = null; // [start, length, pitch] in the recording's time
  let notes = []; // in video time, transposed: what is drawn
  let offset = 0;
  let semitones = 0;
  let enabled = true;
  let status = null;
  let range = { ...GUIDE_DEFAULT_RANGE };
  // Per mic: [[time, note or null]], and per chart note the times it was hit.
  let trails = {};
  let hits = new Map();
  let lastSung = -Infinity;
  let colors = null;

  const place = () => {
    notes = (rawNotes || []).map(([start, length, pitch]) => [
      start + offset,
      length,
      pitch + semitones,
    ]);
    range = guideRange(notes);
  };

  const fetchNotes = async (forSong) => {
    try {
      const response = await fetch(PikaraokeConfig.notesUrl);
      const data = response.ok ? await response.json() : null;
      if (forSong === song) {
        rawNotes = data;
        place();
      }
    } catch (e) {
      console.log("Note chart unavailable", e);
    }
  };

  // From each now_playing update: which song, its chart, offset and key.
  const update = (np) => {
    const playing = np.now_playing_url || null;
    if (playing !== song) {
      song = playing;
      rawNotes = null;
      fetched = false;
      trails = {};
      hits = new Map();
      lastSung = -Infinity;
      place();
    }
    enabled = np.note_guide !== false;
    status = np.now_playing_chart || null;
    const lyrics = np.now_playing_lyrics || null;
    const newOffset = lyrics ? lyrics.offset || 0 : 0;
    const newSemitones = np.now_playing_transpose || 0;
    if (newOffset !== offset || newSemitones !== semitones) {
      offset = newOffset;
      semitones = newSemitones;
      hits = new Map();
      place();
    }
    if (song && status === "found" && !fetched) {
      fetched = true;
      fetchNotes(song);
    }
  };

  // A mic's newest pitch frames, from the server: {mic, frames: [[time, note]]}.
  const onPitch = (data) => {
    if (!song) return;
    const trail = (trails[data.mic] = trails[data.mic] || []);
    for (const [time, note] of data.frames) {
      trail.push([time, note]);
      if (note === null) continue;
      lastSung = time;
      const i = guideNoteAt(notes, time);
      if (i >= 0 && Math.abs(guideFold(note, notes[i][2]) - notes[i][2]) <= GUIDE_HIT_TOLERANCE) {
        if (!hits.has(i)) hits.set(i, []);
        hits.get(i).push(time);
      }
    }
    const cutoff = trail[trail.length - 1][0] - GUIDE_TRAIL_S;
    while (trail.length && trail[0][0] < cutoff) trail.shift();
  };

  const visible = (time) =>
    Boolean(enabled && song && (notes.length > 0 || time - lastSung < GUIDE_IDLE_S));

  const readColors = () => {
    const style = getComputedStyle(canvas);
    const get = (name) => style.getPropertyValue(name).trim();
    return {
      bar: get("--note-bar"),
      hit: get("--note-hit"),
      now: get("--note-now"),
      mics: [get("--note-mic-1"), get("--note-mic-2"), get("--note-mic-3"), get("--note-mic-4")],
    };
  };

  const draw = () => {
    const time = getTime();
    const show = visible(time);
    if ((canvas.style.display !== "none") !== show) canvas.style.display = show ? "" : "none";
    if (!show) return;

    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const scale = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * scale) || canvas.height !== Math.round(height * scale)) {
      canvas.width = Math.round(width * scale);
      canvas.height = Math.round(height * scale);
      colors = null;
    }
    colors = colors || readColors();
    const ctx = canvas.getContext("2d");
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const span = GUIDE_PAST_S + GUIDE_FUTURE_S;
    const nowX = (GUIDE_PAST_S / span) * width;
    const x = (t) => nowX + ((t - time) / span) * width;
    const rows = range.high - range.low + 1;
    const rowHeight = height / rows;
    const clamp = (pitch) => Math.min(range.high, Math.max(range.low, pitch));
    const y = (pitch) => height - (clamp(pitch) - range.low + 1) * rowHeight;
    // Bars thick enough to read from the sofa; neighbouring notes may overlap.
    const barHeight = Math.max(rowHeight * 0.9, height * GUIDE_BAR_SHARE);

    // The bars, and where each was hit.
    const from = time - GUIDE_PAST_S;
    const to = time + GUIDE_FUTURE_S;
    notes.forEach(([start, length, pitch], i) => {
      if (start > to || start + length < from) return;
      const top = y(pitch) + (rowHeight - barHeight) / 2;
      const left = x(start);
      const barWidth = Math.max(barHeight / 2, (length / span) * width - 2);
      // Bars still to sing are bright; bars passed by are dimmed, unless hit.
      ctx.fillStyle = colors.bar;
      ctx.globalAlpha = start + length < time ? 0.35 : 0.9;
      roundRect(ctx, left, top, barWidth, barHeight, barHeight / 2);
      ctx.fill();
      const noteHits = hits.get(i);
      if (noteHits) {
        ctx.save();
        roundRect(ctx, left, top, barWidth, barHeight, barHeight / 2);
        ctx.clip();
        ctx.fillStyle = colors.hit;
        ctx.globalAlpha = 1;
        const frame = (GUIDE_FRAME_S / span) * width;
        for (const t of noteHits) ctx.fillRect(x(t) - frame, top, frame * 2 + 1, barHeight);
        ctx.restore();
      }
    });
    ctx.globalAlpha = 1;

    // The "now" line.
    ctx.fillStyle = colors.now;
    ctx.fillRect(nowX - 1, 0, 2, height);

    // Each mic's pitch, folded onto the bar it is singing against.
    Object.keys(trails).forEach((mic) => {
      const color = colors.mics[(Number(mic) - 1) % colors.mics.length];
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = Math.max(3, barHeight * 0.3);
      ctx.lineCap = "round";
      ctx.beginPath();
      let drawing = false;
      let last = null;
      for (const [t, note] of trails[mic]) {
        if (t < from || t > time + 0.1 || note === null) {
          drawing = false;
          continue;
        }
        const placed = placePitch(note, t);
        const px = x(t);
        const py = y(placed) + rowHeight / 2;
        if (drawing) ctx.lineTo(px, py);
        else ctx.moveTo(px, py);
        drawing = true;
        last = [px, py];
      }
      ctx.stroke();
      if (last && time - lastSung < 0.3) {
        ctx.beginPath();
        ctx.arc(last[0], last[1], ctx.lineWidth * 1.2, 0, 2 * Math.PI);
        ctx.fill();
      }
    });
  };

  // Where to draw a sung note: on the bar sounding then, octave aside; with
  // no bar, inside the lane's range.
  const placePitch = (note, t) => {
    const i = guideNoteAt(notes, t);
    if (i >= 0) return guideFold(note, notes[i][2]);
    const middle = (range.low + range.high) / 2;
    return guideFold(note, middle);
  };

  const roundRect = (ctx, x, y, w, h, r) => {
    const radius = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + w, y, x + w, y + h, radius);
    ctx.arcTo(x + w, y + h, x, y + h, radius);
    ctx.arcTo(x, y + h, x, y, radius);
    ctx.arcTo(x, y, x + w, y, radius);
    ctx.closePath();
  };

  const loop = () => {
    try {
      draw();
    } catch (e) {
      console.log("Note guide draw failed", e);
    }
    window.requestAnimationFrame(loop);
  };

  const start = (laneCanvas, timeSource) => {
    canvas = laneCanvas;
    getTime = timeSource;
    window.requestAnimationFrame(loop);
  };

  // How far through a lyrics line the singing is, for the lyrics wipe.
  const lineProgress = (start, end, time) => guideLineProgress(notes, start, end, time);

  return { start, update, onPitch, lineProgress, visible };
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    NoteGuide,
    guideNoteAt,
    guideFold,
    guideRange,
    guideLineProgress,
  };
}
