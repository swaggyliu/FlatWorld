/* Original, quiet Web Audio score and contact sound. No downloads or autoplay. */
(() => {
  class PushTAudio {
    constructor() {
      this.ctx = null;
      this.musicEnabled = true;
      this.sfxEnabled = true;
      this.playing = false;
      this.visible = true;
      this.timer = null;
      this.step = 0;
      this.nextNote = 0;
      this.lastPush = -Infinity;
    }

    unlock() {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) return;
      try {
        if (!this.ctx) {
          this.ctx = new AudioContext();
          this.music = this.ctx.createGain();
          this.effects = this.ctx.createGain();
          this.winGain = this.ctx.createGain();
          this.winGain.gain.value = 0;
          this.music.connect(this.ctx.destination);
          this.effects.connect(this.ctx.destination);
          this.winGain.connect(this.ctx.destination);
          this.sync();
        }
        if (this.ctx.state === "suspended") this.ctx.resume().catch(() => {});
        this.schedule();
      } catch (_) { /* Sound is optional on browsers without an audio device. */ }
    }

    configure(music, sfx) {
      this.musicEnabled = music;
      this.sfxEnabled = sfx;
      this.sync();
    }

    setPlaying(playing, visible = true) {
      this.playing = playing;
      this.visible = visible;
      this.sync();
    }

    sync() {
      if (!this.ctx) return;
      const active = this.playing && this.visible;
      this.music.gain.setTargetAtTime(active && this.musicEnabled ? 0.22 : 0, this.ctx.currentTime, 0.04);
      this.effects.gain.setTargetAtTime(active && this.sfxEnabled ? 0.3 : 0, this.ctx.currentTime, 0.02);
      // Victory remains audible after the round stops, while respecting mute and visibility.
      this.winGain.gain.setTargetAtTime(this.visible && this.sfxEnabled ? 0.5 : 0, this.ctx.currentTime, 0.02);
      if (active && this.musicEnabled) {
        if (this.timer === null) {
          this.nextNote = this.ctx.currentTime + 0.04;
          this.timer = setInterval(() => this.schedule(), 100);
          this.schedule();
        }
      } else if (this.timer !== null) {
        clearInterval(this.timer);
        this.timer = null;
      }
    }

    tone(midi, start, duration, volume, channel, type = "sine") {
      const osc = this.ctx.createOscillator();
      const envelope = this.ctx.createGain();
      osc.type = type;
      osc.frequency.value = 440 * Math.pow(2, (midi - 69) / 12);
      envelope.gain.setValueAtTime(0, start);
      envelope.gain.linearRampToValueAtTime(volume, start + 0.025);
      envelope.gain.exponentialRampToValueAtTime(0.0001, start + duration);
      osc.connect(envelope);
      envelope.connect(channel);
      osc.onended = () => { osc.disconnect(); envelope.disconnect(); };
      osc.start(start);
      osc.stop(start + duration + 0.02);
    }

    schedule() {
      if (!this.ctx || this.ctx.state !== "running" || !this.playing || !this.visible || !this.musicEnabled) return;
      const beat = 60 / 78 / 2;
      // Cmaj7 / Am7 / Fmaj7 / G6, with a gentle pentatonic melody.
      const chords = [[48, 55, 59, 64], [45, 52, 55, 60], [41, 48, 52, 57], [43, 50, 55, 59]];
      const melody = [72, null, 76, 79, null, 76, 74, null,
        72, null, 69, 72, 76, null, 74, null,
        69, null, 72, 76, null, 72, 69, null,
        71, null, 74, 79, 76, null, 74, null];
      if (this.nextNote < this.ctx.currentTime) this.nextNote = this.ctx.currentTime + 0.04;
      while (this.nextNote < this.ctx.currentTime + 0.2) {
        const index = this.step % melody.length;
        const chord = chords[Math.floor(index / 8)];
        if (index % 8 === 0) {
          for (const note of chord) this.tone(note, this.nextNote, 2.7, 0.025, this.music, "triangle");
        }
        if (melody[index] !== null) this.tone(melody[index], this.nextNote, 0.85, 0.09, this.music);
        this.step += 1;
        this.nextNote += beat;
      }
    }

    win() {
      if (!this.ctx || !this.sfxEnabled || !this.visible) return;
      try {
        if (this.ctx.state === "suspended") this.ctx.resume().catch(() => {});
        const now = this.ctx.currentTime + 0.02;
        const arp = [72, 76, 79, 84]; // Same C5 E5 G5 C6 fanfare as the mini program.
        for (let i = 0; i < arp.length; i++) {
          this.tone(arp[i], now + i * 0.09, 0.35, 0.16, this.winGain, "triangle");
        }
        for (const midi of [84, 88, 91]) {
          this.tone(midi, now + 0.3, 0.9, 0.1, this.winGain, "sine");
        }
      } catch (_) { /* Sound is optional on browsers without an audio device. */ }
    }

    push(strength) {
      if (!this.ctx || this.ctx.state !== "running" || !this.sfxEnabled || !this.playing || !this.visible) return;
      const now = this.ctx.currentTime;
      if (now - this.lastPush < 0.16) return;
      this.lastPush = now;
      const osc = this.ctx.createOscillator();
      const envelope = this.ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(170 + Math.min(1, strength) * 70, now);
      osc.frequency.exponentialRampToValueAtTime(65, now + 0.11);
      envelope.gain.setValueAtTime(0, now);
      envelope.gain.linearRampToValueAtTime(0.12 + Math.min(1, strength) * 0.12, now + 0.008);
      envelope.gain.exponentialRampToValueAtTime(0.0001, now + 0.14);
      osc.connect(envelope);
      envelope.connect(this.effects);
      osc.onended = () => { osc.disconnect(); envelope.disconnect(); };
      osc.start(now);
      osc.stop(now + 0.16);
    }
  }
  window.PushTAudio = PushTAudio;
})();
