// Render a short buffered timeline instead of stepping at the network tick rate.
// Gameplay, drag steering and success checks still use the latest server state.
(() => {
  class PushTMotion {
    constructor(delay = 60) {
      this.delay = delay;
      this.clear();
    }

    clear() { this.frames = []; }

    push(state, now) {
      const previous = this.frames.at(-1)?.state;
      if (previous && (previous.round_id !== state.round_id
          || previous.phase !== state.phase || previous.paused !== state.paused
          || JSON.stringify(previous.goal) !== JSON.stringify(state.goal)
          || (state.left?.t ?? 0) < (previous.left?.t ?? 0))) this.clear();
      if (state.paused || state.phase !== "race") this.clear();
      if (this.frames.at(-1)?.time === now) this.frames.pop();
      this.frames.push({ state, time: now });
      if (this.frames.length > 12) this.frames.shift();
    }

    sample(now) {
      if (!this.frames.length) return null;
      const target = now - this.delay;
      while (this.frames.length > 2 && this.frames[1].time <= target) this.frames.shift();
      const first = this.frames[0];
      const last = this.frames[1] || first;
      const blend = last.time > first.time
        ? Math.max(0, Math.min(1, (target - first.time) / (last.time - first.time))) : 1;
      const result = { ...last.state };
      for (const name of ["left", "right"]) {
        const side = last.state[name];
        if (!side?.states) continue;
        const before = first.state[name]?.states;
        const states = side.states.map((pose, i) => {
          const old = before?.[i];
          const output = [...pose];
          if (old) {
            output[0] = old[0] + (pose[0] - old[0]) * blend;
            output[1] = old[1] + (pose[1] - old[1]) * blend;
            // Interpolate across +/- pi by the short path.
            const angle = Math.atan2(Math.sin(pose[2] - old[2]), Math.cos(pose[2] - old[2]));
            output[2] = old[2] + angle * blend;
          }
          return output;
        });
        // The welded T is one rigid shape. Draw its bar from the same pose as
        // its stem so small solver constraint errors cannot make it wobble.
        const offset = last.state.shape?.bar_offset;
        if (states.length === 3 && Number.isFinite(offset) && side.geom?.length === 3) {
          const stem = states[1], bar = states[2];
          const height = side.geom[1][1] + side.geom[2][1];
          const c = Math.cos(stem[2]), s = Math.sin(stem[2]);
          bar[0] = stem[0] + c * offset - s * height;
          bar[1] = stem[1] + s * offset + c * height;
          bar[2] = stem[2];
        }
        result[name] = { ...side, states };
      }
      return result;
    }
  }
  window.PushTMotion = PushTMotion;
})();
