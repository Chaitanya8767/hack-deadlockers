/**
 * AudioWorkletProcessor for capturing remote caller audio,
 * downsampling to 16 kHz mono, converting to Int16 PCM,
 * and emitting ~250 ms frames (4000 samples = 8000 bytes).
 */
class PCMWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.frameSamples = 4000; // 250ms at 16kHz
    this.buffer = new Int16Array(this.frameSamples);
    this.bufferIndex = 0;
    this.resampleRatio = sampleRate / this.targetSampleRate;
    this.samplePosition = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }

    // Mix down to mono if multiple channels exist, otherwise use channel 0
    const channels = input;
    const numChannels = channels.length;
    const inputLength = channels[0].length;

    while (this.samplePosition < inputLength) {
      const idx = Math.floor(this.samplePosition);
      const nextIdx = Math.min(idx + 1, inputLength - 1);
      const frac = this.samplePosition - idx;

      let sample = 0;
      for (let c = 0; c < numChannels; c++) {
        const s0 = channels[c][idx];
        const s1 = channels[c][nextIdx];
        sample += s0 + frac * (s1 - s0);
      }
      sample /= numChannels;

      // Clamp to [-1.0, 1.0] and convert to 16-bit signed integer
      const clamped = Math.max(-1.0, Math.min(1.0, sample));
      const int16 = clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);

      this.buffer[this.bufferIndex++] = int16;

      if (this.bufferIndex >= this.frameSamples) {
        // Send copy of the 250ms frame
        const frame = new Int16Array(this.buffer);
        this.port.postMessage(frame.buffer, [frame.buffer]);
        this.bufferIndex = 0;
      }

      this.samplePosition += this.resampleRatio;
    }

    this.samplePosition -= inputLength;
    return true;
  }
}

registerProcessor('pcm-worklet-processor', PCMWorkletProcessor);

