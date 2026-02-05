// SPDX-FileCopyrightText: 2025 LiveKit, Inc.
//
// SPDX-License-Identifier: Apache-2.0
import { type APIConnectOptions, AudioByteStream, log, shortuuid, tts } from '@livekit/agents';
import type { AudioFrame } from '@livekit/rtc-node';
import type { TTSModels } from './models.js';

const SIMPLISMART_TTS_BASE_URL = 'https://api.simplismart.live/tts';
const SAMPLE_RATE = 24000;
const NUM_CHANNELS = 1;

export interface TTSOptions {
  apiKey?: string;
  baseUrl?: string;
  model?: TTSModels | string;
  voice?: string;
  temperature?: number;
  topP?: number;
  repetitionPenalty?: number;
  maxTokens?: number;
}

const defaultTTSOptions: Required<Omit<TTSOptions, 'apiKey' | 'baseUrl'>> &
  Pick<TTSOptions, 'apiKey' | 'baseUrl'> = {
  apiKey: process.env.SIMPLISMART_API_KEY,
  baseUrl: SIMPLISMART_TTS_BASE_URL,
  model: 'canopylabs/orpheus-3b-0.1-ft',
  voice: 'tara',
  temperature: 0.7,
  topP: 0.9,
  repetitionPenalty: 1.5,
  maxTokens: 1000,
};

export class TTS extends tts.TTS {
  #opts: TTSOptions;
  #logger = log();
  label = 'simplismart.TTS';
  private abortController = new AbortController();

  constructor(opts: Partial<TTSOptions> = {}) {
    const mergedOpts = { ...defaultTTSOptions, ...opts };
    super(SAMPLE_RATE, NUM_CHANNELS, { streaming: false });

    if (mergedOpts.apiKey === undefined) {
      throw new Error(
        'SimpliSmart API key is required, whether as an argument or as $SIMPLISMART_API_KEY',
      );
    }

    this.#opts = mergedOpts;
  }

  get model(): string {
    return this.#opts.model ?? defaultTTSOptions.model;
  }

  get provider(): string {
    return 'SimpliSmart';
  }

  synthesize(
    text: string,
    connOptions?: APIConnectOptions,
    abortSignal?: AbortSignal,
  ): ChunkedStream {
    return new ChunkedStream(this, text, this.#opts, connOptions, abortSignal);
  }

  stream(): tts.SynthesizeStream {
    throw new Error('Streaming is not supported on SimpliSmart TTS');
  }

  async close(): Promise<void> {
    this.abortController.abort();
  }
}

export class ChunkedStream extends tts.ChunkedStream {
  #opts: TTSOptions;
  #logger = log();
  #tts: TTS;
  #text: string;
  label = 'simplismart.ChunkedStream';

  constructor(
    tts: TTS,
    text: string,
    opts: TTSOptions,
    connOptions?: APIConnectOptions,
    abortSignal?: AbortSignal,
  ) {
    super(text, tts, connOptions, abortSignal);
    this.#tts = tts;
    this.#opts = opts;
    this.#text = text;
  }

  protected async run() {
    const payload: Record<string, string | number> = {
      prompt: this.#text,
      voice: this.#opts.voice ?? defaultTTSOptions.voice,
      model: this.#opts.model ?? defaultTTSOptions.model,
      temperature: this.#opts.temperature ?? defaultTTSOptions.temperature,
      top_p: this.#opts.topP ?? defaultTTSOptions.topP,
      repetition_penalty: this.#opts.repetitionPenalty ?? defaultTTSOptions.repetitionPenalty,
      max_tokens: this.#opts.maxTokens ?? defaultTTSOptions.maxTokens,
    };

    const baseUrl = this.#opts.baseUrl ?? SIMPLISMART_TTS_BASE_URL;
    const timeout = 30000; // Default timeout, connOptions is private
    const requestId = shortuuid();

    try {
      const controller = this.abortController;

      const timeoutId = setTimeout(() => controller.abort(), timeout);

      const response = await fetch(baseUrl, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.#opts.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        this.#logger.error(`Simplismart TTS API error: ${response.status} - ${errorText}`);
        throw new Error(`Simplismart TTS API Error: ${errorText}`);
      }

      if (!response.body) {
        throw new Error('Simplismart TTS API returned no response body');
      }

      // Initialize the output
      const sampleRate = SAMPLE_RATE;
      const numChannels = NUM_CHANNELS;
      const bstream = new AudioByteStream(sampleRate, numChannels);

      let lastFrame: AudioFrame | undefined;
      const sendLastFrame = (segmentId: string, final: boolean) => {
        if (lastFrame) {
          if (!this.queue.closed) {
            this.queue.put({ requestId, segmentId, frame: lastFrame, final });
          }
          lastFrame = undefined;
        }
      };

      // Read the streaming response
      const reader = response.body.getReader();
      const segmentId = requestId;

      try {
        while (true) {
          const { done, value } = await reader.read();

          if (done) {
            break;
          }

          // Process audio chunks
          const buffer = value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
          for (const frame of bstream.write(buffer)) {
            sendLastFrame(segmentId, false);
            lastFrame = frame;
          }
        }

        // Flush remaining frames
        for (const frame of bstream.flush()) {
          sendLastFrame(segmentId, false);
          lastFrame = frame;
        }

        // Send final frame
        sendLastFrame(segmentId, true);
      } finally {
        reader.releaseLock();
      }

      if (!this.queue.closed) {
        this.queue.close();
      }
    } catch (error: unknown) {
      if (error instanceof Error && error.name === 'AbortError') {
        if (!this.queue.closed) {
          this.queue.close();
        }
        return;
      }
      if (error instanceof Error) {
        this.#logger.error(`Error during Simplismart TTS processing: ${error.message}`);
        throw error;
      }
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.#logger.error(`Unexpected error in Simplismart TTS: ${errorMessage}`);
      throw new Error(`Unexpected error in Simplismart TTS: ${errorMessage}`);
    } finally {
      if (!this.queue.closed) {
        this.queue.close();
      }
    }
  }
}
