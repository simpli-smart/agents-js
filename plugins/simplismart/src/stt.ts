// SPDX-FileCopyrightText: 2025 LiveKit, Inc.
//
// SPDX-License-Identifier: Apache-2.0
import {
  type APIConnectOptions,
  type AudioBuffer,
  AudioByteStream,
  log,
  mergeFrames,
  stt,
  waitForAbort,
} from '@livekit/agents';
import type { AudioFrame } from '@livekit/rtc-node';
import { WebSocket } from 'ws';
import type { STTModels } from './models.js';

const SIMPLISMART_BASE_URL = 'https://api.simplismart.live/predict';
const SIMPLISMART_WS_URL = 'wss://api.simplismart.live/ws/audio';

export interface STTOptions {
  apiKey?: string;
  baseUrl?: string;
  streaming?: boolean;
  model?: STTModels | string;
  language?: string;
  task?: 'transcribe' | 'translate';
  withoutTimestamps?: boolean;
  vadModel?: 'silero' | 'frame';
  vadFilter?: boolean;
  vadOnset?: number | null;
  vadOffset?: number | null;
  minSpeechDurationMs?: number;
  maxSpeechDurationS?: number;
  minSilenceDurationMs?: number;
  speechPadMs?: number;
  initialPrompt?: string | null;
  hotwords?: string | null;
  numSpeakers?: number;
  compressionRatioThreshold?: number | null;
  beamSize?: number;
  temperature?: number;
  multilingual?: boolean;
  maxTokens?: number | null;
  logProbThreshold?: number | null;
  lengthPenalty?: number;
  repetitionPenalty?: number;
  strictHallucinationReduction?: boolean;
}

const defaultSTTOptions: Required<
  Omit<
    STTOptions,
    | 'apiKey'
    | 'baseUrl'
    | 'initialPrompt'
    | 'hotwords'
    | 'vadOnset'
    | 'vadOffset'
    | 'compressionRatioThreshold'
    | 'maxTokens'
    | 'logProbThreshold'
  >
> &
  Pick<
    STTOptions,
    | 'apiKey'
    | 'baseUrl'
    | 'initialPrompt'
    | 'hotwords'
    | 'vadOnset'
    | 'vadOffset'
    | 'compressionRatioThreshold'
    | 'maxTokens'
    | 'logProbThreshold'
  > = {
  apiKey: process.env.SIMPLISMART_API_KEY,
  baseUrl: SIMPLISMART_BASE_URL,
  streaming: false,
  model: 'openai/whisper-large-v3-turbo',
  language: 'en',
  task: 'transcribe',
  withoutTimestamps: true,
  vadModel: 'frame',
  vadFilter: true,
  vadOnset: 0.5,
  vadOffset: null,
  minSpeechDurationMs: 0,
  maxSpeechDurationS: 30,
  minSilenceDurationMs: 2000,
  speechPadMs: 400,
  initialPrompt: null,
  hotwords: null,
  numSpeakers: 0,
  compressionRatioThreshold: 2.4,
  beamSize: 4,
  temperature: 0.0,
  multilingual: false,
  maxTokens: 400,
  logProbThreshold: -1.0,
  lengthPenalty: 1,
  repetitionPenalty: 1.01,
  strictHallucinationReduction: false,
};

export class STT extends stt.STT {
  #opts: STTOptions;
  #logger = log();
  label = 'simplismart.STT';
  private abortController = new AbortController();

  constructor(opts: Partial<STTOptions> = {}) {
    const mergedOpts = { ...defaultSTTOptions, ...opts };
    super({
      streaming: mergedOpts.streaming ?? false,
      interimResults: false,
      alignedTranscript: 'word',
    });

    if (mergedOpts.apiKey === undefined) {
      throw new Error(
        'SimpliSmart API key is required, whether as an argument or as $SIMPLISMART_API_KEY',
      );
    }

    this.#opts = mergedOpts;
  }

  get provider(): string {
    return 'Simplismart';
  }

  get model(): string {
    return this.#opts.model ?? defaultSTTOptions.model;
  }

  #createWav(buffer: AudioBuffer): Buffer {
    const frame = mergeFrames(buffer);
    const bitsPerSample = 16;
    const byteRate = (frame.sampleRate * frame.channels * bitsPerSample) / 8;
    const blockAlign = (frame.channels * bitsPerSample) / 8;

    const header = Buffer.alloc(44);
    header.write('RIFF', 0);
    header.writeUInt32LE(36 + frame.data.byteLength, 4);
    header.write('WAVE', 8);
    header.write('fmt ', 12);
    header.writeUInt32LE(16, 16);
    header.writeUInt16LE(1, 20);
    header.writeUInt16LE(frame.channels, 22);
    header.writeUInt32LE(frame.sampleRate, 24);
    header.writeUInt32LE(byteRate, 28);
    header.writeUInt16LE(blockAlign, 32);
    header.writeUInt16LE(16, 34);
    header.write('data', 36);
    header.writeUInt32LE(frame.data.byteLength, 40);
    return Buffer.concat([header, Buffer.from(frame.data.buffer)]);
  }

  async _recognize(
    buffer: AudioBuffer,
    options?: { language?: string; connOptions?: APIConnectOptions },
    abortSignal?: AbortSignal,
  ): Promise<stt.SpeechEvent> {
    const resolvedLanguage = options?.language ?? this.#opts.language ?? 'en';
    const wavBytes = this.#createWav(buffer);
    const audioB64 = wavBytes.toString('base64');

    const payload: Record<string, any> = {
      audio_data: audioB64,
      language: resolvedLanguage,
      model: this.#opts.model ?? defaultSTTOptions.model,
      task: this.#opts.task ?? defaultSTTOptions.task,
      without_timestamps: this.#opts.withoutTimestamps ?? defaultSTTOptions.withoutTimestamps,
      vad_model: this.#opts.vadModel ?? defaultSTTOptions.vadModel,
      vad_filter: this.#opts.vadFilter ?? defaultSTTOptions.vadFilter,
      min_speech_duration_ms:
        this.#opts.minSpeechDurationMs ?? defaultSTTOptions.minSpeechDurationMs,
      max_speech_duration_s: this.#opts.maxSpeechDurationS ?? defaultSTTOptions.maxSpeechDurationS,
      min_silence_duration_ms:
        this.#opts.minSilenceDurationMs ?? defaultSTTOptions.minSilenceDurationMs,
      speech_pad_ms: this.#opts.speechPadMs ?? defaultSTTOptions.speechPadMs,
      num_speakers: this.#opts.numSpeakers ?? defaultSTTOptions.numSpeakers,
      beam_size: this.#opts.beamSize ?? defaultSTTOptions.beamSize,
      temperature: this.#opts.temperature ?? defaultSTTOptions.temperature,
      multilingual: this.#opts.multilingual ?? defaultSTTOptions.multilingual,
      length_penalty: this.#opts.lengthPenalty ?? defaultSTTOptions.lengthPenalty,
      repetition_penalty: this.#opts.repetitionPenalty ?? defaultSTTOptions.repetitionPenalty,
      strict_hallucination_reduction:
        this.#opts.strictHallucinationReduction ?? defaultSTTOptions.strictHallucinationReduction,
    };

    if (this.#opts.vadOnset !== null && this.#opts.vadOnset !== undefined) {
      payload.vad_onset = this.#opts.vadOnset;
    }
    if (this.#opts.vadOffset !== null && this.#opts.vadOffset !== undefined) {
      payload.vad_offset = this.#opts.vadOffset;
    }
    if (this.#opts.initialPrompt !== null && this.#opts.initialPrompt !== undefined) {
      payload.initial_prompt = this.#opts.initialPrompt;
    }
    if (this.#opts.hotwords !== null && this.#opts.hotwords !== undefined) {
      payload.hotwords = this.#opts.hotwords;
    }
    if (
      this.#opts.compressionRatioThreshold !== null &&
      this.#opts.compressionRatioThreshold !== undefined
    ) {
      payload.compression_ratio_threshold = this.#opts.compressionRatioThreshold;
    }
    if (this.#opts.maxTokens !== null && this.#opts.maxTokens !== undefined) {
      payload.max_tokens = this.#opts.maxTokens;
    }
    if (this.#opts.logProbThreshold !== null && this.#opts.logProbThreshold !== undefined) {
      payload.log_prob_threshold = this.#opts.logProbThreshold;
    }

    const baseUrl = this.#opts.baseUrl ?? SIMPLISMART_BASE_URL;
    const timeout = options?.connOptions?.timeout ?? 30000;

    try {
      const controller = new AbortController();
      if (abortSignal) {
        abortSignal.addEventListener('abort', () => controller.abort());
      }

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
        this.#logger.error(`Simplismart API error: ${response.status} - ${errorText}`);
        throw new Error(`Simplismart API Error: ${errorText}`);
      }

      const responseJson = await response.json();
      const timestamps = responseJson.timestamps || [];
      const transcription = responseJson.transcription || [];
      const info = responseJson.info || {};
      const detectedLanguage = info.language || resolvedLanguage;
      const requestId = responseJson.request_id || '';

      const startTime = timestamps.length > 0 ? timestamps[0][0] : 0.0;
      const endTime = timestamps.length > 0 ? timestamps[timestamps.length - 1][1] : 0.0;
      const text = transcription.join('');

      return {
        type: stt.SpeechEventType.FINAL_TRANSCRIPT,
        requestId,
        alternatives: [
          {
            language: detectedLanguage,
            text,
            startTime,
            endTime,
          },
        ],
      };
    } catch (error: any) {
      if (error.name === 'AbortError') {
        throw new Error('Simplismart API request timed out');
      }
      if (error instanceof Error) {
        this.#logger.error(`Error during Simplismart STT processing: ${error.message}`);
        throw error;
      }
      this.#logger.error(`Unexpected error in Simplismart STT: ${error}`);
      throw new Error(`Unexpected error in Simplismart STT: ${error}`);
    }
  }

  stream(options?: { language?: string; connOptions?: APIConnectOptions }): SpeechStream {
    if (!this.#opts.streaming) {
      throw new Error('Streaming is not enabled. Set streaming: true in constructor options.');
    }

    const optsLanguage = options?.language ?? this.#opts.language ?? 'en';
    const streamOpts: STTOptions = {
      ...this.#opts,
      language: optsLanguage,
    };

    return new SpeechStream(this, streamOpts, options?.connOptions);
  }

  async close() {
    this.abortController.abort();
  }
}

export class SpeechStream extends stt.SpeechStream {
  #opts: STTOptions;
  #logger = log();
  #requestId: string;
  #wsUrl: string;
  #apiKey: string;
  label = 'simplismart.SpeechStream';

  private static readonly CHUNK_DURATION_MS = 50;
  private static readonly SAMPLE_RATE = 16000;

  constructor(stt: STT, opts: STTOptions, connOptions?: APIConnectOptions) {
    super(stt, SpeechStream.SAMPLE_RATE, connOptions);
    this.#opts = opts;
    this.#requestId = String(Date.now());
    this.#wsUrl = SIMPLISMART_WS_URL;
    this.#apiKey = opts.apiKey!;
  }

  protected async run() {
    let ws: WebSocket | null = null;

    while (!this.input.closed && !this.closed) {
      try {
        ws = await this.#connectWS();
        await this.#sendInitialConfig(ws);
        await this.#runWS(ws);
      } catch (error: any) {
        if (!this.closed && !this.input.closed) {
          this.#logger.error(`Simplismart WebSocket error: ${error.message}`);
          // Reconnect logic could be added here if needed
          await new Promise((resolve) => setTimeout(resolve, 1000));
        } else {
          break;
        }
      } finally {
        if (ws) {
          ws.close();
        }
      }
    }

    this.closed = true;
  }

  async #connectWS(): Promise<WebSocket> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.#wsUrl, {
        headers: { Authorization: `Bearer ${this.#apiKey}` },
      });

      const timeout = setTimeout(() => {
        ws.close();
        reject(new Error('WebSocket connection timeout'));
      }, this._connOptions?.timeout ?? 30000);

      ws.on('open', () => {
        clearTimeout(timeout);
        resolve(ws);
      });

      ws.on('error', (error) => {
        clearTimeout(timeout);
        reject(error);
      });
    });
  }

  async #sendInitialConfig(ws: WebSocket): Promise<void> {
    try {
      const configMessage = { language: this.#opts.language ?? 'en' };
      ws.send(JSON.stringify(configMessage));
      this.#logger.debug('Sent initial config for Simplismart model', {
        request_id: this.#requestId,
        language: this.#opts.language,
      });
    } catch (error: any) {
      this.#logger.error(`Failed to send initial configuration: ${error.message}`, {
        request_id: this.#requestId,
      });
      throw new Error(`Failed to send initial config: ${error.message}`);
    }
  }

  async #runWS(ws: WebSocket): Promise<void> {
    const sendTask = async () => {
      const samples50ms = Math.floor(SpeechStream.SAMPLE_RATE / 20);
      const stream = new AudioByteStream(SpeechStream.SAMPLE_RATE, 1, samples50ms);

      const abortPromise = waitForAbort(this.abortSignal);

      try {
        while (!this.closed) {
          const result = await Promise.race([this.input.next(), abortPromise]);

          if (result === undefined) return; // aborted
          if (result.done) {
            break;
          }

          const data = result.value;

          let frames: AudioFrame[];
          if (data === SpeechStream.FLUSH_SENTINEL) {
            frames = stream.flush();
          } else if (data.sampleRate === SpeechStream.SAMPLE_RATE && data.channels === 1) {
            frames = stream.write(data.data.buffer as ArrayBuffer);
          } else {
            throw new Error(`sample rate or channel count of frame does not match`);
          }

          for await (const frame of frames) {
            ws.send(frame.data.buffer);
          }
        }
      } finally {
        ws.close();
      }
    };

    const listenTask = async () => {
      return new Promise<void>((resolve, reject) => {
        ws.on('message', (msg: Buffer) => {
          try {
            const transcriptText = msg.toString('utf-8');
            this.#handleTranscriptData(transcriptText);
          } catch (error: any) {
            this.#logger.error(`Failed to process Simplismart message: ${error.message}`);
            reject(error);
          }
        });

        ws.on('close', () => {
          resolve();
        });

        ws.on('error', (error) => {
          reject(error);
        });
      });
    };

    await Promise.all([sendTask(), listenTask()]);
  }

  #handleTranscriptData(data: string): void {
    const transcriptText = data;
    const requestId = this.#requestId;

    try {
      // Create usage event
      const usageEvent: stt.SpeechEvent = {
        type: stt.SpeechEventType.RECOGNITION_USAGE,
        requestId: JSON.stringify({
          original_id: requestId,
          processing_latency: 0.0,
        }),
        recognitionUsage: {
          audioDuration: 0.0,
        },
      };
      this.queue.put(usageEvent);

      // Create speech data
      const speechData: stt.SpeechData = {
        language: this.#opts.language ?? 'en',
        text: transcriptText,
      };

      // Create final transcript event
      const speechEvent: stt.SpeechEvent = {
        type: stt.SpeechEventType.FINAL_TRANSCRIPT,
        requestId,
        alternatives: [speechData],
      };
      this.queue.put(speechEvent);

      this.#logger.debug('Transcript processed successfully', {
        request_id: this.#requestId,
        text_length: transcriptText.length,
        language: this.#opts.language,
      });
    } catch (error: any) {
      this.#logger.error(`Error processing transcript data: ${error.message}`, {
        request_id: this.#requestId,
        transcript_text: transcriptText,
      });
      throw error;
    }
  }
}
