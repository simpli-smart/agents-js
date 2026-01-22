// SPDX-FileCopyrightText: 2025 LiveKit, Inc.
//
// SPDX-License-Identifier: Apache-2.0
import {
  type JobContext,
  type JobProcess,
  WorkerOptions,
  cli,
  defineAgent,
  voice,
} from '@livekit/agents';
import * as openai from '@livekit/agents-plugin-openai';
import * as silero from '@livekit/agents-plugin-silero';
import * as simplismart from '@livekit/agents-plugin-simplismart';
import { fileURLToPath } from 'node:url';

const SIMPLISMART_API_KEY = process.env.SIMPLISMART_API_KEY;

if (!SIMPLISMART_API_KEY) {
  throw new Error('SIMPLISMART_API_KEY environment variable is required');
}

export default defineAgent({
  prewarm: async (proc: JobProcess) => {
    proc.userData.vad = await silero.VAD.load();
  },
  entry: async (ctx: JobContext) => {
    await ctx.connect();

    const agent = new voice.Agent({
      instructions:
        'you have to reply in english. do not use emojis, asterisks, markdown, or other special characters in your responses.',
    });

    const session = new voice.AgentSession({
      vad: ctx.proc.userData.vad! as silero.VAD,
      stt: new simplismart.STT({
        baseUrl: 'http://http.wn2zm9klni.ss-in.s9t.link/predict',
        streaming: false,
        apiKey: SIMPLISMART_API_KEY,
        language: 'en',
      }),
      llm: new openai.LLM({
        model: 'google/gemma-3-4b-it',
        apiKey: SIMPLISMART_API_KEY,
        baseURL: 'https://api.simplismart.live',
      }),
      tts: new simplismart.TTS({
        baseUrl: 'https://api.simplismart.live/tts',
        apiKey: SIMPLISMART_API_KEY,
        model: 'canopylabs/orpheus-3b-0.1-ft',
      }),
    });

    await session.start({
      agent,
      room: ctx.room,
    });
  },
});

cli.runApp(new WorkerOptions({ agent: fileURLToPath(import.meta.url) }));
