// SPDX-FileCopyrightText: 2025 LiveKit, Inc.
//
// SPDX-License-Identifier: Apache-2.0
import { Plugin } from '@livekit/agents';

export * from './stt.js';
export * from './tts.js';
export * from './models.js';

class SimplismartPlugin extends Plugin {
  constructor() {
    super({
      title: 'simplismart',
      version: '0.1.0',
      package: '@livekit/agents-plugin-simplismart',
    });
  }
}

Plugin.registerPlugin(new SimplismartPlugin());
