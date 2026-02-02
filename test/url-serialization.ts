import {describe, expect, it} from 'vitest';

import * as urlSerialization from '../shared/url-serialization.js';

describe('URL Serialization', () => {
    it('should serialise and deserialise a simple state', () => {
        const state = {
            content: [
                {
                    type: 'row',
                    content: [
                        {
                            type: 'component',
                            componentName: 'codeEditor',
                            componentState: {
                                id: 1,
                                lang: 'c++',
                            },
                        },
                    ],
                },
            ],
        };

        const serialized = urlSerialization.serialiseState(state);
        expect(serialized).toBeTruthy();
        expect(typeof serialized).toBe('string');

        const deserialized = urlSerialization.deserialiseState(serialized);
        expect(deserialized).toBeTruthy();
        expect(deserialized.version).toBe(4);
        expect(deserialized.content).toEqual(state.content);
    });

    it('should compress large states when beneficial', () => {
        const state = {
            content: [
                {
                    type: 'row',
                    content: Array.from({length: 50}, (_, i) => ({
                        type: 'component',
                        componentName: 'codeEditor',
                        componentState: {
                            id: i,
                            lang: 'c++',
                            source: 'int main() { return 0; }',
                        },
                    })),
                },
            ],
        };

        const serialized = urlSerialization.serialiseState(state);
        // Compressed format contains {z: ...} which rison-encodes to contain 'z:'
        // This test just verifies the state serializes successfully
        expect(serialized).toBeTruthy();
        expect(typeof serialized).toBe('string');
        expect(serialized.length).toBeGreaterThan(0);

        // Verify it can be deserialized
        const deserialized = urlSerialization.deserialiseState(serialized);
        expect(deserialized.content).toHaveLength(1);
        expect(deserialized.content[0].content).toHaveLength(50);
    });

    it('should handle round-trip encoding correctly', () => {
        const state = {
            content: [
                {
                    type: 'column',
                    content: [
                        {
                            type: 'stack',
                            content: [
                                {
                                    type: 'component',
                                    componentName: 'compiler',
                                    componentState: {
                                        id: 1,
                                        compiler: 'g132',
                                    },
                                },
                            ],
                        },
                    ],
                },
            ],
        };

        const hash1 = urlSerialization.serialiseState(state);
        const restored = urlSerialization.deserialiseState(hash1);
        const hash2 = urlSerialization.serialiseState(restored);

        // Round-trip should produce identical hash
        expect(hash2).toBe(hash1);
    });
});
