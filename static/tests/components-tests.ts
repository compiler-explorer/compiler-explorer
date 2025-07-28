// Copyright (c) 2025, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import {describe, expect, it} from 'vitest';
import {
    COMPILER_COMPONENT_NAME,
    DIFF_VIEW_COMPONENT_NAME,
    EDITOR_COMPONENT_NAME,
    EXECUTOR_COMPONENT_NAME,
    OPT_VIEW_COMPONENT_NAME,
    OUTPUT_COMPONENT_NAME,
    TOOL_COMPONENT_NAME,
    TREE_COMPONENT_NAME,
} from '../../static/components.interfaces.js';
import {
    createComponentConfig,
    createLayoutItem,
    fromGoldenLayoutConfig,
    toGoldenLayoutConfig,
} from '../../static/components.js';

describe('Components validation', () => {
    describe('fromGoldenLayoutConfig', () => {
        describe('Input validation', () => {
            it('should throw for null input', () => {
                expect(() => fromGoldenLayoutConfig(null as any)).toThrow('Invalid configuration: must be an object');
            });

            it('should throw for undefined input', () => {
                expect(() => fromGoldenLayoutConfig(undefined as any)).toThrow(
                    'Invalid configuration: must be an object',
                );
            });

            it('should throw for non-object input', () => {
                expect(() => fromGoldenLayoutConfig('string' as any)).toThrow(
                    'Invalid configuration: must be an object',
                );
                expect(() => fromGoldenLayoutConfig(123 as any)).toThrow('Invalid configuration: must be an object');
            });

            it('should accept arrays as valid objects', () => {
                // Arrays are objects in JavaScript, so they pass the initial validation
                // The content validation will handle any structure issues later
                expect(() => fromGoldenLayoutConfig([] as any)).not.toThrow();
            });
        });

        describe('Valid configurations', () => {
            it('should accept empty configuration', () => {
                const config = {};
                const result = fromGoldenLayoutConfig(config);
                expect(result).toEqual(config);
            });

            it('should accept configuration with empty content array', () => {
                const config = {content: []};
                const result = fromGoldenLayoutConfig(config);
                expect(result).toEqual(config);
            });

            it('should accept valid compiler component', () => {
                const config = {
                    content: [
                        {
                            type: 'component',
                            componentName: COMPILER_COMPONENT_NAME,
                            componentState: {
                                source: 1,
                                lang: 'c++',
                            },
                        },
                    ],
                };
                const result = fromGoldenLayoutConfig(config);
                expect(result).toEqual(config);
            });

            it('should accept valid editor component', () => {
                const config = {
                    content: [
                        {
                            type: 'component',
                            componentName: EDITOR_COMPONENT_NAME,
                            componentState: {
                                id: 1,
                                lang: 'c++',
                            },
                        },
                    ],
                };
                const result = fromGoldenLayoutConfig(config);
                expect(result).toEqual(config);
            });

            it('should accept valid executor component', () => {
                const config = {
                    content: [
                        {
                            type: 'component',
                            componentName: EXECUTOR_COMPONENT_NAME,
                            componentState: {
                                source: 1,
                                lang: 'c++',
                                compilationPanelShown: true,
                                compilerOutShown: false,
                            },
                        },
                    ],
                };
                const result = fromGoldenLayoutConfig(config);
                expect(result).toEqual(config);
            });

            it('should accept valid output component', () => {
                const config = {
                    content: [
                        {
                            type: 'component',
                            componentName: OUTPUT_COMPONENT_NAME,
                            componentState: {
                                tree: 1,
                                compiler: 2,
                                editor: 3,
                            },
                        },
                    ],
                };
                const result = fromGoldenLayoutConfig(config);
                expect(result).toEqual(config);
            });

            it('should accept valid tool component', () => {
                const config = {
                    content: [
                        {
                            type: 'component',
                            componentName: TOOL_COMPONENT_NAME,
                            componentState: {
                                tree: 1,
                                toolId: 'readelf',
                                id: 2,
                                editorid: 3,
                            },
                        },
                    ],
                };
                const result = fromGoldenLayoutConfig(config);
                expect(result).toEqual(config);
            });

            it('should accept valid layout items', () => {
                const config = {
                    content: [
                        {
                            type: 'row',
                            content: [
                                {
                                    type: 'component',
                                    componentName: EDITOR_COMPONENT_NAME,
                                    componentState: {},
                                },
                                {
                                    type: 'column',
                                    content: [
                                        {
                                            type: 'component',
                                            componentName: COMPILER_COMPONENT_NAME,
                                            componentState: {source: 1, lang: 'c++'},
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                };
                const result = fromGoldenLayoutConfig(config);
                expect(result).toEqual(config);
            });
        });

        describe('Invalid content structure', () => {
            it('should throw for non-array content', () => {
                const config = {content: 'not-array'} as any;
                expect(() => fromGoldenLayoutConfig(config)).toThrow('Configuration content must be an array');
            });

            it('should throw for content with non-object items', () => {
                const config = {content: ['string-item']} as any;
                expect(() => fromGoldenLayoutConfig(config)).toThrow('Invalid item 0: must be an object');
            });

            it('should throw for items missing type', () => {
                const config = {content: [{}]} as any;
                expect(() => fromGoldenLayoutConfig(config)).toThrow("Invalid item 0: missing 'type' property");
            });

            it('should throw for items with unknown type', () => {
                const config = {content: [{type: 'unknown'}]} as any;
                expect(() => fromGoldenLayoutConfig(config)).toThrow("Invalid item 0: unknown type 'unknown'");
            });
        });

        describe('Invalid component configurations', () => {
            it('should throw for component missing componentName', () => {
                const config = {
                    content: [
                        {
                            type: 'component',
                            componentState: {},
                        },
                    ],
                };
                expect(() => fromGoldenLayoutConfig(config)).toThrow(
                    "Invalid item 0: missing 'componentName' property",
                );
            });

            it('should throw for component with non-string componentName', () => {
                const config = {
                    content: [
                        {
                            type: 'component',
                            componentName: 123,
                            componentState: {},
                        },
                    ],
                };
                expect(() => fromGoldenLayoutConfig(config)).toThrow(
                    "Invalid item 0: 'componentName' must be a string",
                );
            });

            it('should throw for invalid compiler state', () => {
                const config = {
                    content: [
                        {
                            type: 'component',
                            componentName: COMPILER_COMPONENT_NAME,
                            componentState: {
                                // Missing required properties
                                invalidProp: true,
                            },
                        },
                    ],
                };
                expect(() => fromGoldenLayoutConfig(config)).toThrow(
                    "Invalid item 0: invalid component state for component 'compiler'",
                );
            });

            it('should throw for invalid executor state missing boolean flags', () => {
                const config = {
                    content: [
                        {
                            type: 'component',
                            componentName: EXECUTOR_COMPONENT_NAME,
                            componentState: {
                                source: 1,
                                lang: 'c++',
                                // Missing compilationPanelShown and compilerOutShown
                            },
                        },
                    ],
                };
                expect(() => fromGoldenLayoutConfig(config)).toThrow(
                    "Invalid item 0: invalid component state for component 'executor'",
                );
            });

            it('should throw for invalid output state missing numeric properties', () => {
                const config = {
                    content: [
                        {
                            type: 'component',
                            componentName: OUTPUT_COMPONENT_NAME,
                            componentState: {
                                tree: 'not-number',
                                compiler: 1,
                                editor: 1,
                            },
                        },
                    ],
                };
                expect(() => fromGoldenLayoutConfig(config)).toThrow(
                    "Invalid item 0: invalid component state for component 'output'",
                );
            });

            it('should throw for invalid tool state', () => {
                const config = {
                    content: [
                        {
                            type: 'component',
                            componentName: TOOL_COMPONENT_NAME,
                            componentState: {
                                tree: 1,
                                // Missing required toolId, id, editorid
                            },
                        },
                    ],
                };
                expect(() => fromGoldenLayoutConfig(config)).toThrow(
                    "Invalid item 0: invalid component state for component 'tool'",
                );
            });

            it('should throw for unknown component name', () => {
                const config = {
                    content: [
                        {
                            type: 'component',
                            componentName: 'unknown-component',
                            componentState: {},
                        },
                    ],
                };
                expect(() => fromGoldenLayoutConfig(config)).toThrow(
                    "Invalid item 0: invalid component state for component 'unknown-component'",
                );
            });
        });

        describe('Invalid layout item configurations', () => {
            it('should throw for layout item missing content', () => {
                const config = {
                    content: [
                        {
                            type: 'row',
                        },
                    ],
                } as any;
                expect(() => fromGoldenLayoutConfig(config)).toThrow(
                    "Invalid item 0: layout items must have a 'content' array",
                );
            });

            it('should throw for layout item with non-array content', () => {
                const config = {
                    content: [
                        {
                            type: 'column',
                            content: 'not-array',
                        },
                    ],
                } as any;
                expect(() => fromGoldenLayoutConfig(config)).toThrow(
                    "Invalid item 0: layout items must have a 'content' array",
                );
            });
        });

        describe('Complex nested validation', () => {
            it('should validate deeply nested structures', () => {
                const config = {
                    content: [
                        {
                            type: 'row',
                            content: [
                                {
                                    type: 'column',
                                    content: [
                                        {
                                            type: 'stack',
                                            content: [
                                                {
                                                    type: 'component',
                                                    componentName: EDITOR_COMPONENT_NAME,
                                                    componentState: {},
                                                },
                                                {
                                                    type: 'component',
                                                    componentName: DIFF_VIEW_COMPONENT_NAME,
                                                    componentState: {},
                                                },
                                            ],
                                        },
                                    ],
                                },
                                {
                                    type: 'component',
                                    componentName: COMPILER_COMPONENT_NAME,
                                    componentState: {source: 1, lang: 'c++'},
                                },
                            ],
                        },
                    ],
                };
                const result = fromGoldenLayoutConfig(config);
                expect(result).toEqual(config);
            });

            it('should throw for invalid nested component', () => {
                const config = {
                    content: [
                        {
                            type: 'row',
                            content: [
                                {
                                    type: 'component',
                                    componentName: EDITOR_COMPONENT_NAME,
                                    componentState: {},
                                },
                                {
                                    type: 'column',
                                    content: [
                                        {
                                            type: 'component',
                                            componentName: COMPILER_COMPONENT_NAME,
                                            componentState: {
                                                // Invalid state - missing required properties
                                                invalidProp: true,
                                            },
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                };
                expect(() => fromGoldenLayoutConfig(config)).toThrow('invalid component state for component');
            });
        });

        describe('Edge cases', () => {
            it('should handle null component state', () => {
                const config = {
                    content: [
                        {
                            type: 'component',
                            componentName: COMPILER_COMPONENT_NAME,
                            componentState: null,
                        },
                    ],
                };
                expect(() => fromGoldenLayoutConfig(config)).toThrow(
                    "Invalid item 0: invalid component state for component 'compiler'",
                );
            });

            it('should preserve additional properties', () => {
                const config = {
                    content: [
                        {
                            type: 'component',
                            componentName: EDITOR_COMPONENT_NAME,
                            componentState: {},
                            title: 'Custom Title',
                            isClosable: false,
                            width: 50,
                        },
                    ],
                    settings: {
                        showPopoutIcon: false,
                    },
                    maximisedItemId: null,
                };
                const result = fromGoldenLayoutConfig(config);
                expect(result).toEqual(config);
            });

            it('should handle mixed valid and invalid compiler states', () => {
                // Test different valid combinations for compiler
                const validConfigs = [
                    {source: 1, lang: 'c++'},
                    {source: 1, compiler: 'gcc'},
                    {tree: 1, lang: 'c++'},
                ];

                for (const state of validConfigs) {
                    const config = {
                        content: [
                            {
                                type: 'component',
                                componentName: COMPILER_COMPONENT_NAME,
                                componentState: state,
                            },
                        ],
                    };
                    expect(() => fromGoldenLayoutConfig(config)).not.toThrow();
                }
            });
        });
    });

    describe('Helper functions', () => {
        describe('createComponentConfig', () => {
            it('should create valid component config', () => {
                const config = createComponentConfig(EDITOR_COMPONENT_NAME, {id: 1, lang: 'c++'});
                expect(config).toEqual({
                    type: 'component',
                    componentName: EDITOR_COMPONENT_NAME,
                    componentState: {id: 1, lang: 'c++'},
                });
            });

            it('should include optional properties', () => {
                const config = createComponentConfig(
                    COMPILER_COMPONENT_NAME,
                    {source: 1, lang: 'c++'},
                    {
                        title: 'Custom Compiler',
                        isClosable: false,
                        width: 50,
                        height: 200,
                    },
                );
                expect(config).toEqual({
                    type: 'component',
                    componentName: COMPILER_COMPONENT_NAME,
                    componentState: {source: 1, lang: 'c++'},
                    title: 'Custom Compiler',
                    isClosable: false,
                    width: 50,
                    height: 200,
                });
            });
        });

        describe('createLayoutItem', () => {
            it('should create valid layout item', () => {
                const content = [
                    createComponentConfig(EDITOR_COMPONENT_NAME, {}),
                    createComponentConfig(COMPILER_COMPONENT_NAME, {source: 1, lang: 'c++'}),
                ];
                const layoutItem = createLayoutItem('row', content);
                expect(layoutItem).toEqual({
                    type: 'row',
                    content,
                });
            });

            it('should include optional properties', () => {
                const content = [createComponentConfig(DIFF_VIEW_COMPONENT_NAME, {})];
                const layoutItem = createLayoutItem('stack', content, {
                    isClosable: true,
                    width: 100,
                    activeItemIndex: 0,
                });
                expect(layoutItem).toEqual({
                    type: 'stack',
                    content,
                    isClosable: true,
                    width: 100,
                    activeItemIndex: 0,
                });
            });
        });

        describe('toGoldenLayoutConfig', () => {
            it('should pass through config unchanged', () => {
                const config = {
                    content: [
                        createLayoutItem('row', [
                            createComponentConfig(EDITOR_COMPONENT_NAME, {id: 1}),
                            createComponentConfig(COMPILER_COMPONENT_NAME, {source: 1, lang: 'c++'}),
                        ]),
                    ],
                    settings: {showPopoutIcon: false},
                };
                const result = toGoldenLayoutConfig(config);
                expect(result).toBe(config);
            });
        });
    });

    describe('Integration tests', () => {
        it('should round-trip through validation', () => {
            const originalConfig = {
                content: [
                    {
                        type: 'row',
                        content: [
                            {
                                type: 'component',
                                componentName: EDITOR_COMPONENT_NAME,
                                componentState: {id: 1, lang: 'c++'},
                            },
                            {
                                type: 'component',
                                componentName: COMPILER_COMPONENT_NAME,
                                componentState: {source: 1, lang: 'c++'},
                            },
                        ],
                    },
                ],
                settings: {showPopoutIcon: false},
            };

            const validated = fromGoldenLayoutConfig(originalConfig);
            const backToGolden = toGoldenLayoutConfig(validated);

            expect(backToGolden).toEqual(originalConfig);
        });

        it('should work with complex real-world config', () => {
            const complexConfig = {
                content: [
                    {
                        type: 'row',
                        content: [
                            {
                                type: 'column',
                                width: 50,
                                content: [
                                    {
                                        type: 'component',
                                        componentName: EDITOR_COMPONENT_NAME,
                                        componentState: {id: 1, lang: 'c++', source: '#include <iostream>'},
                                        height: 60,
                                    },
                                    {
                                        type: 'component',
                                        componentName: TREE_COMPONENT_NAME,
                                        componentState: {id: 1, cmakeArgs: '-DCMAKE_BUILD_TYPE=Release'},
                                        height: 40,
                                    },
                                ],
                            },
                            {
                                type: 'stack',
                                width: 50,
                                content: [
                                    {
                                        type: 'component',
                                        componentName: COMPILER_COMPONENT_NAME,
                                        componentState: {
                                            source: 1,
                                            lang: 'c++',
                                            compiler: 'gcc',
                                            options: '-O2',
                                        },
                                        title: 'GCC',
                                    },
                                    {
                                        type: 'component',
                                        componentName: OPT_VIEW_COMPONENT_NAME,
                                        componentState: {id: 1, source: 'test'},
                                        title: 'Optimization',
                                    },
                                ],
                            },
                        ],
                    },
                ],
                settings: {
                    showPopoutIcon: false,
                    showMaximiseIcon: true,
                    showCloseIcon: true,
                },
                maximisedItemId: null,
            };

            expect(() => fromGoldenLayoutConfig(complexConfig)).not.toThrow();
            const result = fromGoldenLayoutConfig(complexConfig);
            expect(result).toEqual(complexConfig);
        });
    });
});
