// Copyright (c) 2022, Compiler Explorer Authors
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

import {AnnotatedCfgDescriptor, AnnotatedNodeDescriptor} from '../types/compilation/cfg.interfaces.js';

import IntervalTree from '@flatten-js/interval-tree';

// Much of the algorithm is inspired from
// https://cutter.re/docs/api/widgets/classGraphGridLayout.html
// Thanks to the cutter team for their great documentation!

// TODO(jeremy-rifkin)
function assert(condition: boolean, message?: string, ...args: any[]): asserts condition {
    if (!condition) {
        const stack = new Error('Assertion Error').stack;
        throw (
            (message
                ? `Assertion error in llvm-print-after-all-parser: ${message}`
                : `Assertion error in llvm-print-after-all-parser`) +
            (args.length > 0 ? `\n${JSON.stringify(args)}\n` : '') +
            `\n${stack}`
        );
    }
}

enum SegmentType {
    Horizontal,
    Vertical,
}

type Coordinate = {
    x: number;
    y: number;
};

type GridCoordinate = {
    row: number;
    col: number;
};

type EdgeCoordinate = Coordinate & GridCoordinate;

type EdgeSegment = {
    start: EdgeCoordinate;
    end: EdgeCoordinate;
    horizontalOffset: number;
    verticalOffset: number;
    type: SegmentType; // is this point the end of a horizontal or vertical segment
};

type Edge = {
    color: string;
    dest: number;
    mainColumn: number;
    path: EdgeSegment[];
};

type BoundingBox = {
    rows: number;
    cols: number;
};

type Block = {
    data: AnnotatedNodeDescriptor;
    edges: Edge[];
    dagEdges: number[];
    treeEdges: number[];
    treeParent: number | null;
    row: number;
    col: number;
    boundingBox: BoundingBox;
    coordinates: Coordinate;
    incidentEdgeCount: number;
};

enum DfsState {
    NotVisited,
    Pending,
    Visited,
}

type ColumnDescriptor = {
    width: number;
    totalOffset: number;
};
type RowDescriptor = {
    height: number;
    totalOffset: number;
};
type EdgeColumnMetadata = {
    subcolumns: number;
    intervals: IntervalTree<EdgeSegment>[]; // pointers to segments
};
type EdgeRowMetadata = {
    subrows: number;
    intervals: IntervalTree<EdgeSegment>[]; // pointers to segments
};

const EDGE_SPACING = 10;

export class GraphLayoutCore {
    // We use an adjacency list here
    blocks: Block[] = [];
    columnCount: number;
    rowCount: number;
    blockColumns: ColumnDescriptor[];
    blockRows: RowDescriptor[];
    edgeColumns: (ColumnDescriptor & EdgeColumnMetadata)[];
    edgeRows: (RowDescriptor & EdgeRowMetadata)[];
    readonly layoutTime: number;

    constructor(cfg: AnnotatedCfgDescriptor) {
        // block id -> block
        const blockMap: Record<string, number> = {};
        for (const node of cfg.nodes) {
            const block = {
                data: node,
                edges: [],
                dagEdges: [],
                treeEdges: [],
                treeParent: null,
                row: 0,
                col: 0,
                boundingBox: {rows: 0, cols: 0},
                coordinates: {x: 0, y: 0},
                incidentEdgeCount: 0,
            };
            this.blocks.push(block);
            blockMap[node.id] = this.blocks.length - 1;
        }
        for (const {from, to, color} of cfg.edges) {
            // TODO: Backend can return dest: "null"
            // e.g. for the simple program
            // void baz(int n) {
            //     if(n % 2 == 0) {
            //         foo();
            //     } else {
            //         bar();
            //     }
            // }
            if (from in blockMap && to in blockMap) {
                this.blocks[blockMap[from]].edges.push({
                    color,
                    dest: blockMap[to],
                    mainColumn: -1,
                    path: [],
                });
            }
        }
        //console.log(this.blocks);
        const start = performance.now();
        this.layout();
        const end = performance.now();
        this.layoutTime = end - start;
    }

    dfs(visited: DfsState[], order: number[], node: number) {
        if (visited[node] === DfsState.Visited) {
            return;
        }
        if (visited[node] === DfsState.NotVisited) {
            visited[node] = DfsState.Pending;
            const block = this.blocks[node];
            for (const edge of block.edges) {
                this.blocks[edge.dest].incidentEdgeCount++;
                // If we reach another pending node it's a loop edge.
                // If we reach an unvisited node it's fine, if we reach a visited node that's also part of the dag
                if (visited[edge.dest] !== DfsState.Pending) {
                    block.dagEdges.push(edge.dest);
                }
                this.dfs(visited, order, edge.dest);
            }
            visited[node] = DfsState.Visited;
            order.push(node);
        } else {
            // visited[node] == DfsState.Pending
            // If we reach a node in the stack then this is a loop edge; we do nothing
        }
    }

    computeDag() {
        // Returns a topological order of blocks
        // Breaks loop edges with DFS
        // Can consider doing non-recursive dfs later if needed
        const visited = Array(this.blocks.length).fill(DfsState.NotVisited);
        const order: number[] = [];
        // TODO: Need an actual function entry point from the backend, or will it always be at index 0?
        this.dfs(visited, order, 0);
        for (let i = 0; i < this.blocks.length; i++) {
            this.dfs(visited, order, i);
        }
        // we've computed a post-DFS ordering which is always a reverse topological ordering
        return order.reverse();
    }

    assignRows(topologicalOrder) {
        for (const i of topologicalOrder) {
            const block = this.blocks[i];
            //console.log(block);
            for (const j of block.dagEdges) {
                const target = this.blocks[j];
                target.row = Math.max(target.row, block.row + 1);
            }
        }
    }

    computeTree(topologicalOrder) {
        // DAG is reduced to a tree based on what's vertically adjacent
        //
        // For something like
        //
        //             +-----+
        //             |  A  |
        //             +-----+
        //            /       \
        //     +-----+         +-----+
        //     |  B  |         |  C  |
        //     +-----+         +-----+
        //            \       /
        //             +-----+
        //             |  D  |
        //             +-----+
        //
        // The tree is chosen to be either of the following depending on what the topological order happens to be
        // This doesn't matter too much as far as readability goes
        //
        //      A            A
        //     / \          / \
        //    B   C   or   B   C
        //    |                |
        //    D                D
        for (const i of topologicalOrder) {
            // Only dag edges are considered
            // Edges - dag edges = the set of back edges
            const block = this.blocks[i];
            for (const j of block.dagEdges) {
                const target = this.blocks[j];
                if (target.treeParent === null && target.row === block.row + 1) {
                    block.treeEdges.push(j);
                    target.treeParent = i;
                }
            }
        }
    }

    adjustSubtree(root: number, rowShift: number, columnShift: number) {
        const block = this.blocks[root];
        block.row += rowShift;
        block.col += columnShift;
        for (const j of block.treeEdges) {
            this.adjustSubtree(j, rowShift, columnShift);
        }
    }

    // Note: Currently O(n^2)
    computeTreeColumnPositions(node: number) {
        const block = this.blocks[node];
        if (block.treeEdges.length === 0) {
            block.row = 0;
            block.col = 0;
            block.boundingBox = {
                rows: 1,
                cols: 2,
            };
        } else if (block.treeEdges.length === 1) {
            const childIndex = block.treeEdges[0];
            const child = this.blocks[childIndex];
            block.row = 0;
            block.col = child.col;
            block.boundingBox = {
                rows: 1 + child.boundingBox.rows,
                cols: child.boundingBox.cols,
            };
            this.adjustSubtree(childIndex, 1, 0);
        } else {
            // If the node has more than two children we'll just center between the
            //let selectedTreeEdges = block.treeEdges.slice(0, 2);
            const boundingBox = {
                rows: 0,
                cols: 0,
            };
            // Compute bounding box of all the subtrees and adjust
            for (const i of block.treeEdges) {
                const child = this.blocks[i];
                this.adjustSubtree(i, 1, boundingBox.cols);
                boundingBox.rows += child.boundingBox.rows;
                boundingBox.cols += child.boundingBox.cols;
            }
            // Position parent
            boundingBox.rows++;
            block.boundingBox = boundingBox;
            block.row = 0;
            // between immediate children
            const [left, right] = [this.blocks[block.treeEdges[0]], this.blocks[block.treeEdges[1]]];
            block.col = Math.floor((left.col + right.col) / 2); // TODO
        }
    }

    assignColumns(topologicalOrder) {
        // Note: Currently not taking shape into account like Cutter does.
        // Post DFS order means we compute all children before their parents
        for (const i of topologicalOrder.slice().reverse()) {
            this.computeTreeColumnPositions(i);
        }
        // We have a forrest, CFGs can have multiple source nodes
        const trees = Array.from(this.blocks.entries()).filter(([_, block]) => block.treeParent === null);
        // Place trees next to each other
        let offset = 0;
        for (const [i, tree] of trees) {
            this.adjustSubtree(i, 0, offset);
            offset += tree.boundingBox.cols;
        }
    }

    setupRowsAndColumns() {
        //console.log(this.blocks);
        this.rowCount = Math.max(...this.blocks.map(block => block.row)) + 1; // one more row for zero offset
        this.columnCount = Math.max(...this.blocks.map(block => block.col)) + 2; // blocks are two-wide
        this.blockRows = Array(this.rowCount)
            .fill(0)
            .map(() => ({
                height: 0,
                totalOffset: 0,
            }));
        this.blockColumns = Array(this.columnCount)
            .fill(0)
            .map(() => ({
                width: 0,
                totalOffset: 0,
            }));
        this.edgeRows = Array(this.rowCount + 1)
            .fill(0)
            .map(() => ({
                height: 2 * EDGE_SPACING,
                totalOffset: 0,
                subrows: 0,
                intervals: [],
            }));
        this.edgeColumns = Array(this.columnCount + 1)
            .fill(0)
            .map(() => ({
                width: 2 * EDGE_SPACING,
                totalOffset: 0,
                subcolumns: 0,
                intervals: [],
            }));
    }

    computeEdgeMainColumns() {
        // This is heavily inspired by Cutter
        // We use a sweep line algorithm processing the CFG top to bottom keeping track of when columns are most
        // recently blocked. Cutter uses an augmented binary tree to assist with finding empty columns, for now this
        // just naively iterates.
        enum EventType {
            Edge = 0,
            Block = 1,
        }
        type Event = {
            blockIndex: number;
            edgeIndex: number;
            row: number;
            type: EventType;
        };
        const events: Event[] = [];
        for (const [i, block] of this.blocks.entries()) {
            events.push({
                blockIndex: i,
                edgeIndex: -1,
                row: block.row,
                type: EventType.Block,
            });
            for (const [j, edge] of block.edges.entries()) {
                events.push({
                    blockIndex: i,
                    edgeIndex: j,
                    row: Math.max(block.row + 1, this.blocks[edge.dest].row),
                    type: EventType.Edge,
                });
            }
        }
        // Sort by row (max(src row, target row) for edges), edge row n is before block row n
        events.sort((a: Event, b: Event) => {
            if (a.row === b.row) {
                return a.type - b.type;
            } else {
                return a.row - b.row;
            }
        });
        //
        const blockedColumns = Array(this.columnCount + 1).fill(-1);
        for (const event of events) {
            if (event.type === EventType.Block) {
                const block = this.blocks[event.blockIndex];
                blockedColumns[block.col + 1] = block.row;
            } else {
                const source = this.blocks[event.blockIndex];
                const edge = source.edges[event.edgeIndex];
                const target = this.blocks[edge.dest];
                const sourceColumn = source.col + 1;
                const targetColumn = target.col + 1;
                const topRow = Math.min(source.row + 1, target.row);
                if (blockedColumns[sourceColumn] < topRow) {
                    // use column under source block
                    edge.mainColumn = sourceColumn;
                } else if (blockedColumns[targetColumn] < topRow) {
                    // use column of the target
                    edge.mainColumn = targetColumn;
                } else {
                    const leftCandidate =
                        sourceColumn -
                        1 -
                        blockedColumns
                            .slice(0, sourceColumn)
                            .reverse()
                            .findIndex(v => v < topRow);
                    const rightCandidate = sourceColumn + blockedColumns.slice(sourceColumn).findIndex(v => v < topRow);
                    // hamming distance
                    const distanceLeft =
                        Math.abs(sourceColumn - leftCandidate) + Math.abs(targetColumn - leftCandidate);
                    const distanceRight =
                        Math.abs(sourceColumn - rightCandidate) + Math.abs(targetColumn - rightCandidate);
                    // "figure 8" logic from cutter
                    // Takes a longer path that produces less crossing
                    if (target.row < source.row) {
                        if (
                            targetColumn < sourceColumn &&
                            blockedColumns[sourceColumn + 1] < topRow &&
                            sourceColumn - targetColumn <= distanceLeft + 2
                        ) {
                            edge.mainColumn = sourceColumn + 1;
                            continue;
                        } else if (
                            targetColumn > sourceColumn &&
                            blockedColumns[sourceColumn - 1] < topRow &&
                            targetColumn - sourceColumn <= distanceRight + 2
                        ) {
                            edge.mainColumn = sourceColumn - 1;
                            continue;
                        }
                    }
                    if (distanceLeft === distanceRight) {
                        // TODO: Could also try this
                        /*if(target.row <= source.row) {
                            if(leftCandidate === sourceColumn - 1) {
                                edge.mainColumn = leftCandidate;
                                continue;
                            } else if(rightCandidate === sourceColumn + 1) {
                                edge.mainColumn = rightCandidate;
                                continue;
                            }
                        }*/
                        // Place true branches on the left
                        // TODO: Need to investigate further block placement stuff here
                        // TODO: Need to investigate further offset placement stuff for the start segments
                        if (edge.color === 'green') {
                            edge.mainColumn = leftCandidate;
                        } else {
                            edge.mainColumn = rightCandidate;
                        }
                    } else if (distanceLeft < distanceRight) {
                        edge.mainColumn = leftCandidate;
                    } else {
                        edge.mainColumn = rightCandidate;
                    }
                }
            }
        }
    }

    // eslint-disable-next-line max-statements
    addEdgePaths() {
        // (start: GridCoordinate, end: GridCoordinate) => ({
        const makeSegment = (start: [number, number], end: [number, number]): EdgeSegment => ({
            start: {
                //...start,
                row: start[0],
                col: start[1],
                x: 0,
                y: 0,
            },
            end: {
                //...end,
                row: end[0],
                col: end[1],
                x: 0,
                y: 0,
            },
            horizontalOffset: 0,
            verticalOffset: 0,
            type: start[1] === end[1] ? SegmentType.Vertical : SegmentType.Horizontal,
        });
        for (const block of this.blocks) {
            for (const edge of block.edges) {
                const target = this.blocks[edge.dest];
                // start just below the source block
                edge.path.push(makeSegment([block.row + 1, block.col + 1], [block.row + 1, block.col + 1]));
                // horizontal segment over to main column
                edge.path.push(makeSegment([block.row + 1, block.col + 1], [block.row + 1, edge.mainColumn]));
                // vertical segment down the main column
                edge.path.push(makeSegment([block.row + 1, edge.mainColumn], [target.row, edge.mainColumn]));
                // horizontal segment over to the target column
                edge.path.push(makeSegment([target.row, edge.mainColumn], [target.row, target.col + 1]));
                // finish at the target block
                edge.path.push(makeSegment([target.row, target.col + 1], [target.row, target.col + 1]));
                // Simplify segments
                // Simplifications performed are eliminating (non-sentinel) edges which don't move anywhere and folding
                // VV -> V and HH -> H.
                let movement;
                do {
                    movement = false;
                    // i needs to start one into the range since we compare with i - 1
                    for (let i = 1; i < edge.path.length; i++) {
                        const prevSegment = edge.path[i - 1];
                        const segment = edge.path[i];
                        // sanity checks
                        for (let j = 0; j < edge.path.length; j++) {
                            const segment = edge.path[j];
                            if (
                                (segment.type === SegmentType.Vertical && segment.start.col !== segment.end.col) ||
                                (segment.type === SegmentType.Horizontal && segment.start.row !== segment.end.row)
                            ) {
                                throw Error("Segment type doesn't match coordinates");
                            }
                            if (j > 0) {
                                const prev = edge.path[j - 1];
                                if (prev.end.row !== segment.start.row || prev.end.col !== segment.start.col) {
                                    throw Error("Adjacent segment start/endpoints don't match");
                                }
                            }
                            if (j < edge.path.length - 1) {
                                const next = edge.path[j + 1];
                                if (segment.end.row !== next.start.row || segment.end.col !== next.start.col) {
                                    throw Error("Adjacent segment start/endpoints don't match");
                                }
                            }
                        }
                        // If a segment doesn't go anywhere and is not a sentinel it can be eliminated
                        if (
                            segment.start.col === segment.end.col &&
                            segment.start.row === segment.end.row &&
                            i !== edge.path.length - 1
                        ) {
                            edge.path.splice(i, 1);
                            movement = true;
                            continue;
                        }
                        // VV -> V
                        // HH -> H
                        if (prevSegment.type === segment.type) {
                            if (
                                (prevSegment.type === SegmentType.Vertical &&
                                    prevSegment.start.col !== segment.start.col) ||
                                (prevSegment.type === SegmentType.Horizontal &&
                                    prevSegment.start.row !== segment.start.row)
                            ) {
                                throw Error(
                                    "Adjacent horizontal or vertical segments don't share a common row or column",
                                );
                            }
                            prevSegment.end = segment.end;
                            edge.path.splice(i, 1);
                            movement = true;
                            continue;
                        }
                    }
                } while (movement);
                // sanity checks
                for (let j = 0; j < edge.path.length; j++) {
                    const segment = edge.path[j];
                    if (
                        (segment.type === SegmentType.Vertical && segment.start.col !== segment.end.col) ||
                        (segment.type === SegmentType.Horizontal && segment.start.row !== segment.end.row)
                    ) {
                        throw Error("Segment type doesn't match coordinates (post-simplification)");
                    }
                    if (j > 0) {
                        const prev = edge.path[j - 1];
                        if (prev.end.row !== segment.start.row || prev.end.col !== segment.start.col) {
                            throw Error("Adjacent segment start/endpoints don't match (post-simplification)");
                        }
                    }
                    if (j < edge.path.length - 1) {
                        const next = edge.path[j + 1];
                        if (segment.end.row !== next.start.row || segment.end.col !== next.start.col) {
                            throw Error("Adjacent segment start/endpoints don't match (post-simplification)");
                        }
                    }
                }
                // Compute subrows/subcolumns
                for (const segment of edge.path) {
                    if (segment.type === SegmentType.Vertical) {
                        if (segment.start.col !== segment.end.col) {
                            throw Error('Vertical segment changes column');
                        }
                        const col = this.edgeColumns[segment.start.col];
                        let inserted = false;
                        for (const tree of col.intervals) {
                            if (!tree.intersect_any([segment.start.row, segment.end.row])) {
                                tree.insert([segment.start.row, segment.end.row], segment);
                                inserted = true;
                                break;
                            }
                        }
                        if (!inserted) {
                            const tree = new IntervalTree<EdgeSegment>();
                            col.intervals.push(tree);
                            col.subcolumns++;
                            tree.insert([segment.start.row, segment.end.row], segment);
                        }
                    } else {
                        // horizontal
                        if (segment.start.row !== segment.end.row) {
                            throw Error('Horizontal segment changes row');
                        }
                        const row = this.edgeRows[segment.start.row];
                        let inserted = false;
                        for (const tree of row.intervals) {
                            if (!tree.intersect_any([segment.start.col, segment.end.col])) {
                                tree.insert([segment.start.col, segment.end.col], segment);
                                inserted = true;
                                break;
                            }
                        }
                        if (!inserted) {
                            const tree = new IntervalTree<EdgeSegment>();
                            row.intervals.push(tree);
                            row.subrows++;
                            tree.insert([segment.start.col, segment.end.col], segment);
                        }
                    }
                }
            }
        }
        // Throw everything away and do it all again, but smarter
        for (const edgeColumn of this.edgeColumns) {
            for (const intervalTree of edgeColumn.intervals) {
                intervalTree.clear();
            }
        }
        for (const edgeRow of this.edgeRows) {
            for (const intervalTree of edgeRow.intervals) {
                intervalTree.clear();
            }
        }
        // Edge kind is the primary heuristic for subrow/column assignment
        // For horizontal edges, think of left/vertical/right terminology rotated 90 degrees right
        enum EdgeKind {
            LEFTU = -2,
            LEFTCORNER = -1,
            VERTICAL = 0,
            RIGHTCORNER = 1,
            RIGHTU = 2,
            NULL = NaN,
        }
        const segments: {
            segment: EdgeSegment;
            length: number;
            kind: EdgeKind;
            tiebreaker: number;
        }[] = [];
        for (const block of this.blocks) {
            for (const edge of block.edges) {
                const edgeLength = edge.path
                    .map(({start, end}) => Math.abs(start.col - end.col) + Math.abs(start.row - end.row))
                    .reduce((A, x) => A + x);
                const target = this.blocks[edge.dest];
                for (const [i, segment] of edge.path.entries()) {
                    let kind = EdgeKind.NULL;
                    if (i === 0) {
                        if (edge.path.length === 1) {
                            // Segment will be vertical
                            kind = EdgeKind.VERTICAL;
                        } else {
                            // There will be a next
                            const next = edge.path[i + 1];
                            if (next.end.col > segment.end.col) {
                                kind = EdgeKind.RIGHTCORNER;
                            } else {
                                kind = EdgeKind.LEFTCORNER;
                            }
                        }
                    } else if (i === edge.path.length - 1) {
                        // There will be a previous segment, i !== 0, but no next
                        const previous = edge.path[i - 1];
                        if (previous.start.col > segment.end.col) {
                            kind = EdgeKind.RIGHTCORNER;
                        } else {
                            kind = EdgeKind.LEFTCORNER;
                        }
                    } else {
                        // There will be both a previous and a next
                        const next = edge.path[i + 1];
                        const previous = edge.path[i - 1];
                        if (segment.type === SegmentType.Vertical) {
                            if (previous.start.col < segment.start.col && next.end.col < segment.start.col) {
                                kind = EdgeKind.LEFTU;
                            } else if (previous.start.col > segment.start.col && next.end.col > segment.start.col) {
                                kind = EdgeKind.RIGHTU;
                            } else if (previous.start.col > segment.end.col) {
                                kind = EdgeKind.RIGHTCORNER;
                            } else {
                                assert(previous.start.col < segment.end.col);
                                kind = EdgeKind.LEFTCORNER;
                            }
                        } else {
                            // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
                            assert(segment.type === SegmentType.Horizontal);
                            // Same logic, think rotated 90 degrees right
                            if (previous.start.row <= segment.start.row && next.end.row < segment.start.row) {
                                kind = EdgeKind.LEFTU;
                            } else if (previous.start.row > segment.start.row && next.end.row > segment.start.row) {
                                kind = EdgeKind.RIGHTU;
                            } else if (previous.start.row > segment.end.row) {
                                kind = EdgeKind.RIGHTCORNER;
                            } else {
                                kind = EdgeKind.LEFTCORNER;
                            }
                        }
                    }
                    assert((kind as any) !== EdgeKind.NULL);
                    segments.push({
                        segment,
                        kind,
                        length:
                            Math.abs(segment.start.col - segment.end.col) +
                            Math.abs(segment.start.row - segment.end.row),
                        tiebreaker: 2 * edgeLength + (target.row >= block.row ? 1 : 0),
                    });
                }
            }
        }
        segments.sort((a, b) => {
            if (a.kind !== b.kind) {
                return a.kind - b.kind;
            } else {
                const kind = a.kind; // a.kind == b.kind
                if (a.length !== b.length) {
                    if (kind <= 0) {
                        // shortest first if coming from the left
                        return a.length - b.length;
                    } else {
                        // coming from the right, shortest last
                        // reverse edge length order
                        return b.length - a.length;
                    }
                } else {
                    if (kind <= 0) {
                        return a.tiebreaker - b.tiebreaker;
                    } else {
                        // coming from the right, reverse
                        return b.tiebreaker - a.tiebreaker;
                    }
                }
            }
        });
        for (const segmentEntry of segments) {
            const {segment} = segmentEntry;
            if (segment.type === SegmentType.Vertical) {
                const col = this.edgeColumns[segment.start.col];
                let inserted = false;
                for (const tree of col.intervals) {
                    if (!tree.intersect_any([segment.start.row, segment.end.row])) {
                        tree.insert([segment.start.row, segment.end.row], segment);
                        inserted = true;
                        break;
                    }
                }
                if (!inserted) {
                    throw Error("Vertical segment couldn't be inserted");
                }
            } else {
                // Horizontal
                const row = this.edgeRows[segment.start.row];
                let inserted = false;
                for (const tree of row.intervals) {
                    if (!tree.intersect_any([segment.start.col, segment.end.col])) {
                        tree.insert([segment.start.col, segment.end.col], segment);
                        inserted = true;
                        break;
                    }
                }
                if (!inserted) {
                    throw Error("Horizontal segment couldn't be inserted");
                }
            }
        }
        // Assign offsets
        for (const edgeColumn of this.edgeColumns) {
            edgeColumn.width = Math.max(EDGE_SPACING + edgeColumn.intervals.length * EDGE_SPACING, 2 * EDGE_SPACING);
            for (const [i, intervalTree] of edgeColumn.intervals.entries()) {
                for (const segment of intervalTree.values) {
                    segment.horizontalOffset = EDGE_SPACING * (i + 1);
                }
            }
        }
        for (const edgeRow of this.edgeRows) {
            edgeRow.height = Math.max(EDGE_SPACING + edgeRow.intervals.length * EDGE_SPACING, 2 * EDGE_SPACING);
            for (const [i, intervalTree] of edgeRow.intervals.entries()) {
                for (const segment of intervalTree.values) {
                    segment.verticalOffset = EDGE_SPACING * (i + 1);
                }
            }
        }
    }

    // eslint-disable-next-line max-statements
    computeCoordinates() {
        // Compute block row widths and heights
        for (const block of this.blocks) {
            // Update block width if it has a ton of incoming edges
            block.data.width = Math.max(block.data.width, (block.incidentEdgeCount - 1) * EDGE_SPACING);
            //console.log(this.blockRows[block.row].height, block.data.height, block.row);
            //console.log(this.blockRows);
            const halfWidth = (block.data.width - this.edgeColumns[block.col + 1].width) / 2;
            //console.log("--->", block.col, this.columnCount);
            this.blockRows[block.row].height = Math.max(this.blockRows[block.row].height, block.data.height);
            this.blockColumns[block.col].width = Math.max(this.blockColumns[block.col].width, halfWidth);
            this.blockColumns[block.col + 1].width = Math.max(this.blockColumns[block.col + 1].width, halfWidth);
        }
        // Compute row total offsets
        for (let i = 0; i < this.rowCount; i++) {
            // edge row 0 is already at the correct offset, this iteration will set the offset for block row 0 and edge
            // row 1.
            this.blockRows[i].totalOffset = this.edgeRows[i].totalOffset + this.edgeRows[i].height;
            this.edgeRows[i + 1].totalOffset = this.blockRows[i].totalOffset + this.blockRows[i].height;
        }
        // Compute column total offsets
        for (let i = 0; i < this.columnCount; i++) {
            // same deal here
            this.blockColumns[i].totalOffset = this.edgeColumns[i].totalOffset + this.edgeColumns[i].width;
            this.edgeColumns[i + 1].totalOffset = this.blockColumns[i].totalOffset + this.blockColumns[i].width;
        }
        // Compute block coordinates and edge paths
        for (const block of this.blocks) {
            block.coordinates.x =
                this.edgeColumns[block.col + 1].totalOffset -
                (block.data.width - this.edgeColumns[block.col + 1].width) / 2;
            block.coordinates.y = this.blockRows[block.row].totalOffset;
            for (const edge of block.edges) {
                if (edge.path.length === 1) {
                    // Special case: Direct dropdown
                    const segment = edge.path[0];
                    const target = this.blocks[edge.dest];
                    segment.start.x = this.edgeColumns[segment.start.col].totalOffset + segment.horizontalOffset;
                    segment.start.y = block.coordinates.y + block.data.height;
                    segment.end.x = this.edgeColumns[segment.end.col].totalOffset + segment.horizontalOffset;
                    segment.end.y = this.edgeRows[target.row].totalOffset + this.edgeRows[target.row].height;
                } else {
                    // push initial point
                    {
                        const segment = edge.path[0];
                        segment.start.x = this.edgeColumns[segment.start.col].totalOffset + segment.horizontalOffset;
                        segment.start.y = block.coordinates.y + block.data.height;
                        segment.end.x = this.edgeColumns[segment.end.col].totalOffset + segment.horizontalOffset;
                        segment.end.y = 0; // this is something we need from the next segment
                    }
                    // first and last handled specially
                    for (const segment of edge.path.slice(1, edge.path.length - 1)) {
                        segment.start.x = this.edgeColumns[segment.start.col].totalOffset + segment.horizontalOffset;
                        segment.start.y = this.edgeRows[segment.start.row].totalOffset + segment.verticalOffset;
                        segment.end.x = this.edgeColumns[segment.end.col].totalOffset + segment.horizontalOffset;
                        segment.end.y = this.edgeRows[segment.end.row].totalOffset + segment.verticalOffset;
                    }
                    // push final point
                    {
                        const target = this.blocks[edge.dest];
                        const segment = edge.path[edge.path.length - 1];
                        segment.start.x = this.edgeColumns[segment.start.col].totalOffset + segment.horizontalOffset;
                        segment.start.y = 0; // something we need from the previous segment
                        segment.end.x = this.edgeColumns[segment.start.col].totalOffset + segment.horizontalOffset;
                        segment.end.y = this.edgeRows[target.row].totalOffset + this.edgeRows[target.row].height;
                    }
                    // apply offsets to neighbor segments
                    for (let i = 0; i < edge.path.length; i++) {
                        const segment = edge.path[i];
                        if (segment.type === SegmentType.Vertical) {
                            if (i > 0) {
                                const prev = edge.path[i - 1];
                                prev.end.x = segment.start.x;
                            }
                            if (i < edge.path.length - 1) {
                                const next = edge.path[i + 1];
                                next.start.x = segment.end.x;
                            }
                        } else {
                            // Horizontal
                            if (i > 0) {
                                const prev = edge.path[i - 1];
                                prev.end.y = segment.start.y;
                            }
                            if (i < edge.path.length - 1) {
                                const next = edge.path[i + 1];
                                next.start.y = segment.end.y;
                            }
                        }
                    }
                }
            }
        }
    }

    layout() {
        const topologicalOrder = this.computeDag();
        //console.log(topologicalOrder);
        this.assignRows(topologicalOrder);
        //console.log(this.blocks);
        this.computeTree(topologicalOrder);
        //console.log(this.blocks);
        this.assignColumns(topologicalOrder);
        //console.log(this.blocks);
        this.setupRowsAndColumns();
        // Edge routing
        this.computeEdgeMainColumns();
        this.addEdgePaths();
        // -- Nothing is pixel aware above this line ---
        // Add pixel coordinates
        this.computeCoordinates();
        //
        ///console.log(this);
    }

    getWidth() {
        const lastCol = this.edgeColumns[this.edgeColumns.length - 1];
        return lastCol.totalOffset + lastCol.width;
    }

    getHeight() {
        const lastRow = this.edgeRows[this.edgeRows.length - 1];
        return lastRow.totalOffset + lastRow.height;
    }
}
