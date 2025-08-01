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

import IntervalTree from '@flatten-js/interval-tree';
import {AnnotatedCfgDescriptor, AnnotatedNodeDescriptor, EdgeColor} from '../types/compilation/cfg.interfaces.js';

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
                : 'Assertion error in llvm-print-after-all-parser') +
            (args.length > 0 ? `\n${JSON.stringify(args)}\n` : '') +
            `\n${stack}`
        );
    }
}

enum SegmentType {
    Horizontal = 0,
    Vertical = 1,
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
    type: SegmentType;
};

type Edge = {
    color: EdgeColor;
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
    NotVisited = 0,
    Pending = 1,
    Visited = 2,
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

enum LayoutEventType { // note: numbering is important for sorting edges first
    Edge = 0,
    Block = 1,
}
type LayoutEvent = {
    blockIndex: number;
    edgeIndex: number;
    row: number;
    type: LayoutEventType;
};

// Edge kind is the primary heuristic for subrow/column assignment
// For horizontal edges, think of left/vertical/right terminology rotated 90 degrees right
enum EdgeKind {
    LEFTU = -2,
    LEFTCORNER = -1,
    VERTICAL = 0,
    RIGHTCORNER = 1,
    RIGHTU = 2,
    // biome-ignore lint/style/useLiteralEnumMembers: ported from cutter
    NULL = Number.NaN,
}

type SegmentInfo = {
    segment: EdgeSegment;
    length: number;
    kind: EdgeKind;
    tiebreaker: number;
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

    constructor(
        cfg: AnnotatedCfgDescriptor,
        readonly centerParents: boolean,
    ) {
        this.populate_graph(cfg);

        const start = performance.now();
        this.layout();
        const end = performance.now();
        this.layoutTime = end - start;
    }

    populate_graph(cfg: AnnotatedCfgDescriptor) {
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
    }

    countEdges() {
        // Count the number of incoming edges for each block, this is used to adjust block widths so arrows don't
        // overflow the sides
        for (const block of this.blocks) {
            for (const edge of block.edges) {
                this.blocks[edge.dest].incidentEdgeCount++;
            }
        }
    }

    static postorderDFS(blocks: Block[], visited: DfsState[], node: number, callback: (node: number) => void) {
        if (visited[node] === DfsState.Visited) {
            return;
        }
        if (visited[node] === DfsState.NotVisited) {
            visited[node] = DfsState.Pending;
            const block = blocks[node];
            for (const edge of block.edges) {
                // If we reach another pending node it's a loop edge.
                // If we reach an unvisited node it's fine, if we reach a visited node that's also part of the dag
                if (visited[edge.dest] !== DfsState.Pending) {
                    block.dagEdges.push(edge.dest);
                }
                this.postorderDFS(blocks, visited, edge.dest, callback);
            }
            visited[node] = DfsState.Visited;
            callback(node);
        } else {
            // If we reach a node in the stack then this is a loop edge; we do nothing
            assert(visited[node] == DfsState.Pending);
        }
    }

    computeDag() {
        // Returns a topological order of blocks
        // Breaks loop edges with DFS
        // Can consider doing non-recursive dfs later if needed
        const visited = Array(this.blocks.length).fill(DfsState.NotVisited);
        // Perform a post-order traversal on the graph, adding numbers to the ordering here
        const order: number[] = [];
        const action = (node: number) => order.push(node);
        // Start with block zero, we assume this is the function entry-point. If that's ever not the case we'll need the
        // back-end to tell us which block is the entry point.
        GraphLayoutCore.postorderDFS(this.blocks, visited, 0, action);
        // It may be the case that not all blocks are reachable from the root, walk all the other blocks to ensure the
        // ordering is made
        for (let i = 0; i < this.blocks.length; i++) {
            GraphLayoutCore.postorderDFS(this.blocks, visited, i, action);
        }
        // We've computed a post-DFS ordering which is always a reverse topological ordering
        return order.reverse();
    }

    assignRows(topologicalOrder: number[]) {
        for (const i of topologicalOrder) {
            const block = this.blocks[i];
            for (const j of block.dagEdges) {
                const target = this.blocks[j];
                target.row = Math.max(target.row, block.row + 1);
            }
        }
    }

    computeTree(topologicalOrder: number[]) {
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

    computeTreeColumnPositions(node: number) {
        // Note: Currently not taking shape into account like Cutter does.
        // Note: Currently O(n^2) due to constant adjustments
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
            // If the node has more than two children we'll just center between the two direct children
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
            if (this.centerParents) {
                // center of bounding box
                block.col = Math.floor(Math.max(boundingBox.cols - 2, 0) / 2);
            } else {
                // center between immediate children
                const [left, right] = [this.blocks[block.treeEdges[0]], this.blocks[block.treeEdges[1]]];
                block.col = Math.floor((left.col + right.col) / 2);
            }
        }
    }

    assignBlockColumns(topologicalOrder: number[]) {
        // Go in reverse topological order, compute subtrees before parents
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

    getLayoutEvents() {
        const events: LayoutEvent[] = [];
        for (const [i, block] of this.blocks.entries()) {
            events.push({
                blockIndex: i,
                edgeIndex: -1,
                row: block.row,
                type: LayoutEventType.Block,
            });
            for (const [j, edge] of block.edges.entries()) {
                events.push({
                    blockIndex: i,
                    edgeIndex: j,
                    row: Math.max(block.row + 1, this.blocks[edge.dest].row),
                    type: LayoutEventType.Edge,
                });
            }
        }
        return events;
    }

    closestUnblockedColumn(sourceColumn: number, topRow: number, blockedColumns: number[]) {
        const leftCandidate =
            sourceColumn -
            1 -
            blockedColumns
                .slice(0, sourceColumn)
                .reverse()
                .findIndex(v => v < topRow);
        const rightCandidate = sourceColumn + blockedColumns.slice(sourceColumn).findIndex(v => v < topRow);
        return [leftCandidate, rightCandidate];
    }

    assignMainColumn(source: Block, target: Block, edge: Edge, blockedColumns: number[]) {
        const sourceColumn = source.col + 1;
        const targetColumn = target.col + 1;
        const topRow = Math.min(source.row + 1, target.row);
        if (blockedColumns[sourceColumn] < topRow) {
            // use column under source block if it isn't blocked
            edge.mainColumn = sourceColumn;
        } else if (blockedColumns[targetColumn] < topRow) {
            // use column of the target if it isn't blocked
            edge.mainColumn = targetColumn;
        } else {
            const [leftCandidate, rightCandidate] = this.closestUnblockedColumn(sourceColumn, topRow, blockedColumns);
            // hamming distance
            const distanceLeft = Math.abs(sourceColumn - leftCandidate) + Math.abs(targetColumn - leftCandidate);
            const distanceRight = Math.abs(sourceColumn - rightCandidate) + Math.abs(targetColumn - rightCandidate);
            // "figure 8" logic from cutter
            // Takes a longer path that produces less crossing
            if (target.row < source.row) {
                if (
                    targetColumn < sourceColumn &&
                    blockedColumns[sourceColumn + 1] < topRow &&
                    sourceColumn - targetColumn <= distanceLeft + 2
                ) {
                    edge.mainColumn = sourceColumn + 1;
                    return;
                }
                if (
                    targetColumn > sourceColumn &&
                    blockedColumns[sourceColumn - 1] < topRow &&
                    targetColumn - sourceColumn <= distanceRight + 2
                ) {
                    edge.mainColumn = sourceColumn - 1;
                    return;
                }
            }
            if (distanceLeft === distanceRight) {
                // Place true branches on the left
                // TODO: Need to investigate further block placement stuff here
                // TODO: Need to investigate further offset placement stuff for the start segments
                // TODO: Could also try something considering if the left/right columns are adjacent and target
                // is <= source
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

    computeEdgeMainColumns() {
        // This is heavily inspired by Cutter
        // An edge from block A to block B is done with a single vertical segment called the "main column." To choose
        // these main columns we use a sweep line algorithm to process the CFG top to bottom while keeping track of
        // which columns blocked. Cutter uses an augmented binary tree to assist with finding close empty columns. For
        // this just naively iterates to find an empty column.
        const events = this.getLayoutEvents();
        // Sort by row (max(src row, target row) for edges), edge row n is before block row n
        events.sort((a: LayoutEvent, b: LayoutEvent) => (a.row === b.row ? a.type - b.type : a.row - b.row));
        // Keep track of the last row where the column was blocked, we'll use that to check if we can route an edge
        // through a column between r0 and r1
        const blockedColumns = Array(this.columnCount + 1).fill(-1);
        for (const event of events) {
            if (event.type === LayoutEventType.Block) {
                const block = this.blocks[event.blockIndex];
                blockedColumns[block.col + 1] = block.row;
            } else {
                const source = this.blocks[event.blockIndex];
                const edge = source.edges[event.edgeIndex];
                const target = this.blocks[edge.dest];
                this.assignMainColumn(source, target, edge, blockedColumns);
            }
        }
    }

    addEdgeSegments(block: Block, edge: Edge) {
        const makeSegment = (
            [start_row, start_col]: [number, number],
            [end_row, end_col]: [number, number],
        ): EdgeSegment => ({
            start: {
                row: start_row,
                col: start_col,
                x: 0,
                y: 0,
            },
            end: {
                row: end_row,
                col: end_col,
                x: 0,
                y: 0,
            },
            horizontalOffset: 0,
            verticalOffset: 0,
            type: start_col === end_col ? SegmentType.Vertical : SegmentType.Horizontal,
        });
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
    }

    simplifyEdgePaths(edge: Edge) {
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
                        (prevSegment.type === SegmentType.Vertical && prevSegment.start.col !== segment.start.col) ||
                        (prevSegment.type === SegmentType.Horizontal && prevSegment.start.row !== segment.start.row)
                    ) {
                        throw Error("Adjacent horizontal or vertical segments don't share a common row or column");
                    }
                    prevSegment.end = segment.end;
                    edge.path.splice(i, 1);
                    movement = true;
                }
            }
        } while (movement);
    }

    checkEdgePaths(edge: Edge) {
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
    }

    routeEdgePaths() {
        for (const block of this.blocks) {
            for (const edge of block.edges) {
                this.addEdgeSegments(block, edge);
                this.simplifyEdgePaths(edge);
                this.checkEdgePaths(edge);
            }
        }
    }

    classifyEdgeSegment(i: number, path: EdgeSegment[]) {
        const segment = path[i];
        let kind = EdgeKind.NULL;
        if (i === 0) {
            if (path.length === 1) {
                // Segment will be vertical
                kind = EdgeKind.VERTICAL;
            } else {
                // There will be a next
                const next = path[i + 1];
                if (next.end.col > segment.end.col) {
                    kind = EdgeKind.RIGHTCORNER;
                } else {
                    kind = EdgeKind.LEFTCORNER;
                }
            }
        } else if (i === path.length - 1) {
            // There will be a previous segment, i !== 0, but no next
            const previous = path[i - 1];
            if (previous.start.col > segment.end.col) {
                kind = EdgeKind.RIGHTCORNER;
            } else {
                kind = EdgeKind.LEFTCORNER;
            }
        } else {
            // There will be both a previous and a next
            const next = path[i + 1];
            const previous = path[i - 1];
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
        return kind;
    }

    getEdgeSegmentInfo() {
        const segments: SegmentInfo[] = [];
        for (const block of this.blocks) {
            for (const edge of block.edges) {
                const edgeLength = edge.path
                    .map(({start, end}) => Math.abs(start.col - end.col) + Math.abs(start.row - end.row))
                    .reduce((A, x) => A + x);
                const target = this.blocks[edge.dest];
                for (const [i, segment] of edge.path.entries()) {
                    const kind = this.classifyEdgeSegment(i, edge.path);
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
        return segments;
    }

    computeEdgeSegmentIntervals() {
        const segments = this.getEdgeSegmentInfo();
        segments.sort((a, b) => {
            if (a.kind !== b.kind) {
                return a.kind - b.kind;
            }
            const kind = a.kind; // a.kind == b.kind
            if (a.length !== b.length) {
                if (kind <= 0) {
                    // shortest first if coming from the left
                    return a.length - b.length;
                }
                // coming from the right, shortest last
                // reverse edge length order
                return b.length - a.length;
            }
            if (kind <= 0) {
                return a.tiebreaker - b.tiebreaker;
            }
            // coming from the right, reverse
            return b.tiebreaker - a.tiebreaker;
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
                    const tree = new IntervalTree<EdgeSegment>();
                    col.intervals.push(tree);
                    col.subcolumns++;
                    tree.insert([segment.start.row, segment.end.row], segment);
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
                    const tree = new IntervalTree<EdgeSegment>();
                    row.intervals.push(tree);
                    row.subrows++;
                    tree.insert([segment.start.col, segment.end.col], segment);
                }
            }
        }
    }

    assignEdgeSegments() {
        this.computeEdgeSegmentIntervals();
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

    updateBlockDimensions() {
        for (const block of this.blocks) {
            // Update block width if it has a ton of incoming edges
            block.data.width = Math.max(block.data.width, (block.incidentEdgeCount - 1) * EDGE_SPACING);
        }
    }

    computeGridDimensions() {
        for (const block of this.blocks) {
            const halfWidth = (block.data.width - this.edgeColumns[block.col + 1].width) / 2;
            this.blockRows[block.row].height = Math.max(this.blockRows[block.row].height, block.data.height);
            this.blockColumns[block.col].width = Math.max(this.blockColumns[block.col].width, halfWidth);
            this.blockColumns[block.col + 1].width = Math.max(this.blockColumns[block.col + 1].width, halfWidth);
        }
    }

    computeGridOffsets() {
        for (let i = 0; i < this.rowCount; i++) {
            // edge row 0 is already at the correct offset, this iteration will set the offset for block row 0 and edge
            // row 1.
            this.blockRows[i].totalOffset = this.edgeRows[i].totalOffset + this.edgeRows[i].height;
            this.edgeRows[i + 1].totalOffset = this.blockRows[i].totalOffset + this.blockRows[i].height;
        }
        for (let i = 0; i < this.columnCount; i++) {
            this.blockColumns[i].totalOffset = this.edgeColumns[i].totalOffset + this.edgeColumns[i].width;
            this.edgeColumns[i + 1].totalOffset = this.blockColumns[i].totalOffset + this.blockColumns[i].width;
        }
    }

    computeBlockCoordinates(block: Block) {
        block.coordinates.x =
            this.edgeColumns[block.col + 1].totalOffset -
            (block.data.width - this.edgeColumns[block.col + 1].width) / 2;
        block.coordinates.y = this.blockRows[block.row].totalOffset;
    }

    computeEdgeCoordinates(block: Block, edge: Edge) {
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

    computeCoordinates() {
        this.updateBlockDimensions();
        this.computeGridDimensions();
        this.computeGridOffsets();
        // Compute block coordinates and edge paths
        for (const block of this.blocks) {
            this.computeBlockCoordinates(block);
            for (const edge of block.edges) {
                this.computeEdgeCoordinates(block, edge);
            }
        }
    }

    layout() {
        this.countEdges();
        const topologicalOrder = this.computeDag();
        this.assignRows(topologicalOrder);
        this.computeTree(topologicalOrder);
        this.assignBlockColumns(topologicalOrder);
        this.setupRowsAndColumns();
        // Edge routing
        this.computeEdgeMainColumns();
        this.routeEdgePaths();
        this.assignEdgeSegments();
        // -- Nothing is pixel aware above this line ---
        // Add pixel coordinates
        this.computeCoordinates();
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
