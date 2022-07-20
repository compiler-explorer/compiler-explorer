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

import { AnnotatedCfgDescriptor, AnnotatedNodeDescriptor } from "./compilation/cfg.interfaces";

// Much of the algorithm is inspired from
// https://cutter.re/docs/api/widgets/classGraphGridLayout.html
// Thanks to the cutter team for their great documentation!

enum EdgeType { Horizontal, Vertical };

type EdgePoint = {
    row: number;
    col: number;
    offsetX: number;
    offsetY: number;
    type: EdgeType; // is this point the end of a horizontal or vertical segment
};

type Coordinate = {
    x: number;
    y: number;
};

type Edge = {
    color: string;
    dest: number;
    mainColumn: number;
    points: EdgePoint[],
    path: Coordinate[]
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
};

enum DfsState { NotVisited, Pending, Visited };

type ColumnDescriptor = {
    width: number;
    totalOffset: number;
};
type RowDescriptor = {
    height: number;
    totalOffset: number;
};
type EdgeColumnRowMetadata = {
    currentOffset: number;
};

export class GraphLayoutCore {
    // We use an adjacency list here
    blocks: Block[] = [];
    columnCount: number;
    rowCount: number;
    blockColumns: ColumnDescriptor[];
    blockRows: RowDescriptor[];
    edgeColumns: (ColumnDescriptor & EdgeColumnRowMetadata)[];
    edgeRows: (RowDescriptor & EdgeColumnRowMetadata)[];

    constructor(cfg: AnnotatedCfgDescriptor) {
        // block id -> block
        const blockMap: Record<string, number> = {};
        for(const node of cfg.nodes) {
            const block = {
                data: node,
                edges: [],
                dagEdges: [],
                treeEdges: [],
                treeParent: null,
                row: 0,
                col: 0,
                boundingBox: {rows: 0, cols: 0},
                coordinates: {x: 0, y: 0}
            };
            this.blocks.push(block);
            blockMap[node.id] = this.blocks.length - 1;
        }
        for(const {from, to, color} of cfg.edges) {
            this.blocks[blockMap[from]].edges.push({
                color,
                dest: blockMap[to],
                mainColumn: -1,
                points: [],
                path: []
            });
        }
        //console.log(this.blocks);
        this.layout();
    }

    dfs(visited: DfsState[], order: number[], node: number) {
        if(visited[node] == DfsState.Visited) {
            return;
        }
        if(visited[node] == DfsState.NotVisited) {
            visited[node] = DfsState.Pending;
            const block = this.blocks[node];
            for(const edge of block.edges) {
                // If we reach another pending node it's a loop edge.
                // If we reach an unvisited node it's fine, if we reach a visited node that's also part of the dag
                if(visited[edge.dest] != DfsState.Pending) {
                    block.dagEdges.push(edge.dest);
                }
                this.dfs(visited, order, edge.dest);
            }
            visited[node] = DfsState.Visited;
            order.push(node);
        } else { // visited[node] == DfsState.Pending
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
        for(let i = 0; i < this.blocks.length; i++) {
            //// Let's assume one entry point for now.... No weird edge cases...
            //if(visited[i] != DfsState.Visited) {
            //    throw Error("");
            //}
            this.dfs(visited, order, i);
        }
        // we've computed a post-DFS ordering which is always a reverse topological ordering
        return order.reverse();
    }

    assignRows(topologicalOrder) {
        for(const i of topologicalOrder) {
            const block = this.blocks[i];
            for(const j of block.dagEdges) {
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
        for(const i of topologicalOrder) {
            // Only dag edges are considered
            // Edges - dag edges = the set of back edges
            const block = this.blocks[i];
            for(const j of block.dagEdges) {
                const target = this.blocks[j];
                if(target.treeParent == null && target.row == block.row + 1) {
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
        for(const j of block.treeEdges) {
            this.adjustSubtree(j, rowShift, columnShift);
        }
    }

    // Note: Currently O(n^2)
    computeTreeColumnPositions(node: number) {
        const block = this.blocks[node];
        if(block.treeEdges.length == 0) {
            block.row = 0;
            block.col = 0;
            block.boundingBox = {
                rows: 1,
                cols: 2
            };
        } else if(block.treeEdges.length == 1) {
            const childIndex = block.treeEdges[0];
            const child = this.blocks[childIndex];
            block.row = 0;
            block.col = child.col;
            block.boundingBox = {
                rows: 1 + child.boundingBox.rows,
                cols: child.boundingBox.cols
            };
            this.adjustSubtree(childIndex, 1, 0);
        } else {
            // If the node has more than two children we'll just center between the
            //let selectedTreeEdges = block.treeEdges.slice(0, 2);
            const boundingBox = {
                rows: 0,
                cols: 0
            };
            // Compute bounding box of all the subtrees and adjust
            for(const i of block.treeEdges) {
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
        for(const i of topologicalOrder.slice().reverse()) {
            this.computeTreeColumnPositions(i);
        }
    }

    computeEdgeMainColumns() {
        // This is heavily inspired by Cutter
        // We use a sweep line algorithm processing the CFG top to bottom keeping track of when columns are most
        // recently blocked. Cutter uses an augmented binary tree to assist with finding empty columns, for now this
        // just naively iterates.
        enum EventType { Edge = 0, Block = 1 };
        type Event = {
            blockIndex: number;
            edgeIndex: number;
            row: number;
            type: EventType;
        };
        const events: Event[] = [];
        for(const [i, block] of this.blocks.entries()) {
            events.push({
                blockIndex: i,
                edgeIndex: -1,
                row: block.row,
                type: EventType.Block
            });
            for(const [j, edge] of block.edges.entries()) {
                events.push({
                    blockIndex: i,
                    edgeIndex: j,
                    row: Math.max(block.row + 1, this.blocks[edge.dest].row),
                    type: EventType.Edge
                })
            }
        }
        // Sort by row (max(src row, target row) for edges), edge row n is before block row n
        events.sort((a: Event, b: Event) => {
            if(a.row == b.row) {
                return a.type - b.type;
            } else {
                return a.row - b.row;
            }
        });
        //
        const blockedColumns = Array(this.columnCount + 1).fill(-1);
        for(const event of events) {
            if(event.type == EventType.Block) {
                const block = this.blocks[event.blockIndex];
                blockedColumns[block.col + 1] = block.row;
            } else {
                const source = this.blocks[event.blockIndex];
                const edge = source.edges[event.edgeIndex];
                const target = this.blocks[edge.dest];
                const sourceColumn = source.col + 1;
                const targetColumn = target.col + 1;
                const topRow = Math.min(source.row + 1, target.row);
                if(blockedColumns[sourceColumn] < topRow) { // try column under source block
                    edge.mainColumn = sourceColumn;
                } else if(blockedColumns[targetColumn] < topRow) { // try column of the target
                    edge.mainColumn = targetColumn;
                } else {
                    const leftCandidate = sourceColumn - 1 - blockedColumns.slice(0, sourceColumn).reverse().findIndex(v => v < topRow);
                    const rightCandidate = sourceColumn + blockedColumns.slice(sourceColumn).findIndex(v => v < topRow);
                    // hamming distance
                    const distanceLeft = Math.abs(sourceColumn - leftCandidate) + Math.abs(sourceColumn - rightCandidate);
                    const distanceRight = Math.abs(targetColumn - leftCandidate) + Math.abs(targetColumn - rightCandidate);
                    // TODO: Handle if they are both equally close
                    if(distanceLeft < distanceRight) {
                        edge.mainColumn = leftCandidate;
                    } else {
                        edge.mainColumn = rightCandidate;
                    }
                }
            }
        }
    }

    addEdgePaths() {
        for(const block of this.blocks) {
            for(const edge of block.edges) {
                const target = this.blocks[edge.dest];
                // start just below the source block
                edge.points.push({
                    row: block.row + 1,
                    col: block.col + 1,
                    offsetX: 0,
                    offsetY: 0,
                    type: EdgeType.Vertical
                });
                // horizontal segment over to main column
                edge.points.push({
                    row: block.row + 1,
                    col: edge.mainColumn,
                    offsetX: 0,
                    offsetY: 0,
                    type: EdgeType.Horizontal
                });
                // vertical segment down the main column
                edge.points.push({
                    row: target.row,
                    col: edge.mainColumn,
                    offsetX: 0,
                    offsetY: 0,
                    type: EdgeType.Vertical
                });
                // horizontal segment over to the target column
                edge.points.push({
                    row: target.row,
                    col: target.col + 1,
                    offsetX: 0,
                    offsetY: 0,
                    type: EdgeType.Horizontal
                });
                // TODO: Merge redundant segments
                // Compute offsets
            }
        }
    }

    setupRowsAndColumns() {
        console.log(this.blocks);
        this.rowCount = Math.max(...this.blocks.map(block => block.row)) + 1; // one more row for zero offset
        this.columnCount = Math.max(...this.blocks.map(block => block.col)) + 2; // blocks are two-wide
        this.blockRows = Array(this.rowCount).fill(0).map(() => ({
            height: 0,
            totalOffset: 0
        }));
        this.blockColumns = Array(this.columnCount).fill(0).map(() => ({
            width: 0,
            totalOffset: 0
        }));
        this.edgeRows = Array(this.rowCount + 1).fill(0).map(() => ({
            height: 20,
            totalOffset: 0,
            currentOffset: 0
        }));
        this.edgeColumns = Array(this.columnCount + 1).fill(0).map(() => ({
            width: 20,
            totalOffset: 0,
            currentOffset: 0
        }));
    }

    computeCoordinates() {
        // Compute block row widths and heights
        for(const block of this.blocks) {
            console.log(this.blockRows[block.row].height, block.data.height, block.row);
            //console.log(this.blockRows);
            const halfWidth = (block.data.width - 20) / 2;
            console.log("--->", block.col, this.columnCount);
            this.blockRows[block.row].height = Math.max(this.blockRows[block.row].height, block.data.height);
            this.blockColumns[block.col].width = Math.max(this.blockColumns[block.col].width, halfWidth);
            this.blockColumns[block.col + 1].width = Math.max(this.blockColumns[block.col + 1].width, halfWidth);
        }
        // Compute row total offsets
        for(let i = 0; i < this.rowCount; i++) {
            // edge row 0 is already at the correct offset, this iteration will set the offset for block row 0 and edge
            // row 1.
            this.blockRows[i].totalOffset = this.edgeRows[i].totalOffset + this.edgeRows[i].height;
            this.edgeRows[i + 1].totalOffset = this.blockRows[i].totalOffset + this.blockRows[i].height;
        }
        // Compute column total offsets
        for(let i = 0; i < this.columnCount; i++) {
            // same deal here
            this.blockColumns[i].totalOffset = this.edgeColumns[i].totalOffset + this.edgeColumns[i].width;
            this.edgeColumns[i + 1].totalOffset = this.blockColumns[i].totalOffset + this.blockColumns[i].width;
        }
        // Compute block coordinates and edge paths
        for(const block of this.blocks) {
            block.coordinates.x = this.edgeColumns[block.col + 1].totalOffset - (block.data.width - 20) / 2;
            block.coordinates.y = this.blockRows[block.row].totalOffset;
            for(const edge of block.edges) {
                // push initial point
                edge.path.push({
                    x: this.edgeColumns[block.col + 1].totalOffset + 10,
                    y: block.coordinates.y + block.data.height
                });
                for(const point of edge.points) {
                    edge.path.push({
                        x: this.edgeColumns[point.col].totalOffset + 10,
                        y: this.edgeRows[point.row].totalOffset + 10
                    });
                }
                // push final point
                const target = this.blocks[edge.dest];
                edge.path.push({
                    x: this.edgeColumns[target.col + 1].totalOffset + 10,
                    y: this.edgeRows[target.row].totalOffset + this.edgeRows[target.row].height
                });
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
        // Add pixel coordinates
        this.computeCoordinates();
        //
        console.log(this);
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
