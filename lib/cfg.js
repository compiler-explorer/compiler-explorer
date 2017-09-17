// Copyright (c) 2012-2017, Najjar Chedy
//
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


var _ = require('underscore-node');

function ControlFlowGraph(rules){
    this.rules = rules;
    
}

function seperateCodeFromData(asmArr) {
    
    var isCode = function(asmArr, index){
        return (((asmArr[index].source !== null) &&
                (asmArr[index+1].source !== null)) ||
                ((asmArr[index].source === null) &&
                 (asmArr[index+1].source !== null)) ||
                 ((asmArr[index].source !== null) &&
                 (asmArr[index+1].source === null)));
    };

    var code = [];
    var first = 0;
    var last = asmArr.length;
    
    if(first == last) return code;

    
    while(first != last-1){
        if(isCode(asmArr, first))
            code.push(JSON.parse(JSON.stringify(asmArr[first])));
        
        ++first;
    }
    if(asmArr[first].source){
        code.push(JSON.parse(JSON.stringify(asmArr[first])));
    }

    return code;
}

    var gccX86 = {
                filterData:  seperateCodeFromData,
                isFunctionEnd:function(x)  { return ((x[0] != ' ') && (x[0] != '.') &&
                                                             (x.indexOf(':') != -1)) ;},

                isBasicBlockEnd: function(x) { return x[0] == ".";} ,

                getInstructionType:function(inst) {
                                            inst = inst.trim();
                                            if (inst === "") return -1;
                                            else if(inst.includes("jmp")) return 0;
                                            else if(inst[0] === 'j') return 1;
                                            else if(!inst.includes("ret")) return 2;
                                            else return 3;
                                          },

                extractNodeName:function(inst) {
                                     var name = inst.match(/.L\d+/);
                                     return name + ":";
                                  },

                isJmpInstruction: function(x) { var trimed = x.trim(); return ( trimed[0] == 'j');}
             };




ControlFlowGraph.prototype.splitToFunctions = function (asmArr, isEnd) {
    if(asmArr.length === 0)   return [];
    var result = [];
    var first = 0;
    var last = asmArr.length;
    var fnRange = {start:first, end:null};
    ++first;
    while(first != last) {
        if(isEnd(asmArr[first].text)) {
            fnRange.end = first;
            result.push(JSON.parse(JSON.stringify(fnRange)));
            fnRange.start = first;
        }
        ++first;
    }

    fnRange.end = last;
    result.push(fnRange);//possiblitiy of bug?! because didn't clone(JSON.parse(JSON.stringify(...))
    return result;
};

ControlFlowGraph.prototype.splitToBasicBlocks = function (asmArr,  range, isEnd, isJmp) {
    var first = range.start;
    var last = range.end;
    if(first == last) return [];
    ++first;

    var rangeBb = {nameId:"start" , start:first , end:null, actionPos:[]};
    var result = [];

    var resetRangeWith = function(rangeBb,  nameId, start) {
            rangeBb.nameId = nameId;//.L1:
            rangeBb.start = start;//after .L1:
            rangeBb.actionPos = [];
    };

    while(first != last) {
        var inst = asmArr[first].text;
        if(isEnd(inst)) {
            rangeBb.end = first;
            result.push(JSON.parse(JSON.stringify(rangeBb)));
            ++first;//risk of a bug.
            //inst is expected to be .L*: where * in 1,2,...
            resetRangeWith(rangeBb, inst, first);
        }
        else if(isJmp(inst)) {
                rangeBb.actionPos.push(first);
        }
        ++first;
    }

    rangeBb.end = last;
    result.push(JSON.parse(JSON.stringify(rangeBb)));
    return result;
};

ControlFlowGraph.prototype.splitToCanonicalBasicBlock = function (basicBlock) {
    var actionPos = basicBlock.actionPos;
    var actPosSz = actionPos.length;
    if (actionPos[actPosSz-1] + 1 == basicBlock.end){
        --actPosSz;
    }

    if(actPosSz === 0)
        return [{
                    nameId: basicBlock.nameId,
                    start: basicBlock.start,
                    end: basicBlock.end
               }];
    else if(actPosSz == 1)
            return [
                {nameId: basicBlock.nameId,start: basicBlock.start,end:actionPos[0]+1},
                {nameId: basicBlock.nameId+"@"+ (actionPos[0]+1),start: actionPos[0]+1,end: basicBlock.end}
               ];
    else {
        var first = 0;
        var last = actPosSz;
        var block_name = basicBlock.nameId;
        var tmp = {nameId:block_name, start:basicBlock.start, end:actionPos[first]+1 };
        var result = [];
        result.push(JSON.parse(JSON.stringify(tmp)));
        while(first != last-1) {
            tmp.nameId = block_name + "@" + (actionPos[first]+1);
            tmp.start = actionPos[first]+1;
            ++first;
            tmp.end = actionPos[first]+1;
            result.push(JSON.parse(JSON.stringify(tmp)));
        }
        
        tmp = {nameId:block_name + "@" + (actionPos[first] + 1), start:actionPos[first] + 1, end:basicBlock.end };
        result.push(JSON.parse(JSON.stringify(tmp)));
        
        return result;

    }

};

ControlFlowGraph.prototype.concatInstructions = function (asmArr, first, last) {
    
    var makeShorter = function(instruction){
        if(instruction.length > 50){
            return instruction.substr(0, 50);
        }
        
        return instruction;
    };
    
    if(first == last) return "";//if last -1 is changed to last this line is no longuer needed
    console.log("look");
    var result = "";
    //TODO: include \n for long templated instructions
    while(first != last-1) {//added to delete last \n and handle the last concat outside loop
        console.log(asmArr[first].text);
        result += makeShorter(asmArr[first].text) + "\n";
        ++first;
    }
    //last concat withou \n
    result += makeShorter(asmArr[first].text);
    console.log(asmArr[first].text);

    return result;
};

ControlFlowGraph.prototype.makeNodes = function (asmArr, arrOfCanonicalBasicBlock) {
    var self = this;
    var node = {};
    var nodes = [];
    /*for(var x of arrOfCanonicalBasicBlock) {
        console.log("node name:")
        console.log(x.nameId);
        node["id"] = x.nameId;
        node["label"] = x.nameId + ((x.nameId.indexOf(":") != -1)? "":":") +"\n"
                        + this.concatInstructions(asmArr,x.start, x.end);
        node["color"] = "#99ccff";//"#FFAFAF";
        node["shape"] = 'box';
        nodes.push(JSON.parse(JSON.stringify(node)));
    }*/
    
    _.each(arrOfCanonicalBasicBlock, function(x){
        console.log("node name:");
        console.log(x.nameId);
        node.id = x.nameId;
        node.label = x.nameId + ((x.nameId.indexOf(":") != -1)? "":":") +"\n"+
                         self.concatInstructions(asmArr,x.start, x.end);
        node.color = "#99ccff";//"#FFAFAF";
        node.shape = 'box';
        nodes.push(JSON.parse(JSON.stringify(node)));
    });
    return nodes;
};

ControlFlowGraph.prototype.makeEdges = function (asmArr, arrOfCanonicalBasicBlock, instructionType, extractNodeNameFromInstruction) {

    var jmpInst = 0;
    var conditionalJmpInst = 1;
    var notRetInst = 2;
    var edge = {};
    var edges = [];

    var setEdge = function(edge, sourceNode, targetNode, color) {
        edge.from = sourceNode;
        edge.to = targetNode;
        edge.arrows = "to";
        edge.color = color;

    };
    
    var hasName = function(asmArr, cbb){
        return gccX86.isBasicBlockEnd(asmArr[cbb.end].text);
    };
    
    var lastInstruction = function(asmArr, cbb){
        
        while(asmArr[cbb.end-1].text === ""){
            --cbb.end;
        }
                
        return asmArr[cbb.end-1].text;
    };
    
    var generateName = function(name, suffix){
        var pos = name.indexOf("@");
        if(pos == -1)
            return name + "@" + suffix;
        
        return name.substring(0, pos+1)+ suffix;
    };

    // note: x.end-1 possible value: jmp .L*, {jne,je,jg,...} .L*, ret/rep ret, call and any other instruction that doesn't change control flow

    //for(var x of arrOfCanonicalBasicBlock) {
    _.each(arrOfCanonicalBasicBlock, function(x){
        var targetNode;
        var lastInst = lastInstruction(asmArr, x);//asmArr[x.end-1].text;
        switch(instructionType(lastInst)) {
            case jmpInst:
                {

                //we have to deal only with jmp destination, jmp instruction are always taken.
                    //edge from jump inst
                    console.log("jmp");
                    targetNode = extractNodeNameFromInstruction(lastInst);
                    setEdge(edge, x.nameId, targetNode, 'blue');

                    edges.push(JSON.parse(JSON.stringify(edge)));
                    console.log(edge);
                }
                break;
            case conditionalJmpInst:
                {
                    console.log("condit jmp");
                //deal with : branche taken, branch not taken
                    targetNode = extractNodeNameFromInstruction(lastInst);
                    setEdge(edge, x.nameId, targetNode, 'green');
                    edges.push(JSON.parse(JSON.stringify(edge)));
                    console.log(edge);

                    targetNode = hasName(asmArr,x)? asmArr[x.end].text: generateName(x.nameId, x.end);
                    setEdge(edge, x.nameId, targetNode, "red");
                    edges.push(JSON.parse(JSON.stringify(edge)));

                    console.log(edge);
                }
                break;
            case notRetInst:
                {
                //precondition: lastInst is not last instruction in asmArr (but it is in canonical basic block)
                //note : asmArr[x.end] expected to be .L*:(name of a basic block)
                //       this .L*: has to be exactly after the last instruction in the current canocial basic block
                    var nextNodeName = asmArr[x.end].text;
                    setEdge(edge, x.nameId, nextNodeName, 'grey');
                    edges.push(JSON.parse(JSON.stringify(edge)));
                    console.log("not ret inst");
                    console.log(edge);
                }
                break;
            default:
                    console.log("expect ret instruction or it's variants(rep ret): "+ lastInst);
                break;
        }
    });

    console.log(edges);

    return edges;
};


ControlFlowGraph.prototype.generateCFGStructure = function (asmArr){
    
    var rules = gccX86;
    
    var self = this;
    var code = rules.filterData(asmArr);
    
    var funcs = self.splitToFunctions(code, rules.isFunctionEnd);
    
    if (!funcs.length) { 
        return funcs; 
    }
    
    var result = {};
    
    //for (var rng of funcs) {
    _.each(funcs, function(rng){
        var basicBlocks = self.splitToBasicBlocks(code, rng, rules.isBasicBlockEnd,
                                                   rules.isJmpInstruction);
        var arrOfCanonicalBasicBlock = [];
        //for (var elm of basicBlocks) {
        _.each(basicBlocks, function(elm){
            var tmp = self.splitToCanonicalBasicBlock(elm);
            arrOfCanonicalBasicBlock = arrOfCanonicalBasicBlock.concat(tmp);
        });
            
        result[code[rng.start].text] = {
            nodes:self.makeNodes(code, arrOfCanonicalBasicBlock),
            edges: self.makeEdges(code, arrOfCanonicalBasicBlock,
                rules.getInstructionType,
                rules.extractNodeName)
            };
        });
        
    return result;
    
};

module.exports.ControlFlowGraph = ControlFlowGraph;