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


var _ = require('underscore-node'),
    logger = require('./logger').logger;
    
    
var InstructionType_jmp = 0;
var InstructionType_conditionalJmpInst = 1;
var InstructionType_notRetInst = 2;    
var InstructionType_retInst = 3;    


// this mess is intended to be ported to another file.
function separateCodeFromData(asmArr) {
    
    var labelRegex = /\.L{1}\d+:/;
    
    var isCode = function(x){
        return  x && x.text && (x.source !== null || labelRegex.test(x.text) || 
                !x.text.includes('.'));
    };
    
    return _.chain(asmArr)
            .map(_.clone)
            .filter(isCode)
            .value();
}

    var gccX86 = {
                filterData:  separateCodeFromData,
                isFunctionEnd:function(x)  { return ((x[0] != ' ') && (x[0] != '.') &&
                                                             (x.indexOf(':') != -1)) ;},

                isBasicBlockEnd: function(x) { return x[0] == ".";} ,

                getInstructionType:function(inst) {
                                            if(inst.includes("jmp")) return InstructionType_jmp;
                                            else if(inst.trim()[0] === 'j') return InstructionType_conditionalJmpInst;
                                            else if(!inst.includes(" ret")) return InstructionType_notRetInst;
                                            else return InstructionType_retInst;
                                          },

                extractNodeName:function(inst) {
                                     var name = inst.match(/\.L{1}\d+/);
                                     return name + ":";
                                  },

                isJmpInstruction: function(x) { var trimed = x.trim(); return ( trimed[0] === 'j');}
             };
 
function separateCodeFromData_(asmArr) {
    
    var labelRegex = /\.LBB\d+_\d+:/;
    var isFunctionName = function(x){
        var functionNameRegex = /((\w|\(|\))*): # @\1/;
        return functionNameRegex.test(x);
    }
    
    var isCode = function(x){
        return  x && x.text && (x.source !== null || labelRegex.test(x.text) || 
                isFunctionName(x.text));
    };
    
    var removeComments = function(x) {
        var pos = x.text.indexOf("#");
        if(pos !== -1)
            x.text = x.text.substring(0, pos-1);
        return x;
    }
    
    return _.chain(asmArr)
            .map(_.clone)
            .filter(isCode)
            .map(removeComments)
            .value();
}

    var clangX86 = {
                filterData:  separateCodeFromData_,
                isFunctionEnd:function(x)  { return ((x[0] !== ' ') && (x[0] !== '.') &&
                                                             (x.indexOf(':') !== -1)) ;},

                isBasicBlockEnd: function(x) { return x[0] == ".";} ,

                getInstructionType:function(inst) {
                                            if(inst.includes("jmp")) return InstructionType_jmp;
                                            else if(inst.trim()[0] === 'j') return InstructionType_conditionalJmpInst;
                                            else if(!inst.includes(" ret")) return InstructionType_notRetInst;
                                            else return InstructionType_retInst;
                                          },

                extractNodeName:function(inst) {
                                     var name = inst.match(/\.LBB\d+_\d+/);
                                     return name + ":";
                                  },

                isJmpInstruction: function(x) { var trimed = x.trim(); return ( trimed[0] == 'j');}
             };

function ControlFlowGraph(compiler){
    if(compiler.includes("clang"))
        this.rules = clangX86;
    else 
        this.rules = gccX86;
}

ControlFlowGraph.prototype.splitToFunctions = function (asmArr, isEnd) {
    if(asmArr.length === 0)   return [];
    var result = [];
    var first = 0;
    var last = asmArr.length;
    var fnRange = {start:first, end:null};
    ++first;
    console.log(asmArr);
    while(first != last) {
        if(isEnd(asmArr[first].text)) {
            fnRange.end = first;
            result.push(_.clone(fnRange));
            fnRange.start = first;
        }
        ++first;
    }

    fnRange.end = last;
    result.push(_.clone(fnRange));
    return result;
};


ControlFlowGraph.prototype.splitToBasicBlocks = function (asmArr,  range, isEnd, isJmp) {
    var first = range.start;
    var last = range.end;
    if(first == last) return [];
    var functionName = asmArr[first].text;
    ++first;

    var rangeBb = {nameId:functionName.substr(0,50) , start:first , end:null, actionPos:[]};
    var result = [];

    var newRangeWith = function (oldRange, nameId, start) {
        return {nameId: nameId, start: start, actionPos: [], end: oldRange.end};
    };

    while(first != last) {
        var inst = asmArr[first].text;
        if(isEnd(inst)) {
            rangeBb.end = first;
            result.push(_.clone(rangeBb));
            ++first;
            //inst is expected to be .L*: where * in 1,2,...
            rangeBb = newRangeWith(rangeBb, inst, first);
        }
        else if(isJmp(inst)) {
                rangeBb.actionPos.push(first);
        }
        ++first;
    }

    rangeBb.end = last;
    result.push(_.clone(rangeBb));
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
        var blockName = basicBlock.nameId;
        var tmp = {nameId:blockName, start:basicBlock.start, end:actionPos[first]+1 };
        var result = [];
        result.push(_.clone(tmp));
        while(first != last-1) {
            tmp.nameId = blockName + "@" + (actionPos[first]+1);
            tmp.start = actionPos[first]+1;
            ++first;
            tmp.end = actionPos[first]+1;
            result.push(_.clone(tmp));
        }
        
        tmp = {nameId:blockName + "@" + (actionPos[first] + 1), start:actionPos[first] + 1, end:basicBlock.end };
        result.push(_.clone(tmp));
        
        return result;

    }

};


ControlFlowGraph.prototype.concatInstructions = function (asmArr, first, last) {
  return _.chain(asmArr.slice(first, last))
    .map(function (x) { return x.text.substr(0, 50); })
    .value()
    .join("\n");
}; 


ControlFlowGraph.prototype.makeNodes = function (asmArr, arrOfCanonicalBasicBlock) {
    var node = {};
    var nodes = [];
    
    _.each(arrOfCanonicalBasicBlock, _.bind(function(x){
        logger.debug("node name:");
        logger.debug(x.nameId);
        node.id = x.nameId;
        node.label = x.nameId + ((x.nameId.indexOf(":") !== -1)? "":":") +"\n"+
                         this.concatInstructions(asmArr,x.start, x.end);
        node.color = "#99ccff";//"#FFAFAF";
        node.shape = 'box';
        nodes.push(_.clone(node));
    }, this));
    return nodes;
};

ControlFlowGraph.prototype.makeEdges = function (asmArr, arrOfCanonicalBasicBlock, instructionType, extractNodeNameFromInstruction) {
    var edge = {};
    var edges = [];

    var setEdge = function(edge, sourceNode, targetNode, color) {
        edge.from = sourceNode;
        edge.to = targetNode;
        edge.arrows = "to";
        edge.color = color;
    };
    var isBasicBlockEnd = this.rules.isBasicBlockEnd;
    
    var hasName = function(asmArr, cbb){
        return isBasicBlockEnd(asmArr[cbb.end].text);
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
        var lastInst = asmArr[x.end-1].text;
        switch(instructionType(lastInst)) {
            case InstructionType_jmp:
                {

                //we have to deal only with jmp destination, jmp instruction are always taken.
                    //edge from jump inst
                    logger.debug("jmp");
                    targetNode = extractNodeNameFromInstruction(lastInst);
                    setEdge(edge, x.nameId, targetNode, 'blue');

                    edges.push(_.clone(edge));
                    logger.debug(edge);
                }
                break;
            case InstructionType_conditionalJmpInst:
                {
                    logger.debug("condit jmp");
                //deal with : branche taken, branch not taken
                    targetNode = extractNodeNameFromInstruction(lastInst);
                    setEdge(edge, x.nameId, targetNode, 'green');
                    edges.push(_.clone(edge));
                    logger.debug(edge);

                    targetNode = hasName(asmArr,x)? asmArr[x.end].text: generateName(x.nameId, x.end);
                    setEdge(edge, x.nameId, targetNode, "red");
                    edges.push(_.clone(edge));

                    logger.debug(edge);
                }
                break;
            case InstructionType_notRetInst:
                {
                //precondition: lastInst is not last instruction in asmArr (but it is in canonical basic block)
                //note : asmArr[x.end] expected to be .L*:(name of a basic block)
                //       this .L*: has to be exactly after the last instruction in the current canocial basic block
                if(asmArr[x.end]){
                    targetNode = asmArr[x.end].text;
                    setEdge(edge, x.nameId, targetNode, 'grey');
                    edges.push(_.clone(edge));
                    logger.debug("not ret inst");
                    logger.debug(edge);   
                 }else {
                     logger.debug(x);
                 }
                }
                break;
            case InstructionType_retInst:
                    logger.debug("expect ret instruction or it's variants(rep ret): "+ lastInst);
                break;
        }
    });

    logger.debug(edges);

    return edges;
};


ControlFlowGraph.prototype.generateCfgStructure = function (asmArr){
    
    
    var self = this;
    var code = this.rules.filterData(asmArr);
    
    var funcs = this.splitToFunctions(code, this.rules.isFunctionEnd);
    
    if (!funcs.length) { 
        return funcs; 
    }
    
    var result = {};
    
    //for (var rng of funcs) {
    _.each(funcs, _.bind(function(rng){
        var basicBlocks = self.splitToBasicBlocks(code, rng, this.rules.isBasicBlockEnd,
                                                   this.rules.isJmpInstruction);
        var arrOfCanonicalBasicBlock = [];
        //for (var elm of basicBlocks) {
        _.each(basicBlocks, function(elm){
            var tmp = self.splitToCanonicalBasicBlock(elm);
            arrOfCanonicalBasicBlock = arrOfCanonicalBasicBlock.concat(tmp);
        });
            
        result[code[rng.start].text] = {
            nodes:this.makeNodes(code, arrOfCanonicalBasicBlock),
            edges: this.makeEdges(code, arrOfCanonicalBasicBlock,
                this.rules.getInstructionType,
                this.rules.extractNodeName)
            };
        }, this));
        
    return result;
    
};

module.exports.ControlFlowGraph = ControlFlowGraph;