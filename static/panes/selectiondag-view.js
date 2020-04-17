// Copyright (c) 2020, Dan Shechter
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

'use strict';

var $ = require('jquery');
var _ = require('underscore');
var saveAs = require('file-saver').saveAs;
var Toggles = require('../toggles');
var ga = require('../analytics');
var Alert = require('../alert');
var d3 = require('d3');
require('d3-graphviz');

require('selectize');

function SelectionDAG(hub, container, state) {
    this.container = container;
    this.eventHub = hub.createEventHub();
    this.domRoot = container.getElement();
    this.domRoot.html($('#selection-dag').html());
    this.defaultCfgOutput = {nodes: [{id: 0, shape: 'box', label: 'No Output'}], edges: []};
    this.alertSystem = new Alert();
    this.alertSystem.prefixMessage = "Load-Saver: ";
    this.binaryModeSupport = {
        nodes: [{
            id: 0,
            shape: 'box',
            label: 'Cfg mode cannot be used when the binary filter is set'
        }], edges: []
    };
    // Note that this might be outdated if no functions were present when creating the link, but that's handled
    // by selectize
    state.options = state.options || {};
    this.savedPos = state.pos;
    this.savedScale = state.scale;
    this.needsMove = this.savedPos && this.savedScale;

    this.currentFunc = state.selectedFn || '';
    this.currentDag = state.selectedDag || 'dag-combine1';
    this.functions = [];
    
    this.initButtons(state);

    this.compilerId = state.id;
    this._editorid = state.editorid;
    this._binaryFilter = false;

    this.functionPicker = $(this.domRoot).find('.function-picker').selectize({
        sortField: 'name',
        valueField: 'name',
        labelField: 'name',
        searchField: ['name'],
        dropdownParent: 'body'
    }).on('change', _.bind(function (e) {
        var selectedFn = this.dags[e.target.value];
        if (selectedFn) {
            this.currentFunc = e.target.value;
            this.updateButtons();
            this.showDAGResults(this.currentFunc, this.currentDag);
            this.resize();
            this.saveState();
        }
    }, this));

    this.initCallbacks();
    this.adaptStructure = function (names) {
        return _.map(names, function (name) {
            return {name: name};
        });
    };
    this.updateButtons();
    this.setTitle();
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'SelectionDAG'
    });
}

SelectionDAG.prototype.onCompileResult = function (id, compiler, result) {
    if (this.compilerId === id) {
        var functionNames = [];
        if (this.supportsSelectionDAG && !$.isEmptyObject(result.dag)) {
            this.dags = result.dag.dags;
            functionNames = Object.keys(this.dags);
            if (functionNames.indexOf(this.currentFunc) === -1) {
                this.currentFunc = functionNames[0];
            }
            this.showDAGResults(this.currentFunc, this.currentDag);
            this.updateButtons();
        } else {
            // We don't reset the current function here as we would lose the saved one if this happened at the beginning
            // (Hint: It *does* happen)
            //this.showDAGResults(this._binaryFilter ? this.binaryModeSupport : this.defaultCfgOutput);
        }

        this.functionPicker[0].selectize.clearOptions();
        this.functionPicker[0].selectize.addOption(functionNames.length ?
            this.adaptStructure(functionNames) : {name: 'The input does not contain functions'});
        this.functionPicker[0].selectize.refreshOptions(false);

        this.functionPicker[0].selectize.clear();
        this.functionPicker[0].selectize.addItem(functionNames.length ?
            this.currentFunc : 'The input does not contain any function', true);
        this.saveState();
    }
};

SelectionDAG.prototype.onCompiler = function (id, compiler) {
    if (id === this.compilerId) {
        this._compilerName = compiler ? compiler.name : '';
        this.supportsSelectionDAG = compiler.supportsSelectionDAG;
        this.setTitle();
    }
};

SelectionDAG.prototype.onFiltersChange = function (id, filters) {
    if (this.compilerId === id) {
        this._binaryFilter = filters.binary;
    }
};

SelectionDAG.prototype.initButtons = function (state) {
    this.toggles = new Toggles(this.domRoot.find('.options'), state.options);

    this.exportDAGSVGButton = this.domRoot.find('.export-dag-svg')[0];
    this.exportDAGPNGButton = this.domRoot.find('.export-dag-png')[0];
    
    this.dagButtons = {
        'dag-combine1': undefined,
        'legalize-types': undefined,
        'dag-combine-lt': undefined,
        legalize: undefined,
        'dag-combine2': undefined,
        isel: undefined,
        scheduler: undefined,
        sunit: undefined,
    };
    
    for (var k in this.dagButtons) {
        this.dagButtons[k] = this.domRoot.find('.' + k)[0];
    }
};

SelectionDAG.prototype.initCallbacks = function () {
    this.eventHub.on('compilerClose', this.onCompilerClose, this);
    this.eventHub.on('compileResult', this.onCompileResult, this);
    this.eventHub.on('compiler', this.onCompiler, this);
    this.eventHub.on('filtersChange', this.onFiltersChange, this);

    this.container.on('destroy', this.close, this);
    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);
    this.eventHub.emit('selectionDAGViewOpened', this.compilerId);
    this.eventHub.emit('requestFilters', this.compilerId);
    this.eventHub.emit('requestCompiler', this.compilerId);
    
    var self = this;
    
    function registerButtonClick(button, dagName) {
        button.onclick = function () {
            self.showDAGResults(self.currentFunc, dagName);
        };
    }

    for (var k in this.dagButtons) {
        registerButtonClick(this.dagButtons[k], k);
    }
    
    this.exportDAGSVGButton.onclick = _.bind(function () {
        this.saveDAGAsSVG();
    }, this);

    this.exportDAGPNGButton.onclick = _.bind(function () {
        this.saveDAGAsPNG();
    }, this);
};

SelectionDAG.prototype.updateButtons = function () {
    Object.values(this.dagButtons).forEach(function (button) {
        button.disabled = true;
    });

    if (!this.currentFunc)
        return;
    
    var dagButtons = this.dagButtons;

    this.dags[this.currentFunc].forEach(function (x) { 
        console.log(x.dagType);
        dagButtons[x.dagType].disabled = false;
    });
};

SelectionDAG.prototype.resize = function () {
};

SelectionDAG.prototype.setTitle = function () {
    this.container.setTitle(
        this._compilerName + 
        ' SelectionDAG Viewer (Editor #' + this._editorid + ', Compiler #' + this.compilerId + ')');
};


SelectionDAG.prototype.getSVGString = function (svgNode) {
    svgNode.setAttribute('xlink', 'http://www.w3.org/1999/xlink');
    var cssStyleText = getCSSStyles(svgNode);
    appendCSS( cssStyleText, svgNode );

    var serializer = new XMLSerializer();
    var svgString = serializer.serializeToString(svgNode);
    svgString = svgString.replace(/(\w+)?:?xlink=/g, 'xmlns:xlink='); // Fix root xlink without namespace
    svgString = svgString.replace(/NS\d+:href/g, 'xlink:href'); // Safari NS namespace fix

    return svgString;

    function getCSSStyles( parentElement ) {
        var i;
        var c;
        var selectorTextArr = [];

        // Add Parent element Id and Classes to the list
        selectorTextArr.push( '#'+parentElement.id );
        for (c = 0; c < parentElement.classList.length; c++)
            if ( !contains('.'+parentElement.classList[c], selectorTextArr) )
                selectorTextArr.push( '.'+parentElement.classList[c] );

        // Add Children element Ids and Classes to the list
        var nodes = parentElement.getElementsByTagName("*");
        for (i = 0; i < nodes.length; i++) {
            var id = nodes[i].id;
            if ( !contains('#'+id, selectorTextArr) )
                selectorTextArr.push( '#'+id );

            var classes = nodes[i].classList;
            for (c = 0; c < classes.length; c++)
                if ( !contains('.'+classes[c], selectorTextArr) )
                    selectorTextArr.push( '.'+classes[c] );
        }

        // Extract CSS Rules
        var extractedCSSText = "";
        for (i = 0; i < document.styleSheets.length; i++) {
            var s = document.styleSheets[i];

            try {
                if(!s.cssRules) continue;
            } catch( e ) {
                if(e.name !== 'SecurityError') throw e; // for Firefox
                continue;
            }

            var cssRules = s.cssRules;
            for (var r = 0; r < cssRules.length; r++) {
                if ( contains( cssRules[r].selectorText, selectorTextArr ) )
                    extractedCSSText += cssRules[r].cssText;
            }
        }


        return extractedCSSText;

        function contains(str,arr) {
            return arr.indexOf(str) !== -1;
        }

    }

    function appendCSS( cssText, element ) {
        var styleElement = document.createElement("style");
        styleElement.setAttribute("type","text/css");
        styleElement.innerHTML = cssText;
        var refNode = element.hasChildNodes() ? element.children[0] : null;
        element.insertBefore( styleElement, refNode );
    }
};

SelectionDAG.prototype.saveDAGAsSVG = function () {
    try {
        var doctype = 
            '<?xml version="1.0" standalone="no"?>' + 
            '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd" ' +
            '[<!ENTITY nbsp "&#160;">]>';
        var name = this.currentFunc + "-" + this.currentDag;
        var graphSvg = this.placeholder.select("svg");
        var svgContents = graphSvg 
            .attr("title", name)
            .attr("version", 1.1)
            .attr("xmlns", "http://www.w3.org/2000/svg")
            .attr("xmlns:xhtml", "http://www.w3.org/1999/xhtml")
            .attr("xmlns:xlink", "http-www.w3.org/1999/xlink")
            .node().parentNode.innerHTML;

        saveAs(new Blob(
            [doctype + svgContents],
            {type: "image/svg+xml"}),
        name + ".svg");
        return true;
    } catch (e) {
        this.alertSystem.notify('Error while saving your selection DAG. Use the clipboard instead.', {
            group: "savelocalerror",
            alertClass: "notification-error",
            dismissTime: 5000
        });
        return false;
    }
};

SelectionDAG.prototype.saveDAGAsPNG = function () {
    try {
        var name = this.currentFunc + "-" + this.currentDag;
        var graphSvg = this.placeholder.select("svg");
        
        // let svgContents = graphSvg
        //     .attr("title", name)
        //     .attr("version", 1.1)
        //     .attr("xmlns", "http://www.w3.org/2000/svg")
        //     .attr("xmlns:xhtml", "http://www.w3.org/1999/xhtml")
        //     .attr("xmlns:xlink", "http-www.w3.org/1999/xlink")
        //     .node().parentNode.innerHTML;

        var svgContents = this.getSVGString(graphSvg.node());
        var width = graphSvg.attr('width').replace('pt', '');
        var height = graphSvg.attr('height').replace('pt', '');
        
        this.SvgToImage(svgContents, 2*width, 2*height, 'png', function (dataBlob) {
            saveAs(dataBlob, name + ".png");
        });         
        return true;
    } catch (e) {
        this.alertSystem.notify('Error while saving your selection DAG. Use the clipboard instead.', {
            group: "savelocalerror",
            alertClass: "notification-error",
            dismissTime: 5000
        });
        return false;
    }
};

SelectionDAG.prototype.SvgToImage =  function (svgString, width, height, fmt, callback ) {
    var imgSrc = 'data:image/svg+xml;base64,' + 
        btoa(unescape(encodeURIComponent(svgString))); // Convert SVG string to data URL

    var canvas = document.createElement("canvas");
    var context = canvas.getContext("2d");

    canvas.width = width;
    canvas.height = height;

    var image = new Image();
    image.onload = function () {
        context.clearRect ( 0, 0, width, height );
        context.drawImage(image, 0, 0, width, height);

        canvas.toBlob( function (blob) {
            var fileSize = Math.round(blob.length / 1024) + ' KB';
            if ( callback ) callback( blob, fileSize );
        });


    };

    image.src = imgSrc;
};

SelectionDAG.prototype.showDAGResults = function (func, dag) {
    if (!func || !dag) {
        return;
    }
    var result = this.dags[func].find(function (o) { return o.dagType === dag; });
    if (!result)
        return;
    var dotText = result.graphvizData;
    this.placeholder = d3.select('.graphviz-placeholder');
    this.graphviz = this.placeholder
        .graphviz({useWorker: false})
        .tweenPrecision('50%')
        .transition(function () {
            return d3.transition()
                .ease(d3.easeLinear)
                .delay(1000)
                .duration(500);
        });
    //.logEvents(true);

    this.graphviz
        .attributer(_.partial(this.graphvizAttributer, this))
        .tweenShapes(false)
        .renderDot(dotText, _.bind(this.dagViewerApp, this));
};

SelectionDAG.prototype.graphvizAttributer = function (selectionDAG, datum) {
    var selection = d3.select(this);
    if (datum.tag === "path" && datum.parent.attributes.class === 'node') {
        selection.attr("fill", "white");
        datum.attributes.fill = "white";         
    }

    if (datum.tag === "text") {
        selection.attr("font-family", "Consolas, \"Liberation Mono\", Courier, monospace");
        datum.attributes["font-family"] = "Consolas, \"Liberation Mono\", Courier, monospace";
        selection.attr("font-size", "0.6em");
        datum.attributes["font-size"] = "0.6em";
    }
    
    if (datum.tag === "polygon" && datum.parent.attributes.class === 'graph') {
        selection.attr("fill", "none");
        datum.attributes.fill = "none";
    }
    
    return;

    /*
    var leftPad = 20;
    var rightPad = 20;
    var topPad = 20;
    var bottomPad = 20;
    var margin = 20; // to avoid scrollbars
    
    if (datum.tag === "svg") {
        var rect = selectionDAG.placeholder.node().getBoundingClientRect();
        var svgWidth = rect.width - margin;
        var svgHeight = rect.height - margin;

        var graphWidth = +datum.attributes.width.replace('pt', '');
        var graphHeight = +datum.attributes.height.replace('pt', '');
        selectionDAG.graphviz.zoomTranslateExtent(
            [[rightPad + graphWidth - svgWidth, bottomPad - svgHeight], 
                [svgWidth - leftPad, svgHeight - topPad - graphHeight]]);
        selection
            .attr("width", svgWidth)
            .attr("height", svgHeight)
            .attr("viewBox", "0 0 " + svgWidth + " " + svgHeight);
        datum.attributes.width = svgWidth;
        datum.attributes.height = svgHeight;
        datum.attributes.viewBox = " 0 0 " + svgWidth + " " + svgHeight;
    }    
     */
};


SelectionDAG.prototype.dagViewerApp = function () {
    console.log("Hooking up app...");

    this.highlightedNode = null;
    this.highlightedSelection = null;
    this.highlightedEdges = null;

    var nodes = this.placeholder.selectAll(".node");

    d3.select(document).on("mouseover", _.partial(function (selectionDAG) {
        var event = d3.event;
        
        if (event.target.nodeName !== 'svg') {// && event.target.parentElement.id != 'graph0') {
            return;
        }
        event.preventDefault();
        event.stopPropagation();
        selectionDAG.unSelectNode();
    }, this));


    // click and mousedown on nodes
    nodes.on("mouseover", _.partial(function (selectionDAG) {
        var event = d3.event;
        event.preventDefault();
        event.stopPropagation();
        selectionDAG.selectNode(d3.select(this));
    }, this));
};

SelectionDAG.prototype.selectNode = function (selection) {
    var node = selection.node();
    if (this.highlightedNode === node)
        return;

    this.unSelectNode();
    this.highlightedNode = node;
    this.highlightedSelection = selection;
    this.originalStroke = selection.selectAll('path').attr("stroke");
    selection.selectAll('path, polyline').attr("stroke", "red");

    var id = this.highlightedSelection.selectAll("title").text();

    this.highlightedEdges = d3.selectAll(".edge").filter(function (e) {
        return e.key.startsWith(id);
    });
    this.highlightedEdges.selectAll("path, polygon").attr("stroke", "red");
    this.highlightedEdges.selectAll("polygon").attr("fill", "red");
    this.highlightedEdges.selectAll("polygon").attr("fill-opacity", "0.4");
};

SelectionDAG.prototype.unSelectNode = function ()
{
    if (this.highlightedNode == null)
        return;

    this.highlightedSelection.selectAll('path, polyline').attr("stroke", this.originalStroke);
    this.highlightedEdges.selectAll('path, polygon').each(function (e) {
        d3.select(this).attr("stroke",e.attributes.stroke);
    });
    this.highlightedEdges.selectAll('polygon').each(function (e) {
        d3.select(this).attr("fill", e.attributes.fill);
        d3.select(this).attr("fill-opacity", "1.0");
    });
    this.highlightedNode = null;
};

SelectionDAG.prototype.onCompilerClose = function (compilerId) {
    if (this.compilerId === compilerId) {
        // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
        // the hierarchy. We can't modify while it's being iterated over.
        this.close();
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

SelectionDAG.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit('selectionDAGViewClosed', this.compilerId);
    //this.cfgVisualiser.destroy();
};

SelectionDAG.prototype.saveState = function () {
    this.container.setState(this.currentState());
};

SelectionDAG.prototype.getEffectiveOptions = function () {
    return this.toggles.get();
};

SelectionDAG.prototype.currentState = function () {
    return {
        id: this.compilerId,
        editorid: this._editorid,
        selectedFn: this.currentFunc,
        selectedDag: this.currentDag,
    };
};

module.exports = {
    SelectionDAG: SelectionDAG
};
