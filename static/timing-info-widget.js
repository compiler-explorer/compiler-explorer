// Copyright (c) 2021, Compiler Explorer Authors
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
var Alert = require('./alert');

// eslint-disable-next-line
var Chart = require('chart.js');

function TimingInfo() {
    this.modal = null;
    this.alertSystem = new Alert();
    this.onLoad = _.identity;
    this.base = window.httpRoot;
    this.ctx = null;

    this.data = {
        labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
        datasets: [{
            label: 'time in ms',
            data: [12, 19, 3, 5, 2, 3],
            borderWidth: 1,
            backgroundColor: [
                'red',
                'orange',
                'yellow',
                'green',
                'blue',
                'indigo',
                'violet',
            ],
        }],
    };
}

TimingInfo.prototype.initializeChartDataFromResult = function (compileResult, totalTime) {
    var timings = [];

    this.data.labels = [];
    this.data.datasets[0].barThickness = 20;
    this.data.datasets[0].data = [];

    if (compileResult.retreivedFromCache) {
        timings.push({
            step: 'Retreive result from cache',
            time: compileResult.retreivedFromCacheTime,
        });

        if (compileResult.packageDownloadAndUnzipTime) {
            timings.push({
                step: 'Download binary from cache',
                time: compileResult.execTime,
            });
        }

        if (compileResult.execResult && compileResult.execResult.execTime) {
            timings.push({
                step: 'Execution',
                time: compileResult.execResult.execTime,
            });
        }
    } else {

        if (compileResult.packageDownloadAndUnzipTime) {
            timings.push({
                step: 'Download binary from cache',
                time: compileResult.execTime,
            });
        } else {

            if (compileResult.execResult) {
                if (compileResult.execResult.buildResult) {
                    if (compileResult.execResult.buildResult.packageDownloadAndUnzipTime) {
                        timings.push({
                            step: 'Download binary from cache',
                            time: compileResult.execResult.buildResult.packageDownloadAndUnzipTime,
                        });
                    } else {
                        if (compileResult.execResult.buildResult.downloads) {
                            timings = timings.concat(compileResult.execResult.buildResult.downloads);
                        }

                        if (compileResult.execResult.buildResult.execTime) {
                            timings.push({
                                step: 'Compilation',
                                time: compileResult.execResult.buildResult.execTime,
                            });
                        }
                    }
                }

                if (compileResult.objdumpTime) {
                    timings.push({
                        step: 'Disassembly',
                        time: compileResult.objdumpTime,
                    });
                }

                if (compileResult.parsingTime) {
                    timings.push({
                        step: 'ASM parsing',
                        time: compileResult.parsingTime,
                    });
                }

                if (compileResult.execResult.execTime) {
                    timings.push({
                        step: 'Execution',
                        time: compileResult.execResult.execTime,
                    });
                }

            } else {
                if (compileResult.downloads) {
                    timings = timings.concat(compileResult.downloads);
                }

                if (!compileResult.didExecute && compileResult.execTime) {
                    timings.push({
                        step: 'Compilation',
                        time: compileResult.execTime,
                    });
                }

                if (compileResult.objdumpTime) {
                    timings.push({
                        step: 'Disassembly',
                        time: compileResult.objdumpTime,
                    });
                }

                if (compileResult.parsingTime) {
                    timings.push({
                        step: 'ASM parsing',
                        time: compileResult.parsingTime,
                    });
                }
            }
        }
    }

    if (compileResult.didExecute) {
        if (compileResult.buildResult) {
            if (compileResult.buildResult.packageDownloadAndUnzipTime) {
                timings.push({
                    step: 'Download binary from cache',
                    time: compileResult.buildResult.packageDownloadAndUnzipTime,
                });
            } else {
                if (compileResult.buildResult.downloads) {
                    timings = timings.concat(compileResult.buildResult.downloads);
                }

                if (compileResult.buildResult.execTime) {
                    timings.push({
                        step: 'Compilation',
                        time: compileResult.buildResult.execTime,
                    });
                }
            }
        }

        timings.push({
            step: 'Execution',
            time: compileResult.execTime,
        });
    }

    var stepsTotal = 0;
    timings.forEach(_.bind(function (timing) {
        this.data.labels.push(timing.step);
        this.data.datasets[0].data.push(timing.time);

        stepsTotal += parseInt(timing.time, 10);
    }, this));

    this.data.labels.push('Network, JS, waiting, etc.');
    this.data.datasets[0].data.push(totalTime - stepsTotal);

    if (totalTime - stepsTotal < 0) {
        this.data.datasets[0].data = [totalTime];
        this.data.labels = ['Browser cache'];
    }
};

TimingInfo.prototype.initializeIfNeeded = function () {
    if ((this.modal === null) || (this.modal.length === 0)) {
        this.modal = $('#timing-info');
    }

    var chartDiv = this.modal.find('#chart');
    chartDiv.html('');

    var canvas = $('<canvas id="timing-chart" width="400" height="400"></canvas>');
    chartDiv.append(canvas);

    this.ctx = canvas[0].getContext('2d');
    this.chart = new Chart(this.ctx, {
        type: 'bar',
        data: this.data,
        options: {
            scales: {
                xAxes: [{
                    ticks: {
                        beginAtZero: true,
                    },
                }],
            },
        },
    });
};

TimingInfo.prototype.run = function (onLoad, compileResult, totalTime) {
    this.initializeChartDataFromResult(compileResult, totalTime);
    this.initializeIfNeeded();
    this.modal.modal('show');
};

module.exports = { TimingInfo: TimingInfo };
