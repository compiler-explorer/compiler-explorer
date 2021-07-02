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
var Settings = require('settings');

// The name of the package itself contains .js, it's not referencing a file
// eslint-disable-next-line requirejs/no-js-extension
var Chart = require('chart.js');


function pushTimingInfo(data, step, time) {
    data.labels.push(step);
    data.datasets[0].data.push(time);
    data.steps += parseInt(time, 10);
}

function concatTimings(data, timings) {
    _.forEach(timings, function (timing) {
        pushTimingInfo(data, timing.step, timing.time);
    });
}

function addBuildResultToTimings(data, buildResult) {
    if (buildResult.packageDownloadAndUnzipTime) {
        pushTimingInfo(data, 'Download binary from cache', buildResult.packageDownloadAndUnzipTime);
    } else {
        if (buildResult.downloads) {
            concatTimings(data, buildResult.downloads);
        }

        if (buildResult.buildsteps) {
            _.forEach(buildResult.buildsteps, function (step) {
                pushTimingInfo(data, step.step, step.execTime);
            });
        } else if (buildResult.execTime) {
            pushTimingInfo(data, 'Compilation', buildResult.execTime);
        }
    }
}

function initializeChartDataFromResult(compileResult, totalTime) {
    var data = {
        steps: 0,
        labels: [],
        datasets: [{
            label: 'time in ms',
            data: [],
            borderWidth: 1,
            barThickness: 20,
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

    if (compileResult.retreivedFromCache) {
        pushTimingInfo(data, 'Retrieve result from cache', compileResult.retreivedFromCacheTime);

        if (compileResult.packageDownloadAndUnzipTime) {
            pushTimingInfo(data, 'Download binary from cache', compileResult.execTime);
        }

        if (compileResult.execResult && compileResult.execResult.execTime) {
            pushTimingInfo(data, 'Execution', compileResult.execResult.execTime);
        }
    } else {
        addBuildResultToTimings(data, compileResult);

        if (!compileResult.packageDownloadAndUnzipTime) {
            if (compileResult.objdumpTime) {
                pushTimingInfo(data, 'Disassembly', compileResult.objdumpTime);
            }

            if (compileResult.parsingTime) {
                pushTimingInfo(data, 'ASM parsing', compileResult.parsingTime);
            }
        }
    }

    if (compileResult.didExecute) {
        if (compileResult.execResult.execTime) {
            pushTimingInfo(data, 'Execution', compileResult.execResult.execTime);
        } else {
            pushTimingInfo(data, 'Execution', compileResult.execTime);
        }
    }

    var stepsTotal = data.steps;
    pushTimingInfo(data, 'Network, JS, waiting, etc.', totalTime - stepsTotal);

    if (totalTime - stepsTotal < 0) {
        data.datasets[0].data = [totalTime];
        data.labels = ['Browser cache'];
    }

    delete data.steps;
    return data;
}

function displayData(data) {
    var modal = $('#timing-info');

    var chartDiv = modal.find('#chart');
    chartDiv.html('');

    var canvas = $('<canvas id="timing-chart" width="400" height="400"></canvas>');
    chartDiv.append(canvas);

    var settings = Settings.getStoredSettings();
    var fontColour = Chart.defaults.color;
    if (settings != null && settings.theme === 'dark') {
        fontColour = '#ffffff';
    }

    var ctx = canvas[0].getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            legend: {
                labels: {
                    fontColor: fontColour,
                    fontSize: 18,
                },
            },
            scales: {
                xAxes: [{
                    gridLines: {
                        color: fontColour,
                    },
                    ticks: {
                        fontColor: fontColour,
                        beginAtZero: true,
                    },
                }],
                yAxes: [{
                    gridLines: {
                        color: fontColour,
                    },
                    ticks: {
                        fontColor: fontColour,
                        beginAtZero: true,
                    },
                }],
            },
        },
    });
    modal.modal('show');
}

function displayCompilationTiming(compileResult, totalTime) {
    var data = initializeChartDataFromResult(compileResult, totalTime);
    displayData(data);
}

module.exports = {
    displayCompilationTiming: displayCompilationTiming,
};
