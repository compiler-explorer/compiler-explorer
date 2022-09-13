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

import $ from 'jquery';
import {Settings} from '../settings';
import {Chart, ChartData, defaults} from 'chart.js';
import 'chart.js/auto';

type Data = ChartData<'bar', number[], string> & {steps: number};

function pushTimingInfo(data: Data, step: string, time: number | string) {
    if (typeof time === 'string') {
        time = parseInt(time, 10);
    }
    data.labels?.push(`${step} (${time}ms)`);
    data.datasets[0].data.push(time);
    data.steps += time;
}

function concatTimings(data: Data, timings: {step: string; time: number}[]) {
    for (const timing of timings) {
        pushTimingInfo(data, timing.step, timing.time);
    }
}

function addBuildResultToTimings(data: Data, buildResult: any) {
    if (buildResult.packageDownloadAndUnzipTime) {
        pushTimingInfo(data, 'Download binary from cache', buildResult.packageDownloadAndUnzipTime);
    } else {
        if (buildResult.downloads) {
            concatTimings(data, buildResult.downloads);
        }

        if (buildResult.buildsteps) {
            for (const step of buildResult.buildsteps) {
                pushTimingInfo(data, step.step, step.execTime);
            }
        } else if (buildResult.execTime) {
            pushTimingInfo(data, 'Compilation', buildResult.execTime);
        }
    }
}

function initializeChartDataFromResult(compileResult: any, totalTime: number): Data {
    const data: Data = {
        steps: 0,
        labels: [],
        datasets: [
            {
                label: 'time in ms',
                data: [],
                borderWidth: 1,
                barThickness: 20,
                backgroundColor: ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],
            },
        ],
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
        if (compileResult.execResult && compileResult.execResult.execTime) {
            pushTimingInfo(data, 'Execution', compileResult.execResult.execTime);
        } else {
            pushTimingInfo(data, 'Execution', compileResult.execTime);
        }
    }
    const stepsTotal = data.steps;
    pushTimingInfo(data, 'Network, JS, waiting, etc.', totalTime - stepsTotal);

    if (totalTime - stepsTotal < 0) {
        data.datasets[0].data = [totalTime];
        data.labels = ['Browser cache'];
    }

    return data;
}

function displayData(data: Data) {
    const modal = $('#timing-info');

    const chartDiv = modal.find('#chart');
    chartDiv.html('');

    const canvas = $('<canvas id="timing-chart" width="400" height="400"></canvas>') as JQuery<HTMLCanvasElement>;
    chartDiv.append(canvas);

    const settings = Settings.getStoredSettings();
    let fontColour = defaults.color.toString();
    if (settings.theme !== 'default') {
        fontColour = '#ffffff';
    }

    new Chart(canvas, {
        type: 'bar',
        data: data,
        options: {
            scales: {
                xAxis: {
                    beginAtZero: true,
                    grid: {
                        color: fontColour,
                        tickColor: fontColour,
                    },
                    ticks: {color: fontColour},
                },
                yAxis: {
                    beginAtZero: true,
                    grid: {
                        color: fontColour,
                        tickColor: fontColour,
                    },
                    ticks: {color: fontColour},
                },
            },
            plugins: {
                legend: {
                    labels: {
                        color: fontColour,
                        font: {size: 18},
                    },
                },
            },
        },
    });
    modal.modal('show');
}

export function displayCompilationTiming(compileResult: any, totalTime: number) {
    const data = initializeChartDataFromResult(compileResult, totalTime);
    displayData(data);
}
