// Copyright (c) 2020, Compiler Explorer Authors
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

import fs from 'fs';
import os from 'os';
import path from 'path';
import {fileURLToPath} from 'url';

/* eslint-disable node/no-unpublished-import */
import CopyWebpackPlugin from 'copy-webpack-plugin';
import CssMinimizerPlugin from 'css-minimizer-webpack-plugin';
import MiniCssExtractPlugin from 'mini-css-extract-plugin';
import MonacoEditorWebpackPlugin from 'monaco-editor-webpack-plugin';
import TerserPlugin from 'terser-webpack-plugin';
import Webpack from 'webpack';
import {WebpackManifestPlugin} from 'webpack-manifest-plugin';

const __dirname = path.resolve(path.dirname(fileURLToPath(import.meta.url)));
const isDev = process.env.NODE_ENV !== 'production';

function log(message) {
    // eslint-disable-next-line no-console
    console.log('webpack: ' + message);
}

log(`compiling for ${isDev ? 'development' : 'production'}.`);
// Memory limits us in most cases, so restrict parallelism to keep us in a sane amount of RAM
const parallelism = Math.floor(os.totalmem() / (4 * 1024 * 1024 * 1024)) + 1;
log(`Limiting parallelism to ${parallelism}`);

const distPath = path.resolve(__dirname, 'out', 'dist');
const staticPath = path.resolve(__dirname, 'out', 'webpack', 'static');
const hasGit = fs.existsSync(path.resolve(__dirname, '.git'));

// Hack alert: due to a variety of issues, sometimes we need to change
// the name here. Mostly it's things like webpack changes that affect
// how minification is done, even though that's supposed not to matter.
const webpackJsHack = '.v27.';
const plugins: Webpack.WebpackPluginInstance[] = [
    new MonacoEditorWebpackPlugin({
        languages: [
            'cpp',
            'go',
            'pascal',
            'python',
            'rust',
            'swift',
            'java',
            'julia',
            'kotlin',
            'scala',
            'ruby',
            'csharp',
            'fsharp',
            'vb',
            'dart',
            'typescript',
            'solidity',
            'scheme',
            'objective-c',
        ],
        filename: isDev ? '[name].worker.js' : `[name]${webpackJsHack}worker.[contenthash].js`,
    }),
    new MiniCssExtractPlugin({
        filename: isDev ? '[name].css' : `[name]${webpackJsHack}[contenthash].css`,
    }),
    new WebpackManifestPlugin({
        fileName: path.resolve(distPath, 'manifest.json'),
        publicPath: '',
    }),
    new Webpack.DefinePlugin({
        'window.PRODUCTION': JSON.stringify(!isDev),
    }),
    new CopyWebpackPlugin({
        patterns: [{from: './static/favicons', to: path.resolve(distPath, 'static', 'favicons')}],
    }),
];

if (isDev) {
    plugins.push(new Webpack.HotModuleReplacementPlugin());
}

// eslint-disable-next-line import/no-default-export
export default {
    mode: isDev ? 'development' : 'production',
    entry: {
        main: './static/main.ts',
        noscript: './static/noscript.ts',
    },
    output: {
        filename: isDev ? '[name].js' : `[name]${webpackJsHack}[contenthash].js`,
        path: staticPath,
    },
    cache: {
        type: 'filesystem',
        buildDependencies: {
            config: [fileURLToPath(import.meta.url)],
        },
    },
    resolve: {
        alias: {
            'monaco-editor$': 'monaco-editor/esm/vs/editor/editor.api',
        },
        fallback: {
            path: 'path-browserify',
        },
        modules: ['./static', './node_modules'],
        extensions: ['.ts', '.js'],
        extensionAlias: {
            '.js': ['.ts', '.js'],
            '.mjs': ['.mts', '.mjs'],
        },
    },
    stats: 'normal',
    devtool: 'source-map',
    optimization: {
        runtimeChunk: 'single',
        splitChunks: {
            cacheGroups: {
                vendors: {
                    test: /[/\\]node_modules[/\\]/,
                    name: 'vendor',
                    chunks: 'all',
                    priority: -10,
                },
            },
        },
        moduleIds: 'deterministic',
        minimizer: [
            new CssMinimizerPlugin(),
            new TerserPlugin({
                parallel: true,
                terserOptions: {
                    ecma: 5,
                    sourceMap: true,
                },
            }),
        ],
    },
    parallelism: parallelism,
    module: {
        rules: [
            {
                test: /\.s?css$/,
                use: [
                    {
                        loader: MiniCssExtractPlugin.loader,
                        options: {
                            publicPath: './',
                        },
                    },
                    'css-loader',
                    'sass-loader',
                ],
            },
            {
                test: /\.(png|woff|woff2|eot|ttf|svg)$/,
                type: 'asset',
                parser: {dataUrlCondition: {maxSize: 8192}},
            },
            {
                test: /\.pug$/,
                loader: './etc/scripts/parsed-pug/parsed_pug_file.js',
                options: {
                    useGit: hasGit,
                },
            },
            {
                test: /\.ts$/,
                loader: 'ts-loader',
            },
            {
                test: /\.js$/,
                loader: 'source-map-loader',
            },
        ],
    },
    plugins: plugins,
};
