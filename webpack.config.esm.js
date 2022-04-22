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

import path from 'path';
import {fileURLToPath} from 'url';

/* eslint-disable node/no-unpublished-import */
import CssMinimizerPlugin from 'css-minimizer-webpack-plugin';
import MiniCssExtractPlugin from 'mini-css-extract-plugin';
import MonacoEditorWebpackPlugin from 'monaco-editor-webpack-plugin';
import TerserPlugin from 'terser-webpack-plugin';
import {DefinePlugin, HotModuleReplacementPlugin, ProvidePlugin} from 'webpack';
import {WebpackManifestPlugin} from 'webpack-manifest-plugin';

const __dirname = path.resolve(path.dirname(fileURLToPath(import.meta.url)));
const isDev = process.env.NODE_ENV !== 'production';
console.log(`webpack config for ${isDev ? 'development' : 'production'}.`);

const distPath = path.resolve(__dirname, 'out', 'dist');
const staticPath = path.join(distPath, 'static');

// Hack alert: due to a variety of issues, sometimes we need to change
// the name here. Mostly it's things like webpack changes that affect
// how minification is done, even though that's supposed not to matter.
const webjackJsHack = '.v7.';
const plugins = [
    new MonacoEditorWebpackPlugin({
        languages: [
            'cpp',
            'go',
            'pascal',
            'python',
            'rust',
            'swift',
            'java',
            'kotlin',
            'scala',
            'ruby',
            'csharp',
            'fsharp',
            'vb',
            'dart',
            'typescript',
            'solidity',
        ],
        filename: isDev ? '[name].worker.js' : `[name]${webjackJsHack}worker.[contenthash].js`,
    }),
    new ProvidePlugin({
        $: 'jquery',
        jQuery: 'jquery',
    }),
    new MiniCssExtractPlugin({
        filename: isDev ? '[name].css' : `[name]${webjackJsHack}[contenthash].css`,
    }),
    new WebpackManifestPlugin({
        fileName: path.join(distPath, 'manifest.json'),
        publicPath: '',
    }),
    new DefinePlugin({
        'window.PRODUCTION': JSON.stringify(!isDev),
    }),
];

if (isDev) {
    plugins.push(new HotModuleReplacementPlugin());
}

// eslint-disable-next-line import/no-default-export
export default {
    mode: isDev ? 'development' : 'production',
    entry: {
        main: './static/main.js',
        noscript: './static/noscript.ts',
    },
    output: {
        filename: isDev ? '[name].js' : `[name]${webjackJsHack}[contenthash].js`,
        path: staticPath,
    },
    resolve: {
        alias: {
            'monaco-editor$': 'monaco-editor/esm/vs/editor/editor.api',
        },
        fallback: {
            path: 'path-browserify',
        },
        modules: ['./static', './node_modules'],
        extensions: ['.tsx', '.ts', '.js'],
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
                test: /\.(html)$/,
                loader: 'html-loader',
            },
            {
                test: /\.tsx?$/,
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
