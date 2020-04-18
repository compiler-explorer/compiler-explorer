const path = require('path'),
    webpack = require('webpack'),
    CopyWebpackPlugin = require('copy-webpack-plugin'),
    MiniCssExtractPlugin = require('mini-css-extract-plugin'),
    ManifestPlugin = require('webpack-manifest-plugin'),
    TerserPlugin = require('terser-webpack-plugin'),
    MonacoEditorWebpackPlugin = require('monaco-editor-webpack-plugin');

const isDev = process.env.NODE_ENV !== "production";

const distPath = path.resolve(__dirname, 'out', 'dist');
const staticPath = path.join(distPath, 'static');

// Hack alert: due to a variety of issues, sometimes we need to change
// the name here. Mostly it's things like webpack changes that affect
// how minification is done, even though that's supposed not to matter.
const webjackJsHack = ".v2.";
const plugins = [
    new MonacoEditorWebpackPlugin({
        languages: ['cpp', 'go', 'rust', 'swift', 'pascal'],
        filename: isDev ? '[name].worker.js' : `[name]${webjackJsHack}worker.[contenthash].js`
    }),
    new CopyWebpackPlugin([
        {
            from: 'node_modules/es6-shim/es6-shim.min.js',
            to: staticPath
        }
    ]),
    new webpack.ProvidePlugin({
        $: 'jquery',
        jQuery: 'jquery'
    }),
    new MiniCssExtractPlugin({
        filename: isDev ? '[name].css' : '[name].[contenthash].css'
    }),
    new ManifestPlugin({
        fileName: path.join(distPath, 'manifest.json'),
        publicPath: ''
    })
];

module.exports = {
    mode: isDev ? 'development' : 'production',
    entry: './static/main.js',
    output: {
        filename: isDev ? '[name].js' : `[name]${webjackJsHack}[contenthash].js`,
        path: staticPath
    },
    resolve: {
        modules: ['./static', './node_modules']
    },
    stats: 'normal',
    devtool: 'source-map',
    optimization: {
        minimize: !isDev,
        minimizer: [
            new TerserPlugin({
                parallel: true,
                sourceMap: true,
                terserOptions: {
                    ecma: 5
                }
            })
        ]
    },
    module: {
        rules: [
            {
                test: /\.css$/,
                exclude: path.resolve(__dirname, 'static/themes/'),
                use: [
                    {
                        loader: MiniCssExtractPlugin.loader,
                        options: {
                            publicPath: './',
                            hmr: isDev
                        }
                    },
                    'css-loader'
                ]
            },
            {
                test: /\.css$/,
                include: path.resolve(__dirname, 'static/themes/'),
                use: ['css-loader']
            },
            {
                test: /\.(png|woff|woff2|eot|ttf|svg)$/,
                loader: 'url-loader?limit=8192'
            },
            {
                test: /\.(html)$/,
                loader: 'html-loader',
                options: {
                    minimize: !isDev
                }
            }
        ]
    },
    plugins: plugins
};
