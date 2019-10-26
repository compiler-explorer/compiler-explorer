const path = require('path'),
    webpack = require('webpack'),
    CopyWebpackPlugin = require('copy-webpack-plugin'),
    MiniCssExtractPlugin = require('mini-css-extract-plugin'),
    ManifestPlugin = require('webpack-manifest-plugin'),
    glob = require("glob"),
    UglifyJsPlugin = require('uglifyjs-webpack-plugin'),
    MonacoEditorWebpackPlugin = require('monaco-editor-webpack-plugin');

const isDev = process.env.NODE_ENV !== "production";

const outputPathRelative = 'dist/';
const staticRelative = 'static/';
const staticPath = path.resolve(__dirname, staticRelative);
const distPath = path.join(staticPath, outputPathRelative);
const assetPath = path.join(staticPath, "assets");
const manifestPath = 'manifest.json';  //if you change this, you also need to update it in the app.js
const outputName = isDev ? '[name].js' : '[name].[chunkhash].js';
const cssName = isDev ? '[name].css' : "[name].[contenthash].css";
const publicPath = isDev ? '/dist/' : 'dist/';
const manifestPlugin = new ManifestPlugin({
    fileName: manifestPath,
    publicPath: './'
});


const assetEntries = glob.sync(`${assetPath}/**/*.*`).reduce((obj, p) => {
    const key = path.basename(p);
    obj[key] = p;
    return obj;
}, {});

let plugins = [
    new MonacoEditorWebpackPlugin({
        languages: ['cpp', 'go', 'rust', 'swift']
    }),
    new CopyWebpackPlugin([
        {
            from: path.join(staticPath, "favicon.ico"),
            to: distPath
        },
        {
            from: 'node_modules/es6-shim/es6-shim.min.js',
            to: distPath
        }
    ]),
    new webpack.ProvidePlugin({
        $: 'jquery',
        jQuery: 'jquery'
    }),
    new MiniCssExtractPlugin({
        filename: cssName

    }),
    manifestPlugin
];

module.exports = [
    //if you change the order of this, make sure to update the config variable in app.js
    //server side stuff
    // currently we just want to shove a cache path onto the static assets so we can get the hash onto the filename
    //this means we can set the cache for eternity
    {
        mode: isDev ? 'development' : 'production',
        entry: assetEntries,
        output: {
            path: path.join(distPath, 'assets'),
            filename: '.[name].ignoreme'
        },
        module: {
            rules: [
                {
                    test: /\.(png|woff|woff2|eot|ttf|svg)$/,
                    use: [{
                        loader: 'file-loader',
                        options: {
                            name: '[name].[hash].[ext]'
                        }
                    }]
                }
            ]
        },
        plugins: [
            manifestPlugin
        ]
    },
    //this is the client side
    {
        mode: isDev ? 'development' : 'production',
        entry: './static/main.js',
        output: {
            filename: outputName,
            path: distPath,
            publicPath: publicPath
        },
        resolve: {
            modules: ['./static', "./node_modules"],
            alias: {
                //is this safe?
                goldenlayout: path.resolve(__dirname, 'node_modules/golden-layout/'),
                lzstring: path.resolve(__dirname, 'node_modules/lz-string/'),
                filesaver: path.resolve(__dirname, 'node_modules/file-saver/')
            }
        },
        stats: "normal",
        devtool: 'source-map',
        optimization: {
            minimize: !isDev,
            minimizer: [new UglifyJsPlugin({
                parallel: true,
                sourceMap: true,
                uglifyOptions: {
                    output: {
                        comments: false,
                        beautify: false
                    }
                }
            })]
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
    }

];
