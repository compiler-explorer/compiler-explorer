const path = require('path'),
    CopyWebpackPlugin = require('copy-webpack-plugin'),
    webpack = require('webpack'),
    ExtractTextPlugin = require('extract-text-webpack-plugin');

const outputPathRelative = 'dist/';
const staticRelative = 'static/';
const staticPath = path.resolve(__dirname, staticRelative);
const distPath = path.join(staticPath, outputPathRelative);
const vsPath = path.join(staticPath, 'vs/');


module.exports = {
    entry: './static/main.js',
    output: {
        filename: 'bundle.js',
        path: distPath,
        publicPath: "/static/dist/",
    },
    resolve: {
        modules: ['./static', "./node_modules"],
        alias: {
            //is this safe?
            goldenlayout:  path.resolve(__dirname, 'node_modules/golden-layout/'),
            lzstring:  path.resolve(__dirname, 'node_modules/lz-string/'),
            filesaver:  path.resolve(__dirname, 'node_modules/file-saver/'),
            vs: path.resolve(__dirname, 'node_modules/monaco-editor/min/vs')
        }
    },
    stats: "errors-only",
    devtool: 'source-map',
    module: {
    rules: [
        {
            test: /\.css$/,
            exclude: path.resolve(__dirname, 'static/themes/'),
            use: ExtractTextPlugin.extract({
                fallback: "style-loader",
                use: "css-loader"
            })
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
            test: /\.pug$/,
            loader: 'pug-loader',
        }
    ]},
    plugins: [
        new CopyWebpackPlugin([
          {
            from: 'node_modules/monaco-editor/min/vs',
            to: vsPath,
          }
        ]),
        new webpack.ProvidePlugin({
            $: 'jquery',
            jQuery: 'jquery'
          }),
        new ExtractTextPlugin("styles.css"),
    ]
};