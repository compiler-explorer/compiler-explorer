const path = require('path'),
    CopyWebpackPlugin = require('copy-webpack-plugin'),
    webpack = require('webpack'),
    ExtractTextPlugin = require("extract-text-webpack-plugin");

const vsPath = path.resolve(__dirname, 'static/vs/')

module.exports = {
    entry: './static/main.js',
    output: {
        filename: 'bundle.js',
        path: path.resolve(__dirname, 'static/dist')
    },
    resolve: {
        modules: ['./static', "./node_modules"],
        alias: {
            //is this safe?
            goldenlayout:  path.resolve(__dirname, 'node_modules/golden-layout/'),
            lzstring:  path.resolve(__dirname, 'node_modules/lz-string/'),
            filesaver:  path.resolve(__dirname, 'node_modules/file-saver/'),
            vs: vsPath
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