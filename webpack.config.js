const path = require('path'),
    webpack = require('webpack'),
    CopyWebpackPlugin = require('copy-webpack-plugin'),
    MiniCssExtractPlugin = require("mini-css-extract-plugin"),
    ManifestPlugin = require('webpack-manifest-plugin'),
    glob = require("glob"),
    TerserPlugin = require('terser-webpack-plugin');
    MonacoWebpackPlugin = require('monaco-editor-webpack-plugin');

const isDev = process.env.NODE_ENV  === "DEV";

const staticPath = path.resolve(__dirname, "static");
const assetPath = path.join(staticPath, "assets");
const distPath =  path.resolve(__dirname, 'dist')
const manifestPath = 'manifest.json';  //if you change this, you also need to update it in the app.js
const outputName = isDev ? 'main.js' : 'bundle.[hash].js';
const cssName = isDev ? '[name].css' :  "[name].[contenthash].css";
const publicPath = '/'
const manifestPlugin = new ManifestPlugin({
    fileName: manifestPath
});


const assetEntries = glob.sync(`${assetPath}/**/*.*`).reduce((obj, p) => {
    const key = path.basename(p);
    obj[key] = p;
    return obj;
}, {});

let plugins = [
    new CopyWebpackPlugin([{
        from: path.join(staticPath, "favicon.ico"),
        to: distPath,
    },
    ]),
    new webpack.ProvidePlugin({
        $: 'jquery',
        jQuery: 'jquery'
    }),
    new MiniCssExtractPlugin(cssName),
    new MonacoWebpackPlugin({
        languages: ['clojure', 'cpp', 'csharp', 'csp', 'fsharp', 'go', 'html', 'java', 'objective', 'perl', 'php', 'python', 'r', 'ruby', 'rust', 'scheme', 'swift'],

    }),
    manifestPlugin
];
const minimizer = []
if(!isDev) {
    minimizer.push(new TerserPlugin({
        parallel:true,
        sourceMap: true,
        terserOptions: {
            ecma: 8,
        }
    }));
}


module.exports = [
    // if you change the order of this, make sure to update the config variable in app.js
    // server side stuff
    // currently we just want to shove a cache path onto the static assets so we can get the hash onto the filename
    // this means we can set the cache for eternity
    {
        mode: isDev ? 'development' : 'production',
        entry: assetEntries,
        output: {
            path: path.join(distPath, 'assets'),
            filename: '.[name].ignoreme',
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
                },
            ],
        },
        plugins:[
            manifestPlugin
        ]
    },
    //this is the client side
    {
        entry: path.join(staticPath, 'main.js'),
        output: {
            filename: outputName,
            path: distPath,
            publicPath: publicPath
        },
        resolve: {
            modules: ['./static', "./node_modules"],
            alias: {
                //is this safe?
                goldenlayout:  path.resolve(__dirname, 'node_modules/golden-layout/'),
                lzstring:  path.resolve(__dirname, 'node_modules/lz-string/'),
                filesaver:  path.resolve(__dirname, 'node_modules/file-saver/')
            }
        },
        stats: "verbose",
        devtool: 'source-map',
        module: {
            rules: [
                {
                    test: /\.css$/,
                    exclude: path.join(staticPath, 'themes'),
                    use: [
                        {
                         loader: MiniCssExtractPlugin.loader,
                          options: {
                            publicPath: publicPath
                          }
                        },
                        "css-loader"
                      ]
                },
                {
                    test: /\.css$/,
                    include: path.join(staticPath, 'themes'),
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
            ]},
            optimization: {
                minimizer: minimizer
            },
            plugins: plugins
        },

    ];
