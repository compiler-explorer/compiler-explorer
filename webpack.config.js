const path = require('path'),
    webpack = require('webpack'),    
    glob = require("glob"),
    CopyWebpackPlugin = require('copy-webpack-plugin'),
    ExtractTextPlugin = require('extract-text-webpack-plugin'),
    ManifestPlugin = require('webpack-manifest-plugin'),
    UglifyJsPlugin = require('uglifyjs-webpack-plugin');

const isDev = process.env.NODE_ENV  === 'DEV';
const isProd = process.env.NODE_ENV  === 'PROD';

const sourceMap = isProd ? 'source-map' : 'cheap-source-map';
const outputPathRelative = 'dist/';
const staticRelative = 'static/';
const staticPath = path.resolve(__dirname, staticRelative);
const distPath = path.join(staticPath, outputPathRelative);
const assetPath = path.join(staticPath, "assets");
const manifestPath = 'manifest.json';  //if you change this, you also need to update it in the app.js
const outputname =isDev ? '[name].js' : '[name].[hash].js';
const cssName = isDev ? '[name].css' :  '[name].[contenthash].css';
const publicPath = isDev ? '/' + outputPathRelative :  outputPathRelative;
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
    }]),
    new webpack.ProvidePlugin({
        $: 'jquery',
        jQuery: 'jquery'
    }),
    new ExtractTextPlugin(cssName),
    manifestPlugin,
    new webpack.IgnorePlugin(/^((fs)|(path)|(os)|(crypto)|(source-map-support))$/, /vs\/language\/typescript\/lib/)
];


//treating people who write make as prod like, with less accurate source maps.
//unsure if this is the right thing, but its a start
if(isProd) {
    plugins.push(new UglifyJsPlugin({
        sourceMap: true,
        parallel: true
    }));
}


module.exports = [
    //if you change the order of this, make sure to update the config variable in app.js
    //server side stuff
    // currently we just want to shove a cache path onto the static assets so we can get the hash onto the filename 
    //this means we can set the cache for eternity 
    {
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
        
        entry: {
            "editor.worker": 'monaco-editor/esm/vs/editor/editor.worker.js',
            "main": './static/main.js'
        },
        output: {
            filename: outputname,
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
        stats: "errors-only",
        devtool: sourceMap,
        module: {
            rules: [
                {
                    test: /\.css$/,
                    exclude: path.resolve(__dirname, 'static/themes/'),
                    use: ExtractTextPlugin.extract({
                        fallback: 'style-loader',
                        use: 'css-loader',
                        publicPath: './'
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
                    test: /\.(html)$/,
                    loader: 'html-loader',
                    options: {
                        minimize: !isDev
                    }
                }
            ]},
            plugins: plugins
        },

    ];
