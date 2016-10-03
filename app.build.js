({
    appDir: "static",
    baseUrl: ".",
    dir: "out/dist",
    generateSourceMaps: true,
    preserveLicenseComments: false,
    optimize: "uglify2",
    removeCombined: true,
    useStrict: true,
    mainConfigFile: "static/main.js",
    skipDirOptimize: true,
    optimizeCss: "standard",
    modules: [
        {
            name: "main"
        }
    ]
})
