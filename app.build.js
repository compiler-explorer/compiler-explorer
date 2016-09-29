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
    modules: [
        {
            name: "main"
        }
    ]
})
