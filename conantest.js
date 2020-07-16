const buildenv = require("./lib/buildenvsetup/buildenvsetup-ceconan");

const compilerInfo = {
    id: "clang800",
    compilerType: "clang",
    exe: "/opt/compiler-explorer/clang-8.0.0/bin/clang-g++",
    options: "",
    buildenvsetup: {
        props: function (key, def) {
            if (key == "host") {
                return "http://ec2-54-93-113-179.eu-central-1.compute.amazonaws.com:10240";
            }
            return def;
        }
    }
};

const env = {
};

// const key = {
//     options: ["--target=armv8"]
// };
const key = {
    options: []
};

const dirPath = "/tmp/hello";
const libraryDetails = {
    "catch2": {
        "staticliblink": ["Catch2"],
        "libpath": [],
        "version": "3.0.0-preview2"
    },
    "fmt": {
        "staticliblink": ["fmtd"],
        "libpath": [],
        "version": "6.2.0"
    }
};

const benv = new buildenv(compilerInfo, env);
benv.setup(key, dirPath, libraryDetails).then(() => {
    console.log("done");
}).catch(e => {
    console.error(e);
});
