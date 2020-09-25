module.exports = {
    CppDemangler: require('./cpp').Demangler,
    DefaultDemangler: require('./default').Demangler,
    PascalDemangler: require('./pascal').Demangler,
    Win32Demangler: require('./win32').Demangler,
};
