import _ from 'underscore';
const options = require('./options').options;

export class LibUtils {
    private copyAndFilterLibraries(allLibraries, filter) {
        const filterLibAndVersion = _.map(filter, (lib) => {
            const match = lib.match(/([\w-]*)\.([\w-]*)/i);
            if (match) {
                return {
                    id: match[1],
                    version: match[2],
                };
            } else {
                return {
                    id: lib,
                    version: false,
                };
            }
        });

        const filterLibIds = new Set();
        _.each(filterLibAndVersion, (lib) => {
            filterLibIds.add(lib.id);
        });

        const copiedLibraries = {};
        _.each(allLibraries, (lib, libid) => {
            if (!filterLibIds.has(libid)) return;

            const libcopy = Object.assign({}, lib);
            libcopy.versions = _.omit(lib.versions, (version, versionid) => {
                for (const filter of filterLibAndVersion) {
                    if (filter.id === libid) {
                        if (!filter.version) return false;
                        if (filter.version === versionid) return false;
                    }
                }

                return true;
            });

            copiedLibraries[libid] = libcopy;
        });

        return copiedLibraries;
    }

    public getSupportedLibraries(supportedLibrariesArr, langId) {
        const allLibs = options.libs[langId];
        if (supportedLibrariesArr && supportedLibrariesArr.length > 0) {
            return this.copyAndFilterLibraries(allLibs, supportedLibrariesArr);
        }
        return allLibs;
    }
}
