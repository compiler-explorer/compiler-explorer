// Copyright (c) 2023, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import express from 'express';
import _ from 'underscore';

import {BuildSystems} from '../../shared/build-systems.js';
import {isString, unique} from '../../shared/common-utils.js';
import {
    CompilerInfo,
    DEDUPABLE_COMPILER_FIELDS,
    DedupableCompilerField,
    DedupedCompilerInfo,
    DedupedCompilerList,
} from '../../types/compiler.interfaces.js';
import {Language, LanguageKey} from '../../types/languages.interfaces.js';
import {assert, unwrap, unwrapString} from '../assert.js';
import {ClientStateNormalizer} from '../clientstate-normalizer.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {IExecutionEnvironment} from '../execution/execution-env.interfaces.js';
import {LocalExecutionEnvironment} from '../execution/index.js';
import {logger} from '../logger.js';
import {ClientOptionsHandler, VersionInfo} from '../options-handler.js';
import {PropertyGetter} from '../properties.interfaces.js';
import {SentryCapture} from '../sentry.js';
import {BaseShortener, getShortenerTypeByKey} from '../shortener/index.js';
import {StorageBase} from '../storage/index.js';
import {CompileHandler} from './compile.js';

function methodNotAllowed(req: express.Request, res: express.Response) {
    res.status(405).send('Method Not Allowed');
}

/** The full, legacy shape returned by {@link ApiHandler.getLibrariesAsArray} when no `fields` filter is applied. */
type LibraryArrayEntry = {
    id: string;
    name: string;
    description: string;
    url: string;
    versions: Array<VersionInfo & {id: string}>;
};

export class ApiHandler {
    public compilers: CompilerInfo[] = [];
    public languages: Partial<Record<LanguageKey, Language>> = {};
    private usedLangIds: LanguageKey[] = [];
    private options: ClientOptionsHandler | null = null;
    public readonly handle: express.Router;
    public readonly shortener: BaseShortener;
    public release = {
        gitReleaseName: '',
        releaseBuildNumber: '',
    };
    private readonly compilationEnvironment: CompilationEnvironment;
    /** Per dedupable field, a map from every value seen in {@link compilers} to the first value with that content. */
    private readonly canonicalDedupeValues = new Map<DedupableCompilerField, Map<unknown, unknown>>();

    constructor(
        public readonly compileHandler: CompileHandler,
        ceProps: PropertyGetter,
        private readonly storageHandler: StorageBase,
        urlShortenService: string,
        compilationEnvironment: CompilationEnvironment,
    ) {
        this.handle = express.Router();
        this.compilationEnvironment = compilationEnvironment;
        const cacheHeader = `public, max-age=${ceProps('apiMaxAgeSecs', 24 * 60 * 60)}`;
        this.handle.use((_, res, next) => {
            res.header('Cache-Control', cacheHeader);
            next();
        });
        this.handle.route('/compilers').get(this.handleCompilers.bind(this)).all(methodNotAllowed);

        this.handle.route('/compilers/:language').get(this.handleCompilers.bind(this)).all(methodNotAllowed);

        this.handle.route('/languages').get(this.handleLanguages.bind(this)).all(methodNotAllowed);

        this.handle.route('/libraries/:language').get(this.handleLangLibraries.bind(this)).all(methodNotAllowed);

        this.handle.route('/libraries').get(this.handleAllLibraries.bind(this)).all(methodNotAllowed);

        this.handle.route('/tools/:language').get(this.handleLangTools.bind(this)).all(methodNotAllowed);

        this.handle
            .route('/asm/:opcode')
            .get((req, res) => res.redirect(`amd64/${req.params.opcode}`))
            .all(methodNotAllowed);

        const maxUploadSize = ceProps('maxUploadSize', '1mb');
        const textParser = express.text({limit: ceProps('bodyParserLimit', maxUploadSize), type: () => true});

        this.handle
            .route('/compiler/:compiler/compile')
            .post(textParser, compileHandler.handle.bind(compileHandler))
            .all(methodNotAllowed);
        this.handle
            .route('/compiler/:compiler/build/:buildSystem')
            .post(compileHandler.handleBuildProject.bind(compileHandler))
            .all(methodNotAllowed);
        // The original, CMake-only spelling of the above. Documented API, so it stays.
        this.handle
            .route('/compiler/:compiler/cmake')
            .post(compileHandler.handleCmake.bind(compileHandler))
            .all(methodNotAllowed);

        if (this.compilationEnvironment.ceProps('localexecutionEndpoint', false)) {
            this.handle.route('/localexecution/:hash').post(this.handleLocalExecution.bind(this)).all(methodNotAllowed);
        }

        this.handle
            .route('/popularArguments/:compiler')
            .post(compileHandler.handlePopularArguments.bind(compileHandler))
            .get(compileHandler.handlePopularArguments.bind(compileHandler))
            .all(methodNotAllowed);
        this.handle
            .route('/optimizationArguments/:compiler')
            .post(compileHandler.handleOptimizationArguments.bind(compileHandler))
            .get(compileHandler.handleOptimizationArguments.bind(compileHandler))
            .all(methodNotAllowed);
        this.handle.route('/shortlinkinfo/:id').get(this.shortlinkInfoHandler.bind(this)).all(methodNotAllowed);

        const shortenerType = getShortenerTypeByKey(urlShortenService);
        this.shortener = new shortenerType(storageHandler);
        this.handle.route('/shortener').post(this.shortener.handle.bind(this.shortener)).all(methodNotAllowed);

        this.handle.route('/version').get(this.handleReleaseName.bind(this)).all(methodNotAllowed);
        this.handle.route('/releaseBuild').get(this.handleReleaseBuild.bind(this)).all(methodNotAllowed);
        // Let's not document this one, eh?
        this.handle.route('/forceServerError').get((req, res) => {
            logger.error(`Forced server error from ${req.ip}`);
            throw new Error('Forced server error');
        });
    }

    shortlinkInfoHandler(req: express.Request, res: express.Response, next: express.NextFunction) {
        const id = unwrapString(req.params.id);
        this.storageHandler
            .expandId(id)
            .then(result => {
                const config = JSON.parse(result.config);

                if (result.created) res.header('Link-Created', result.created.toUTCString());

                if (config.content) {
                    const normalizer = new ClientStateNormalizer();
                    normalizer.fromGoldenLayout(config);

                    res.send(normalizer.normalized);
                } else {
                    res.send(config);
                }
            })
            .catch(err => {
                logger.warn(`Exception thrown when expanding ${id}: `, err);
                logger.warn('Exception value:', err);
                SentryCapture(err, 'shortlinkInfoHandler');
                next({
                    statusCode: 404,
                    message: `ID "${id}" could not be found`,
                });
            });
    }

    handleLanguages(req: express.Request, res: express.Response) {
        this.outputList(this.getAvailableLanguages(), 'Id', req, res);
    }

    getAvailableLanguages(): Language[] {
        // Always expose the build system manifest languages to the frontend even if no compiler reports them as its
        // language, so IDE/tree mode (CMakeLists.txt, Cargo.toml) can resolve them.
        const manifestLangIds = Object.values(BuildSystems)
            .map(buildSystem => buildSystem.manifestLanguageId)
            .filter(langId => this.languages[langId]);
        const langIds = unique([...this.usedLangIds, ...manifestLangIds]);
        return langIds.map(val => {
            const lang = this.languages[val];
            const newLangObj: Language = Object.assign({}, lang);
            if (this.options) {
                newLangObj.defaultCompiler = this.options.options.defaultCompiler[unwrap(lang).id];
                newLangObj.defaultLibs = this.options.options.defaultLibs[unwrap(lang).id];
            }
            return newLangObj;
        });
    }

    /** Resolve the default compiler id for a language, or undefined if none configured. */
    getDefaultCompilerFor(languageId: LanguageKey): string | undefined {
        return this.options?.options.defaultCompiler[languageId];
    }

    filterCompilerProperties(list: CompilerInfo[] | Language[], selectedFields: string[]) {
        return list.map(compiler => {
            return _.pick(compiler, selectedFields);
        });
    }

    /**
     * The supported fields named by a `?dedupe=` request, or an empty array if the client didn't ask for any. An
     * empty result means the response keeps its historic bare-array shape.
     */
    private parseDedupeFields(dedupe: unknown, allowed: boolean): DedupableCompilerField[] {
        if (!allowed || !isString(dedupe)) return [];
        return DEDUPABLE_COMPILER_FIELDS.filter(field => dedupe.split(',').includes(field));
    }

    /**
     * Values of a dedupable field repeat verbatim across compilers (the `toolchain` override alone is one 63KB
     * object shared by ~150 compilers), so serialising each compiler's copy dominates the response. Interning them
     * by content is only worth doing once per compiler list, hence the cache.
     */
    private getCanonicalDedupeValues(field: DedupableCompilerField): Map<unknown, unknown> {
        let canonical = this.canonicalDedupeValues.get(field);
        if (!canonical) {
            canonical = new Map<unknown, unknown>();
            const byContent = new Map<string, unknown>();
            for (const compiler of this.compilers) {
                for (const value of compiler[field] ?? []) {
                    const content = JSON.stringify(value);
                    if (!byContent.has(content)) byContent.set(content, value);
                    canonical.set(value, byContent.get(content));
                }
            }
            this.canonicalDedupeValues.set(field, canonical);
        }
        return canonical;
    }

    private dedupeList(list: Record<string, unknown>[], fields: DedupableCompilerField[]): DedupedCompilerList {
        const tables = fields.map(field => ({
            field,
            table: [] as unknown[],
            canonical: this.getCanonicalDedupeValues(field),
            indices: new Map<unknown, number>(),
        }));

        const compilers = list.map(entry => {
            const deduped: Record<string, unknown> = {...entry};
            for (const {field, table, canonical, indices} of tables) {
                const values = entry[field];
                if (!Array.isArray(values)) continue;
                deduped[field] = values.map(value => {
                    const shared = canonical.get(value) ?? value;
                    let index = indices.get(shared);
                    if (index === undefined) {
                        index = table.length;
                        table.push(shared);
                        indices.set(shared, index);
                    }
                    return index;
                });
            }
            return deduped as DedupedCompilerInfo;
        });

        const refs = Object.fromEntries(tables.map(({field, table}) => [field, table])) as DedupedCompilerList['refs'];
        return {compilers, refs};
    }

    outputList(
        list: CompilerInfo[] | Language[],
        title: string,
        req: express.Request,
        res: express.Response,
        dedupable = false,
    ) {
        if (req.accepts(['text', 'json']) === 'json') {
            if (req.query.fields === 'all') {
                res.send(list);
            } else {
                const defaultfields = [
                    'id',
                    'name',
                    'lang',
                    'compilerType',
                    'semver',
                    'extensions',
                    'monaco',
                    'instructionSet',
                ];
                let fields = defaultfields;
                if (req.query.fields) {
                    assert(isString(req.query.fields));
                    fields = req.query.fields.split(',');
                }
                const filteredList = this.filterCompilerProperties(list, fields);
                const dedupeFields = this.parseDedupeFields(req.query.dedupe, dedupable);
                res.send(
                    dedupeFields.length > 0
                        ? this.dedupeList(filteredList as Record<string, unknown>[], dedupeFields)
                        : filteredList,
                );
            }
            return;
        }

        const maxLength = Math.max(
            ...list
                .map(item => item.id)
                .concat([title])
                .map(item => item.length),
        );
        const header = title.padEnd(maxLength, ' ') + ' | Name\n';
        const body = list.map(lang => lang.id.padEnd(maxLength, ' ') + ' | ' + lang.name).join('\n');
        res.set('Content-Type', 'text/plain');
        res.send(header + body);
    }

    // Overload order matters: `ReturnType<ApiHandler['getLibrariesAsArray']>` (used by the MCP
    // tools, which always call without `fields`) resolves to the *last* overload, so the full
    // non-partial shape must come last.
    getLibrariesAsArray(languageId: LanguageKey, fields: string[] | undefined): Partial<LibraryArrayEntry>[];
    getLibrariesAsArray(languageId: LanguageKey): LibraryArrayEntry[];
    getLibrariesAsArray(
        languageId: LanguageKey,
        fields?: string[],
    ): LibraryArrayEntry[] | Partial<LibraryArrayEntry>[] {
        const libsForLanguageObj = unwrap(this.options).options.libs[languageId];
        if (!libsForLanguageObj) return [];

        // Field filtering supports dotted paths for nested version fields
        // (e.g. `versions.id,versions.version`). Bare `versions` includes the
        // full version objects. Omitting `fields` returns the legacy full shape.
        const VERSION_PREFIX = 'versions.';
        let topLevelFields: string[] | undefined;
        let versionFields: string[] | undefined;
        if (fields && fields.length > 0) {
            topLevelFields = [];
            versionFields = [];
            let allVersionFields = false;
            for (const f of fields) {
                if (f.startsWith(VERSION_PREFIX)) {
                    versionFields.push(f.slice(VERSION_PREFIX.length));
                } else if (f === 'versions') {
                    allVersionFields = true;
                    topLevelFields.push(f);
                } else {
                    topLevelFields.push(f);
                }
            }
            if (allVersionFields) {
                versionFields = undefined;
            } else if (versionFields.length > 0 && !topLevelFields.includes('versions')) {
                topLevelFields.push('versions');
            }
        }

        // Build the full, concretely-typed shape first so callers that omit
        // `fields` (e.g. the MCP tools) get the legacy non-partial type back.
        const fullList: LibraryArrayEntry[] = Object.keys(libsForLanguageObj).map(key => {
            const library = libsForLanguageObj[key];
            const versions = Object.keys(library.versions).map(versionKey => ({
                ...library.versions[versionKey],
                id: versionKey,
            }));
            return {
                id: key,
                name: library.name,
                description: library.description,
                url: library.url,
                versions,
            };
        });

        if (!topLevelFields && !versionFields) return fullList;

        return fullList.map(fullLib => {
            const lib = versionFields
                ? {...fullLib, versions: fullLib.versions.map(v => _.pick(v, versionFields!))}
                : fullLib;
            return topLevelFields ? _.pick(lib, topLevelFields) : lib;
        }) as Partial<LibraryArrayEntry>[];
    }

    getToolsAsArray(languageId: LanguageKey) {
        const toolsForLanguageObj = unwrap(this.options).options.tools[languageId];
        if (!toolsForLanguageObj) return [];

        return Object.keys(toolsForLanguageObj).map(key => {
            const tool = toolsForLanguageObj[key];
            return {
                id: key,
                name: tool.tool.name,
                type: tool.type,
                languageId: tool.tool.languageId || languageId,
                allowStdin: tool.tool.stdinHint !== 'disabled',
                args: tool.tool.args,
                monacoStdin: tool.tool.monacoStdin,
                icon: tool.tool.icon,
                darkIcon: tool.tool.darkIcon,
                stdinHint: tool.tool.stdinHint,
            };
        });
    }

    handleLangLibraries(req: express.Request, res: express.Response, next: express.NextFunction) {
        if (this.options) {
            if (req.params.language) {
                const fieldsParam = req.query.fields;
                const fields = isString(fieldsParam) ? fieldsParam.split(',') : undefined;
                res.send(this.getLibrariesAsArray(req.params.language as LanguageKey, fields));
            } else {
                next({
                    statusCode: 404,
                    message: 'Language is required',
                });
            }
        } else {
            next({
                statusCode: 500,
                message: 'Internal error',
            });
        }
    }

    handleLangTools(req: express.Request, res: express.Response, next: express.NextFunction) {
        if (this.options) {
            if (req.params.language) {
                res.send(this.getToolsAsArray(req.params.language as LanguageKey));
            } else {
                next({
                    statusCode: 404,
                    message: 'Language is required',
                });
            }
        } else {
            next({
                statusCode: 500,
                message: 'Internal error',
            });
        }
    }

    async handleLocalExecution(req: express.Request, res: express.Response, next: express.NextFunction) {
        if (!req.params.hash) {
            next({statusCode: 404, message: 'No hash supplied'});
            return;
        }

        if (!req.body.ExecutionParams) {
            next({statusCode: 404, message: 'No ExecutionParams'});
            return;
        }

        try {
            const env: IExecutionEnvironment = new LocalExecutionEnvironment(this.compilationEnvironment);
            await env.downloadExecutablePackage(unwrapString(req.params.hash));
            const execResult = await env.execute(req.body.ExecutionParams);
            logger.debug('execResult', execResult);
            res.send(execResult);
        } catch (e) {
            logger.error(e);
            next({statusCode: 500, message: 'Internal error'});
        }
    }

    handleAllLibraries(req: express.Request, res: express.Response, next: express.NextFunction) {
        if (this.options) {
            res.send(this.options.options.libs);
        } else {
            next({
                statusCode: 500,
                message: 'Internal error',
            });
        }
    }

    handleCompilers(req: express.Request, res: express.Response) {
        let filteredCompilers = this.compilers;
        if (req.params.language) {
            filteredCompilers = this.compilers.filter(compiler => compiler.lang === req.params.language);
        }

        // Avoid leaking nested tool fields like `tool.exe` over the API: expose
        // only the supported tool IDs. Per-tool metadata is served separately
        // via /api/tools/:language.
        const slimmedCompilers = filteredCompilers.map(compiler =>
            compiler.tools
                ? {
                      ...compiler,
                      tools: (Array.isArray(compiler.tools)
                          ? compiler.tools
                          : Object.keys(compiler.tools)) as unknown as CompilerInfo['tools'],
                  }
                : compiler,
        );

        this.outputList(slimmedCompilers, 'Compiler Name', req, res, true);
    }

    handleReleaseName(req: express.Request, res: express.Response) {
        res.send(this.release.gitReleaseName);
    }

    handleReleaseBuild(req: express.Request, res: express.Response) {
        res.send(this.release.releaseBuildNumber);
    }

    setCompilers(compilers: CompilerInfo[]) {
        this.compilers = compilers;
        this.canonicalDedupeValues.clear();
        this.usedLangIds = unique(this.compilers.map(compiler => compiler.lang));
    }

    setLanguages(languages: Partial<Record<LanguageKey, Language>>) {
        this.languages = languages;
    }

    setOptions(options: ClientOptionsHandler) {
        this.options = options;
    }

    setReleaseInfo(gitReleaseName: string | undefined, releaseBuildNumber: string | undefined) {
        this.release = {
            gitReleaseName: gitReleaseName || '',
            releaseBuildNumber: releaseBuildNumber || '',
        };
    }
}
