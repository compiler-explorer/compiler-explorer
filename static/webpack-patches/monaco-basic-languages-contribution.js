// Patched by CE build: stripped editor contribution side-effect imports
// that duplicate what MonacoEditorWebpackPlugin already includes.
// Upstream bug: https://github.com/microsoft/monaco-editor/issues/5162
// CE issue: https://github.com/compiler-explorer/compiler-explorer/issues/8547

import { languages } from '../editor/editor.api2.js';

const languageDefinitions = {};
const lazyLanguageLoaders = {};
class LazyLanguageLoader {
  static getOrCreate(languageId) {
    if (!lazyLanguageLoaders[languageId]) {
      lazyLanguageLoaders[languageId] = new LazyLanguageLoader(languageId);
    }
    return lazyLanguageLoaders[languageId];
  }
  constructor(languageId) {
    this._languageId = languageId;
    this._loadingTriggered = false;
    this._lazyLoadPromise = new Promise((resolve, reject) => {
      this._lazyLoadPromiseResolve = resolve;
      this._lazyLoadPromiseReject = reject;
    });
  }
  load() {
    if (!this._loadingTriggered) {
      this._loadingTriggered = true;
      languageDefinitions[this._languageId].loader().then(
        (mod) => this._lazyLoadPromiseResolve(mod),
        (err) => this._lazyLoadPromiseReject(err)
      );
    }
    return this._lazyLoadPromise;
  }
}
function registerLanguage(def) {
  const languageId = def.id;
  languageDefinitions[languageId] = def;
  languages.register(def);
  const lazyLanguageLoader = LazyLanguageLoader.getOrCreate(languageId);
  languages.registerTokensProviderFactory(languageId, {
    create: async () => {
      const mod = await lazyLanguageLoader.load();
      return mod.language;
    }
  });
  languages.onLanguageEncountered(languageId, async () => {
    const mod = await lazyLanguageLoader.load();
    languages.setLanguageConfiguration(languageId, mod.conf);
  });
}

export { registerLanguage };
