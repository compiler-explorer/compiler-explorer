import * as monaco from 'monaco-editor';
import * as rust from 'monaco-editor/esm/vs/basic-languages/rust/rust';

monaco.languages.register({id: 'doolang'});
monaco.languages.setLanguageConfiguration('doolang', rust.conf);
monaco.languages.setMonarchTokensProvider('doolang', rust.language);

export default rust.language;
