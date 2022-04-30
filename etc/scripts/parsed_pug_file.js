import {execSync} from 'child_process';
import {getHashDigest} from 'loader-utils';
import * as pug from 'pug';

function execGit(command) {
    const gitResult = execSync(command);
    if (!gitResult) {
        throw new Error(`Failed to execute ${command}`);
    }
    return gitResult.toString();
}

const gitChanges = execGit('git log --date=local --after="3 months ago" "--grep=(#[0-9]*)" --oneline')
    .split('\n')
    .map(line => line.match(/(?<hash>\w+) (?<description>.*)/))
    .filter(x => x)
    .map(match => match.groups);

export default function(content) {
    const filename = this.resourcePath;
    const lastTime = execGit(`git log -1 --format=%cd "${filename}"`).trimEnd();
    const lastCommit = execGit(`git log -1 --format=%h "${filename}"`).trimEnd();
    const compiled = pug.compile(content.toString(), {filename});
    const source = compiled({gitChanges, lastTime, lastCommit});
    const result = {
        hash: getHashDigest(source, 'sha256', 'hex', 16),
        text: source.toString(),
    };
    return `export default ${JSON.stringify(result)};`;
}
